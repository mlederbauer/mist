""" base.py

This file will contain all common operations for the models
including certain superclasses of models

"""
import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from mist.data.datasets import SpectraMolDataset, SpecDataModule
from mist import utils

# Define model_types
model_registry = {}


def register_model(cls):
    """register_model.

    Add an argument to the model_registry.
    Use this as a decorator on classes defined in the rest of the directory.

    """
    model_registry[cls.__name__] = cls
    return cls


def get_model_class(model, **kwargs):
    return model_registry[model]


def build_model(model, **kwargs):
    """build_model"""
    return get_model_class(model)(**kwargs)


class TorchModel(pl.LightningModule, ABC):
    """TorchModel.

    Parent class to hold SpectraModels. Subclasses (MistNet, FingerIDFFN,
    ContrastiveModel, FingerIDXFormer) must implement encode_spectra,
    encode_mol, mol_features, training_step, validation_step, and test_step
    -- this was already the de facto contract every subclass honored, now
    enforced at class-definition time instead of failing at call time.

    spec_features, compute_loss, and dataset_type are deliberately NOT
    abstract: ContrastiveModel wraps another TorchModel (self.main_model)
    and delegates to it for spec_features/compute_loss rather than
    implementing them itself, so those can't be a hard requirement here.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        lr_decay_frac: float = 0.99,
        scheduler: bool = False,
        lr_decay_time: int = 10000,
        **kwargs,
    ):
        """Call init"""
        pl.LightningModule.__init__(self)
        self.learning_rate = learning_rate
        self.results_dir = ""
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.lr_decay_time = lr_decay_time
        self.lr_decay_frac = lr_decay_frac

    @staticmethod
    def dataset_type(mode: Optional[str] = None) -> str:
        """dataset_type."""
        return "default"

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        ## Scheduler
        if not self.scheduler:
            return optimizer
        else:
            scheduler = build_lr_scheduler(
                optimizer,
                lr_decay_frac=self.lr_decay_frac,
                decay_steps=self.lr_decay_time,
                warmup=100,
            )
            ret = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "frequency": 1,
                    "interval": "step",
                },
            }
            return ret

    def forward(self, batch):
        raise NotImplementedError()

    @abstractmethod
    def encode_spectra(self, batch: dict, **kwargs) -> Tuple[torch.Tensor, dict]:
        """Encode a batch of spectra into (embedding, aux_outputs)."""

    @abstractmethod
    def encode_mol(self, batch: dict, **kwargs) -> Tuple[torch.Tensor, dict]:
        """Encode a batch of molecules into (embedding, aux_outputs)."""

    @staticmethod
    @abstractmethod
    def mol_features(mode: Optional[str] = None) -> str:
        """Name of the molecule featurization this model expects."""

    def set_results_dir(self, dir_):
        """Set the results dir to store misc info"""
        self.results_dir = dir_

    def batch_to_device(self, batch: dict) -> None:
        """batch_to_device.

        Convert batch tensors to same device as the model


        Args:
            batch (dict): Batch from data loader

        """
        # Port to cuda
        device = self.device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = batch[key].to(device)

    def encode_all_spectras(
        self, spectras: SpectraMolDataset, no_grad=False, logits=False, **kwargs
    ) -> torch.tensor:
        """encode_all_spectras."""
        with torch.set_grad_enabled(not no_grad):
            spectra_loader = SpecDataModule.get_paired_loader(
                spectras, shuffle=False, **kwargs
            )
            spectra_outputs = []
            for spectra_batch in spectra_loader:
                self.batch_to_device(spectra_batch)

                # Convert to cpu!
                if not logits:
                    model_encoding = self.encode_spectra(spectra_batch)[0].cpu()
                else:
                    model_encoding = self.encode_spectra(
                        spectra_batch,
                        logits=True,
                    )[0].cpu()

                spectra_outputs.append(model_encoding)

            stacked_spectra = torch.cat(spectra_outputs, 0)
        return stacked_spectra

    def encode_all_mols(
        self, mol_library: SpectraMolDataset, no_grad=False, **kwargs
    ) -> torch.tensor:
        """encode_all_mols."""

        # Modify to use only mol  loader

        with torch.set_grad_enabled(not no_grad):
            mol_loader = SpecDataModule.get_mol_loader(
                mol_library, shuffle=False, **kwargs
            )
            mol_outputs = []
            for mol_batch in mol_loader:
                self.batch_to_device(mol_batch)

                # Convert to cpu!
                model_encoding = self.encode_mol(mol_batch)[0].cpu()
                mol_outputs.append(model_encoding)

            stacked_mols = torch.Tensor([])
            if len(mol_outputs) > 0:
                stacked_mols = torch.cat(mol_outputs, 0)
        return stacked_mols

    @abstractmethod
    def training_step(self, batch, batch_idx):
        """Compute and return the training loss for one batch."""

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        """Compute and log the validation loss for one batch."""

    @abstractmethod
    def test_step(self, batch, batch_idx):
        """Compute and log the test loss for one batch."""

    def post_train_modify(self, _checkpoint_callback, debug=False, **kwargs):
        """On fit end, load the best checkpoint"""

        debug_overfit = debug == "test_overfit"
        if debug_overfit:
            logging.info("Not loading best model")
            return

        # Load from checkpoint
        best_checkpoint = _checkpoint_callback.best_model_path

        best_checkpoint_score = None
        if _checkpoint_callback.best_model_score is not None:
            best_checkpoint_score = _checkpoint_callback.best_model_score.item()

        loaded_checkpoint = torch.load(best_checkpoint)
        best_epoch = loaded_checkpoint["epoch"]

        logging.info(
            f"Loading from epoch {best_epoch} with val loss of {best_checkpoint_score}"
        )

        # model = model.load_from_checkpoint(best_checkpoint)
        self.load_state_dict(loaded_checkpoint["state_dict"])

    def load_from_ckpt(self, ckpt_file, **kwargs):

        loaded_checkpoint = torch.load(ckpt_file)
        best_epoch = loaded_checkpoint["epoch"]

        logging.info(f"Loading from epoch {best_epoch} (strict=False)")

        # model = model.load_from_checkpoint(best_checkpoint)
        self.load_state_dict(loaded_checkpoint["state_dict"], strict=False)

    def train_model(
        self,
        module,
        save_dir,
        max_epochs=None,
        gradient_clip_val=5,
        min_epochs=None,
        gpus=0,
        gpu_index: int = None,
        log_version=None,
        log_name="",
        patience: int = 20,
        tune: bool = False,
        prog_bars: bool = True,
        tune_save: bool = False,
        debug: str = None,
        wandb_project: str = None,
        wandb_entity: str = None,
        wandb_run_name: str = None,
        **kwargs,
    ) -> List[dict]:
        """_summary_

        Args:
            module (_type_): _description_
            save_dir (_type_): _description_
            max_epochs (_type_, optional): _description_. Defaults to None.
            gradient_clip_val (int, optional): _description_. Defaults to 5.
            min_epochs (_type_, optional): _description_. Defaults to None.
            gpus (int, optional): _description_. Defaults to 0.
            log_version (_type_, optional): _description_. Defaults to None.
            log_name (str, optional): _description_. Defaults to "".
            patience (int, optional): _description_. Defaults to 20.
            tune (bool, optional): _description_. Defaults to False.
            prog_bars (bool, optional): _description_. Defaults to True.
            tune_save (bool, optional): _description_. Defaults to False.
            debug (str, optional): _description_. Defaults to None.

        Returns:
            List[dict]: _description_
        """
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir, name=log_name, version=log_version
        )

        tb_path = tb_logger.log_dir
        self.set_results_dir(tb_path)

        monitor = "val_loss"
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            dirpath=tb_path,
            filename="best",
            save_weights_only=True,
        )
        # Full trainer state (optimizer, scheduler, epoch, global step) so a
        # preempted/requeued job can resume training exactly where it left
        # off, rather than restarting at epoch 0. save_last keeps this
        # updated to the most recent epoch regardless of val_loss.
        last_ckpt_path = Path(tb_path) / "last.ckpt"
        last_checkpoint_callback = ModelCheckpoint(
            dirpath=tb_path,
            save_last=True,
            save_top_k=0,
            save_weights_only=False,
        )
        callbacks = []
        loggers = [tb_logger]

        # wandb is only enabled for a single (non-hyperopt) training run.
        # Concurrent hyperopt trials (n_jobs>1) calling wandb.init()
        # simultaneously can time out contending for wandb's shared
        # background service process; not worth the complexity to fix for
        # what's a nice-to-have during search, so hyperopt trials stick to
        # TensorBoard + the Optuna study/results table instead.
        if wandb_project is not None and not tune:
            wandb_logger = pl_loggers.WandbLogger(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name or log_version or log_name,
                save_dir=save_dir,
            )
            loggers.append(wandb_logger)

        val_check_interval = 1.0

        if not tune:
            callbacks.append(checkpoint_callback)
            callbacks.append(last_checkpoint_callback)
            console_logger = utils.ConsoleLogger()
            loggers.append(console_logger)
        elif tune_save:
            callbacks.append(checkpoint_callback)

        # Set results dir
        earlystop_callback = EarlyStopping(monitor="val_loss", patience=patience)
        callbacks.append(earlystop_callback)

        # Add in LR Monitor
        callbacks.append(LearningRateMonitor(logging_interval="step"))
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gradient_clip_val=gradient_clip_val,
            min_epochs=min_epochs,
            accelerator="gpu" if gpus >= 1 else None,
            devices=[gpu_index] if gpu_index is not None else (gpus if gpus >= 1 else None),
            logger=loggers,
            enable_progress_bar=prog_bars,
            reload_dataloaders_every_n_epochs=1,
            callbacks=callbacks,
            val_check_interval=val_check_interval,
            enable_checkpointing=False if (tune and not tune_save) else True,
        )
        resume_ckpt = str(last_ckpt_path) if last_ckpt_path.exists() else None
        if resume_ckpt:
            logging.info(f"Resuming training from {resume_ckpt}")
        trainer.fit(self, module, ckpt_path=resume_ckpt)

        if tune:
            val_loss = trainer.callback_metrics.get("val_loss")
            return float(val_loss.item()) if val_loss is not None else float("inf")

        # Modify the model after fit with callbacks as inputs
        # This involves loading the model frorm the best checkpoint
        self.post_train_modify(_checkpoint_callback=checkpoint_callback, **kwargs)

        # Test
        # List of losses
        test_loss = trainer.test(self, module.test_dataloader())
        logging.info(test_loss)
        test_losses = test_loss

        # Turn on eval mode
        self.eval()
        return test_losses


def build_lr_scheduler(
    optimizer, lr_decay_frac: float, decay_steps: int = 10000, warmup: int = 100
):
    """_summary_

    Args:
        optimizer (_type_): _description_
        lr_decay_frac (float): _description_
        decay_steps (int, optional): _description_. Defaults to 10000.
        warmup (int, optional): _description_. Defaults to 100.
    """

    def lr_lambda(step):
        if step >= warmup:
            # Adjust
            step = step - warmup
            rate = lr_decay_frac ** (step // decay_steps)
        else:
            rate = 1 - math.exp(-step / warmup)
        return rate

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler
