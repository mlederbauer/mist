""" base_hyperopt.py

Abstract away common hyperopt functionality. Drives a plain Optuna study
(no Ray Tune -- Ray's dashboard subprocess segfaults on startup on this
cluster due to a protobuf version conflict with ray-lightning's pin, and
its distributed trial orchestration isn't needed for a single-node search).

"""
import logging
import yaml
from pathlib import Path
from typing import Callable

import optuna
import pytorch_lightning as pl

import mist.utils as utils


def run_hyperopt(
    kwargs: dict,
    score_function: Callable,
    param_space_function: Callable,
    initial_points: list,
):
    """run_hyperopt.

    Args:
        kwargs: All dictionary args for hyperopt and train
        score_function: score_function(config, base_args, trial_dir) -> val_loss
        param_space_function: Optuna objective-style function(trial) that
            calls trial.suggest_* to build a config
        initial_points: List of initial params to try first
    """
    kwargs["prog_bars"] = False

    if kwargs["debug"]:
        kwargs["num_h_samples"] = 10
        kwargs["max_epochs"] = 5

    save_dir = Path(kwargs["save_dir"]).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    utils.setup_logger(
        str(save_dir), log_name="hyperopt.log", debug=kwargs.get("debug", False)
    )
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    yaml_args = yaml.dump(kwargs)
    logging.info(f"\n{yaml_args}")
    with open(save_dir / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    # SQLite-backed study: a job resubmitted (e.g. after preemption) with the
    # same --save-dir picks up exactly where the study left off, with no
    # separate checkpoint path to track. A trial that was mid-training at
    # the moment of preemption is not resumed mid-epoch -- Optuna marks it
    # incomplete and it's retried fresh.
    study = optuna.create_study(
        storage=f"sqlite:///{save_dir / 'study.db'}",
        study_name="hyperopt",
        direction="minimize",
        load_if_exists=True,
    )
    if not study.trials:
        for params in initial_points:
            study.enqueue_trial(params)

    def objective(trial: optuna.Trial) -> float:
        param_space_function(trial)
        trial_dir = save_dir / f"trial_{trial.number}"
        return score_function(
            trial.params,
            base_args=kwargs,
            trial_dir=trial_dir,
            trial_number=trial.number,
        )

    study.optimize(
        objective,
        n_trials=kwargs.get("num_h_samples"),
        n_jobs=kwargs.get("max_concurrent", 1),
    )

    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if completed:
        output = {"score": study.best_value, "config": study.best_params}
        out_str = yaml.dump(output, indent=2)
        logging.info(out_str)
        with open(save_dir / "best_trial.yaml", "w") as f:
            f.write(out_str)
    else:
        logging.warning("No trials completed successfully; skipping best_trial.yaml")

    study.trials_dataframe().to_csv(save_dir / "full_res_tbl.tsv", sep="\t", index=None)
