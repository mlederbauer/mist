""" build_reaction_metadata.py

Extract reaction metadata (starting materials, provenance) from a reaction-SMILES
source file (e.g. USPTO_FULL.csv) and join it to a paired-spectra dataset's labels.tsv
by InChIKey, producing reaction_metadata.tsv for auxiliary molecular conditioning
(see mist.data.aux_featurizers).

Dataset-scale (millions of RDKit parses) -- run via Slurm, not the login node. See
run_scripts/submit_build_reaction_metadata.sh.

Multi-product rows: a reaction-SMILES row's product side may have more than one
'.'-separated fragment (frequent in USPTO_FULL -- often near-duplicate salts/
protonation states of the reactants, not a single clean product). Every fragment is
matched against the target inchikey set independently; a row can therefore produce
zero, one, or several reaction_metadata.tsv rows (one per matching fragment), all
sharing the same reaction_id but different inchikeys.

candidates is left empty in this pass -- the column exists so a later source/pass
can populate it (e.g. same-patent byproducts, scaffold-similar compounds) without a
schema change. Any future consumer of candidates MUST exclude the training target's
own compound (by inchikey, not string equality) at load time -- that exclusion
belongs in the consumer, not here, so it protects every candidates-populating source.

Usage:
    python -m mist.build_reaction_metadata \\
        --reaction-source /orcd/data/ccoley/001/uspto_data/USPTO_FULL.csv \\
        --source-name USPTO_FULL \\
        --labels-file /orcd/data/ccoley/001/msms_data/nist23/labels.tsv \\
        --out /home/magled/mist/data/nist23/reaction_metadata.tsv
"""
import argparse
import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def smiles_to_inchikey(smiles: str):
    """Return the InChIKey for smiles, or None if it fails to parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def extract_uspto_row(patent_number: str, year, rxn_smiles: str, row_index: int):
    """Parse one USPTO_FULL row into (reaction_id, starting_materials, comments,
    [(product_smiles, product_inchikey), ...]).

    Returns None if rxn_smiles doesn't split into exactly 3 '>'-separated segments
    (reactants>agents>products) -- not expected in USPTO_FULL as inspected, but a
    future data refresh could introduce malformed rows, so this is a skip-and-count
    guard rather than an assumption.
    """
    if not rxn_smiles:
        return None
    segments = rxn_smiles.split(">")
    if len(segments) != 3:
        return None

    reactants_raw, _agents_raw, products_raw = segments
    reactant_smis = [s for s in reactants_raw.split(".") if s]
    product_smis = [s for s in products_raw.split(".") if s]
    if not product_smis:
        return None

    reaction_id = f"{patent_number}_{row_index}"
    starting_materials = ";".join(reactant_smis)
    comments = f"patent={patent_number};year={year}"

    products = []
    for product_smiles in product_smis:
        inchikey = smiles_to_inchikey(product_smiles)
        if inchikey is not None:
            products.append((product_smiles, inchikey))

    return reaction_id, starting_materials, comments, products


def build_from_uspto(
    reaction_source: str,
    source_name: str,
    target_inchikeys: set,
    out_path: Path,
    chunksize: int = 50_000,
):
    """Stream reaction_source in chunks, match products against
    target_inchikeys, and append matching rows to out_path as they're found."""
    today = date.today().isoformat()

    columns = [
        "inchikey",
        "product_smiles",
        "reaction_id",
        "starting_materials",
        "candidates",
        "source",
        "date_added",
        "comments",
    ]

    n_rows = 0
    n_skipped_malformed = 0
    n_fragments_seen = 0
    n_fragments_matched = 0
    matched_inchikeys = set()
    matched_reaction_ids = set()

    wrote_header = False
    per_patent_counter = {}

    for chunk in pd.read_csv(
        reaction_source, chunksize=chunksize, dtype=str, keep_default_na=False
    ):
        out_rows = []
        for patent_number, year, rxn_smiles in zip(
            chunk["PatentNumber"], chunk["Year"], chunk["reactions"]
        ):
            n_rows += 1
            row_index = per_patent_counter.get(patent_number, 0)
            per_patent_counter[patent_number] = row_index + 1

            extracted = extract_uspto_row(patent_number, year, rxn_smiles, row_index)
            if extracted is None:
                n_skipped_malformed += 1
                continue

            reaction_id, starting_materials, comments, products = extracted
            n_fragments_seen += len(products)

            for product_smiles, inchikey in products:
                if inchikey not in target_inchikeys:
                    continue
                n_fragments_matched += 1
                matched_inchikeys.add(inchikey)
                matched_reaction_ids.add(reaction_id)
                out_rows.append(
                    {
                        "inchikey": inchikey,
                        "product_smiles": product_smiles,
                        "reaction_id": reaction_id,
                        "starting_materials": starting_materials,
                        "candidates": "",
                        "source": source_name,
                        "date_added": today,
                        "comments": comments,
                    }
                )

        if out_rows:
            out_df = pd.DataFrame(out_rows, columns=columns)
            out_df.to_csv(
                out_path,
                sep="\t",
                mode="a" if wrote_header else "w",
                header=not wrote_header,
                index=False,
            )
            wrote_header = True

        logging.info(
            f"{n_rows} rows processed, {n_fragments_matched} fragments matched, "
            f"{len(matched_inchikeys)} unique inchikeys covered so far"
        )

    if not wrote_header:
        # No matches at all -- still write an empty file with headers so
        # downstream code has a well-formed (if empty) file to load.
        pd.DataFrame(columns=columns).to_csv(out_path, sep="\t", index=False)

    logging.info(
        "done: "
        f"{n_rows} rows processed, {n_skipped_malformed} skipped (malformed), "
        f"{n_fragments_seen} product fragments parsed, "
        f"{n_fragments_matched} fragments matched to target inchikeys, "
        f"{len(matched_inchikeys)} unique inchikeys covered, "
        f"{len(matched_reaction_ids)} unique reaction_ids in output"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reaction-source",
        required=True,
        help="Path to a reaction-SMILES CSV (PatentNumber,Year,reactions columns, "
        "reactants>agents>products SMILES). Currently only this USPTO_FULL-style "
        "format is supported.",
    )
    parser.add_argument(
        "--source-name",
        required=True,
        help="Value for the 'source' column, e.g. USPTO_FULL",
    )
    parser.add_argument(
        "--labels-file",
        required=True,
        help="labels.tsv to join against (must have an 'inchikey' column)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for reaction_metadata.tsv",
    )
    parser.add_argument("--chunksize", type=int, default=50_000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    labels = pd.read_csv(args.labels_file, sep="\t", dtype=str)
    target_inchikeys = set(labels["inchikey"].dropna()) - {""}
    logging.info(f"Loaded {len(target_inchikeys)} target inchikeys from {args.labels_file}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    build_from_uspto(
        reaction_source=args.reaction_source,
        source_name=args.source_name,
        target_inchikeys=target_inchikeys,
        out_path=out_path,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
