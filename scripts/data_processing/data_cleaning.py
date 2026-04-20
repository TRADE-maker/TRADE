from pathlib import Path

import pandas as pd
from scripts.constants import ANTIBIOTIC_PATH


def self_cleaning(
        file: pd.DataFrame, label: str
) -> pd.DataFrame:
    before = len(file)
    file = file.drop_duplicates(subset="InChIKeys", keep="first")
    after = len(file)
    print(f"[Self-cleaning] Removed {before - after} duplicate rows from {label} based on InChIKeys.")
    return file


def comparison_cleaning(
        pos_file: pd.DataFrame, neg_file: pd.DataFrame
) -> [pd.DataFrame, pd.DataFrame]:
    common_keys = set(pos_file["InChIKeys"]).intersection(set(neg_file["InChIKeys"]))
    positive_ligands = pos_file[~pos_file["InChIKeys"].isin(common_keys)]
    negative_ligands = neg_file[~neg_file["InChIKeys"].isin(common_keys)]
    print(f"[Comparison-cleaning] Total common InChIKeys removed: {len(common_keys)}")

    return [positive_ligands, negative_ligands]


def data_cleaning(
        root_path: Path = Path(ANTIBIOTIC_PATH / 'Processed')
) -> None:
    pos_file = root_path / f'Positive_Ligands.csv'
    neg_file = root_path / f'Negative_Ligands.csv'
    positive_ligands = pd.read_csv(pos_file)
    negative_ligands = pd.read_csv(neg_file)

    if "InChIKeys" not in positive_ligands.columns or "InChIKeys" not in negative_ligands.columns:
        raise ValueError("Both files must contain an 'InChIKeys' column.")
    positive_ligands = self_cleaning(file=positive_ligands, label='positive_ligands')
    negative_ligands = self_cleaning(file=negative_ligands, label='negative_ligands')
    positive_ligands, negative_ligands = comparison_cleaning(pos_file=positive_ligands, neg_file=negative_ligands)

    positive_ligands.to_csv(pos_file, index=False)
    negative_ligands.to_csv(neg_file, index=False)


if __name__ == '__main__':
    from tap import tapify
    tapify(data_cleaning)
