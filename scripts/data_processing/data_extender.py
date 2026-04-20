import csv
import warnings
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

from scripts.constants import DOWNLOAD_PATH, EXTEND_FEATURES

warnings.simplefilter("ignore", category=DeprecationWarning)


def extend_from_file(
        group_data, save_path: Path, column_name: str, label: str
):
    """Process a subset of the dataset (grouped by a specific column), compute molecular descriptors, and append the
    results to a CSV file.

    :param group_data: The dataset with the basic data that need to be extended.
    :param save_path: The path where the processed data should be saved.
    :param column_name: The name of the column that contains SMILES strings.
    :param label: The name of the calculation objects.
    """
    # Extract SMILES column as a NumPy array
    group_smiles = np.array(group_data[[column_name]])
    # Collect descriptor results in a list for efficiency
    results = []
    for smiles in tqdm(group_smiles, desc=label):
        mol = Chem.MolFromSmiles(smiles[0])
        vals = Descriptors.CalcMolDescriptors(mol)
        results.append(vals)
    expanding_data = pd.DataFrame(results, columns=EXTEND_FEATURES)
    # Reset index to ensure proper merging
    group_data.reset_index(drop=True, inplace=True)
    expanding_data.reset_index(drop=True, inplace=True)
    df = pd.concat([group_data, expanding_data], axis=1)
    # Append results to the specified CSV file
    df.to_csv(save_path, index=False, mode='a', header=False)


def process_csv_file(csv_file, save_path, groupby_name, column_name):
    # Read the header (column names) from the CSV file.
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        basic_features = next(reader)

    FEATURES = basic_features + EXTEND_FEATURES
    file_name = save_path / f'{csv_file.name}'

    with open(file_name, mode='w', newline='', encoding='utf8') as cfa:
        writer = csv.writer(cfa)
        writer.writerow(FEATURES)
    # Load the entire CSV file as a Pandas DataFrame
    data = pd.read_csv(csv_file)
    if groupby_name is not None:
        grouped = data.groupby(groupby_name)
        for group_value, group_data in grouped:
            extend_from_file(group_data=group_data, save_path=file_name, column_name=column_name, label=str(group_value))
    else:
        extend_from_file(group_data=data, save_path=file_name, column_name=column_name, label=csv_file.stem)


def extend(
        save_path: Path, groupby_name: str = None, data_file: Path = Path(DOWNLOAD_PATH), column_name: str = 'SMILES'
):
    """Read multiple CSV files, group data by a specified column, compute molecular descriptors, and save the results
    to new CSV files.

    :param save_path: Directory where processed CSV files will be stored.
    :param data_file: Directory containing input CSV files.
    :param groupby_name: Column name used to group data before processing.
    :param column_name: Column name that contains SMILES strings.
    """
    csv_files = list(data_file.glob('*.csv'))
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as t:
        futures = [t.submit(process_csv_file, csv_file, save_path, groupby_name, column_name) for csv_file in csv_files]
        for future in futures:
            future.result()


if __name__ == '__main__':

    from tap import tapify
    tapify(extend)
