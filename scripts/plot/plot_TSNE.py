import random

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from rdkit import Chem
from tqdm import tqdm

from scripts.constants import EXTEND_FEATURES, DATA_DIR

from pathlib import Path

ORIGINAL_FILE = Path(DATA_DIR / 'Original_space')
EMP_FILE = Path(DATA_DIR / 'After_Emp')
PHYCHEM_FILE = Path(DATA_DIR / 'After_phychem')
STRUCTURE_FILE = Path(DATA_DIR / 'After_structure')
RANKING_FILE = Path(DATA_DIR / 'After_ranking')
SOURCE_SET_PATH = Path(DATA_DIR / 'TrainingSet' / 'SourceDataset.csv')
TARGET_SET_PATH = Path(DATA_DIR / 'TrainingSet' / 'TargetDataset.csv')
HIT_PATH = Path(DATA_DIR / 'HIT')


def sample_stream_des(folder_path: Path, frac: float):
    sampled_rows = []
    if folder_path.name == 'SourceDataset.csv':
        iterator = [SOURCE_SET_PATH]
    elif folder_path.name == 'TargetDataset.csv':
        iterator = [TARGET_SET_PATH]
    else:
        iterator = list(folder_path.glob("*.csv"))

    for file in tqdm(iterator, desc=folder_path.name):
        try:
            data = pd.read_csv(file, usecols=EXTEND_FEATURES)
            mask = [random.random() < frac for _ in range(len(data))]
            sampled = data[mask]
            sampled = sampled.replace([np.inf, -np.inf], np.nan).dropna()
            if not sampled.empty:
                sampled_rows.append(sampled)
        except Exception as e:
            print(f"{file}: {e}")
    data = pd.concat(sampled_rows, ignore_index=True)

    if sampled_rows:
        return data
    else:
        return pd.DataFrame()


def sample_stream_str(folder_path: Path, frac: float):
    sampled_rows = []

    if folder_path.name == 'SourceDataset.csv':
        iterator = [SOURCE_SET_PATH]
    elif folder_path.name == 'TargetDataset.csv':
        iterator = [TARGET_SET_PATH]
    else:
        iterator = list(folder_path.glob("*.csv"))

    for file in tqdm(iterator, desc=folder_path.name):
        try:
            sampled_fps = []
            data = pd.read_csv(file, usecols=['smiles'])
            mask = [random.random() < frac for _ in range(len(data))]
            sampled = data[mask]
            for smi in sampled['smiles']:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    fp = Chem.RDKFingerprint(mol, minPath=2, fpSize=2048)
                    arr = np.array(list(fp), dtype=float)
                    sampled_fps.append(arr)

            if sampled_fps:
                sampled_fps = pd.DataFrame(sampled_fps)
                sampled_rows.append(sampled_fps)
        except Exception as e:
            print(f"{file}: {e}")
    data = pd.concat(sampled_rows, ignore_index=True)
    length = len(data)

    if sampled_rows:
        return data, length
    else:
        return pd.DataFrame()


def plot_by_descriptors():
    datasets = [
        ("ORI", ORIGINAL_FILE, 1500 / 9266896),
        ("EMP", EMP_FILE, 400 / 2044218),
        ("PHYC", PHYCHEM_FILE, 200 / 59679),
        ("STRU", STRUCTURE_FILE, 100 / 2042),
        ("RANK", RANKING_FILE, 50 / 1000),

        ("SOUR", SOURCE_SET_PATH, 800 / 2514),
        ("TAR", TARGET_SET_PATH, 1.0),
        ("HIT", HIT_PATH, 1.0)
    ]

    labels, all_data, sizes = [], [], []

    for name, folder, frac in datasets:
        labels.append(name)
        data = sample_stream_des(folder_path=folder, frac=frac)
        all_data.append(data)
        sizes.append(data.shape[0])
        print(f"{name}: {len(data)}")

    combined_data = np.concatenate(all_data, axis=0)
    print(f"Finished combined_data, total rows: {combined_data.shape[0]}")
    from sklearn.preprocessing import StandardScaler
    combined_data = StandardScaler().fit_transform(combined_data)

    tsne = TSNE(n_components=2, random_state=1, perplexity=2, n_iter=10000)
    embedded = tsne.fit_transform(combined_data)
    print("Finished t-SNE")

    indices = [0] + np.cumsum(sizes).tolist()
    columns_dict = {}
    for i, label in enumerate(labels):
        seg = embedded[indices[i]:indices[i + 1], :]
        columns_dict[f"{label}_Dim1"] = seg[:, 0]
        columns_dict[f"{label}_Dim2"] = seg[:, 1]

    max_len = max(len(col) for col in columns_dict.values())
    for k in columns_dict:
        if len(columns_dict[k]) < max_len:
            columns_dict[k] = np.pad(columns_dict[k], (0, max_len - len(columns_dict[k])), constant_values=np.nan)

    df = pd.DataFrame(columns_dict)
    df.to_csv("tsne_by_descriptors.csv", index=False)

    plt.figure(figsize=(10, 8))
    indices = [0] + np.cumsum(sizes).tolist()
    colors = plt.colormaps["tab10"].resampled(len(labels))
    for i, label in enumerate(labels):
        seg = embedded[indices[i]:indices[i + 1], :]
        plt.scatter(
            seg[:, 0], seg[:, 1],
            label=f"{label} ({sizes[i]})",
            alpha=0.6,
            s=10,
            color=colors(i)
        )
    plt.title("t-SNE visualization by dataset", fontsize=16)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.show()


def plot_by_structures():
    datasets = [
        ("ORI", ORIGINAL_FILE, 1500 / 9266896),
        ("EMP", EMP_FILE, 400 / 2044218),
        ("PHYC", PHYCHEM_FILE, 200 / 59679),
        ("STRU", STRUCTURE_FILE, 100 / 2042),
        ("RANK", RANKING_FILE, 50 / 1000),

        ("SOUR", SOURCE_SET_PATH, 800 / 2514),
        ("TAR", TARGET_SET_PATH, 1.0),
        ("HIT", HIT_PATH, 1.0)
    ]

    labels, all_data, sizes = [], [], []

    for name, folder, frac in datasets:
        labels.append(name)
        data, n_rows = sample_stream_str(folder_path=folder, frac=frac)
        all_data.append(data)
        sizes.append(data.shape[0])
        print(f"{name}: {len(data)}")

    combined_data = np.concatenate(all_data, axis=0)
    print(f"Finished combined_data, total rows: {combined_data.shape[0]}")
    from sklearn.preprocessing import StandardScaler
    combined_data = StandardScaler().fit_transform(combined_data)

    tsne = TSNE(n_components=2, random_state=1, perplexity=3, n_iter=1000)
    embedded = tsne.fit_transform(combined_data)
    print("Finished t-SNE")

    indices = [0] + np.cumsum(sizes).tolist()
    columns_dict = {}
    for i, label in enumerate(labels):
        seg = embedded[indices[i]:indices[i + 1], :]
        columns_dict[f"{label}_Dim1"] = seg[:, 0]
        columns_dict[f"{label}_Dim2"] = seg[:, 1]

    max_len = max(len(col) for col in columns_dict.values())
    for k in columns_dict:
        if len(columns_dict[k]) < max_len:
            columns_dict[k] = np.pad(columns_dict[k], (0, max_len - len(columns_dict[k])), constant_values=np.nan)

    df = pd.DataFrame(columns_dict)
    df.to_csv("tsne_by_structures.csv", index=False)

    plt.figure(figsize=(10, 8))
    indices = [0] + np.cumsum(sizes).tolist()
    colors = plt.colormaps["tab10"].resampled(len(labels))
    for i, label in enumerate(labels):
        seg = embedded[indices[i]:indices[i + 1], :]
        plt.scatter(
            seg[:, 0], seg[:, 1],
            label=f"{label} ({sizes[i]})",
            alpha=0.6,
            s=10,
            color=colors(i)
        )
    plt.title("t-SNE visualization by dataset", fontsize=16)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.show()


def plot_distribution():
    plot_by_descriptors()
    plot_by_structures()


if __name__ == '__main__':
    from tap import tapify

    tapify(plot_distribution)
