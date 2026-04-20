import os
from functools import partial
import random
from typing import Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

from tqdm import tqdm
from pandas import DataFrame
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.metrics import auc
from rdkit.ML.Cluster import Butina
from multiprocessing import Pool, cpu_count
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import pdist, squareform
from trade.constants import SOURCE_SET_PATH, TARGET_SET_PATH
from scripts.constants import EXTEND_FEATURES, SHAP_FEATURES

_shared_group_1 = None
_shared_group_2 = None
_features = None
_shared_fingerprints = None


def _init_worker_phychem(data1, data2, features):
    global _shared_group_1, _shared_group_2, _features
    _shared_group_1 = data1.values
    _shared_group_2 = data2.values
    _features = features


def _init_worker_structure(fingerprints):
    global _shared_fingerprints
    _shared_fingerprints = fingerprints


def overlap_coefficient(group_1, group_2, grids=1000, lower_q=2, upper_q=98):
    if len(set(group_1)) < 5 and len(set(group_2)) < 5:
        return 1.0
    if np.std(group_1) == 0:
        group_1 = group_1 + np.random.normal(0, 1e-8, size=group_1.shape)
    if np.std(group_2) == 0:
        group_2 = group_2 + np.random.normal(0, 1e-8, size=group_2.shape)

    q1_low, q1_high = np.percentile(group_1, [lower_q, upper_q])
    q2_low, q2_high = np.percentile(group_2, [lower_q, upper_q])
    clip_low = min(q1_low, q2_low)
    clip_high = max(q1_high, q2_high)

    group_1 = np.clip(group_1, clip_low, clip_high)
    group_2 = np.clip(group_2, clip_low, clip_high)

    kde_1 = gaussian_kde(group_1)
    kde_2 = gaussian_kde(group_2)
    xs = np.linspace(clip_low, clip_high, grids)
    g1 = kde_1(xs)
    g2 = kde_2(xs)

    overlap = auc(xs, np.minimum(g1, g2))
    norm = min(auc(xs, g1), auc(xs, g2))
    return overlap / norm


def compute_distance(i):
    fp_i = _shared_fingerprints[i]
    sims = DataStructs.BulkTanimotoSimilarity(fp_i, _shared_fingerprints[:i])
    return [1 - sim for sim in sims]


def ClusterByFp(fingerprints, threshold: float = 0.2):
    n_fps = len(fingerprints)
    args = range(1, n_fps)
    results = []
    n_proc = int(os.environ.get("LSB_DJOB_NUMPROC", cpu_count()))
    func = partial(compute_distance)
    with Pool(processes=n_proc, initializer=_init_worker_structure, initargs=(fingerprints,)) as pool:
        for result in tqdm(pool.imap(func, args, chunksize=64), total=len(args), desc="Distance computation"):
            results.append(result)

    distance_matrix = []
    for row in results:
        distance_matrix.extend(row)
    del results
    clusters = Butina.ClusterData(data=distance_matrix, nPts=n_fps, distThresh=threshold, isDistData=True)
    return distance_matrix, clusters


def run_single_overlap(feature_group):
    global _shared_group_1, _shared_group_2, _features
    results = []
    for f in feature_group:
        idx = _features.index(f)
        x1 = _shared_group_1[:, idx]
        x2 = _shared_group_2[:, idx]
        ov = overlap_coefficient(x1, x2)
        results.append((f, ov))
    return results


def run_parallel_overlap(group_1, group_2, features, group_size=3):
    feature_groups = [features[i:i + group_size] for i in range(0, len(features), group_size)]
    n_proc = int(os.environ.get("LSB_DJOB_NUMPROC", cpu_count()))
    results = []

    with Pool(processes=n_proc, initializer=_init_worker_phychem, initargs=(group_1, group_2, features)) as pool:
        for group_result in tqdm(pool.imap(run_single_overlap, feature_groups),
                                 total=len(feature_groups), desc="Overlap computation"):
            results.extend(group_result)
    return pd.DataFrame(results, columns=["Feature", "Overlap"])


def tsne_by_categories(datasets):
    labels, all_data, sizes = [], [], []

    for name, dataset in datasets:
        labels.append(name)
        all_data.append(dataset)
        sizes.append(dataset.shape[0])
    results = {}
    for category, feature_names in tqdm(SHAP_FEATURES.items(), desc="t-sne processing", total=len(SHAP_FEATURES)):
        if category == 'NP Characters':
            continue

        combined_data = pd.concat(all_data, axis=0)
        classified_data = pd.DataFrame(combined_data)[feature_names].to_numpy()

        classified_data = StandardScaler().fit_transform(classified_data)

        tsne = TSNE(n_components=2, random_state=1, perplexity=2, n_iter=1000)
        embedded = tsne.fit_transform(classified_data)

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
        results[category] = pd.DataFrame(columns_dict)

    return results


def phychem_diversity_analysis(
        mol_descriptors: pd.DataFrame, n_representatives: int = 100
) -> tuple[DataFrame, DataFrame]:
    descriptors_scaled = StandardScaler().fit_transform(mol_descriptors)
    cos_dist = pdist(descriptors_scaled, metric='cosine')
    sim_matrix = 1 - squareform(cos_dist)
    sim_data = pd.DataFrame(sim_matrix, index=mol_descriptors.index, columns=mol_descriptors.index)

    if mol_descriptors.shape[0] > n_representatives:
        kmeans = KMeans(n_clusters=n_representatives, random_state=0, n_init=10)
        kmeans.fit(descriptors_scaled)
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, descriptors_scaled)
        representative_idx = sorted(closest)
        reps = mol_descriptors.iloc[representative_idx]

        reps_scaled = StandardScaler().fit_transform(reps)
        cos_dist_reps = pdist(reps_scaled, metric='cosine')
        sim_matrix_reps = 1 - squareform(cos_dist_reps)
        vague_sim_data = pd.DataFrame(sim_matrix_reps, index=reps.index, columns=reps.index)
    else:
        vague_sim_data = sim_data.copy()

    return sim_data, vague_sim_data


def structure_diversity_analysis(
        mol_list: pd.DataFrame, threshold: float = 0.6
) -> tuple[pd.DataFrame, pd.DataFrame]:
    Clustering_smiles = np.array(mol_list)
    Fingerprint = []
    valid_idx = []
    for i, smiles in enumerate(tqdm(Clustering_smiles, desc='Clustering')):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"[WARN] Cannot parse SMILES: {smiles}")
            continue
        fp = Chem.RDKFingerprint(mol, minPath=1, fpSize=1024)
        Fingerprint.append(fp)
        valid_idx.append(i)

    distance_matrix, clusters = ClusterByFp(Fingerprint, threshold=threshold)

    sim_matrix_full = 1 - squareform(distance_matrix)
    sim_data = pd.DataFrame(sim_matrix_full, index=valid_idx, columns=valid_idx)

    representative_idx = [valid_idx[random.choice(cluster)] for cluster in clusters]
    reps_fp = [Fingerprint[i] for i in representative_idx]

    n_reps = len(reps_fp)
    reps_dist = []
    for i in range(1, n_reps):
        sims = DataStructs.BulkTanimotoSimilarity(reps_fp[i], reps_fp[:i])
        reps_dist.extend([1 - sim for sim in sims])

    reps_sim_matrix = 1 - squareform(reps_dist)
    vague_sim_data = pd.DataFrame(reps_sim_matrix, index=representative_idx, columns=representative_idx)
    return sim_data, vague_sim_data


def structure_scaffold_analysis(
        mol_list: pd.DataFrame,
) -> pd.DataFrame:
    smiles_array = np.array(mol_list)
    scaffolds = []

    for i, smile in enumerate(tqdm(smiles_array, desc='Extracting scaffolds')):
        smi = smile if isinstance(smile, (list, np.ndarray)) else smile
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"[WARN] Cannot parse SMILES: {smi}")
            scaffolds.append(None)
            continue

        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is not None and scaffold.GetNumAtoms():
            scaffold_smi = Chem.MolToSmiles(scaffold)
        else:
            frags = Chem.GetMolFrags(mol, asMols=True)
            scaffold_smi = Chem.MolToSmiles(max(frags, key=lambda x: x.GetNumAtoms()))
        scaffolds.append(scaffold_smi)

    unique_scaffolds = pd.Series(scaffolds).unique()
    scaffold_fps = []
    for smi in tqdm(unique_scaffolds, desc='scaffold fps processing'):
        mol = Chem.MolFromSmiles(smi)
        fp = Chem.RDKFingerprint(mol, fpSize=1024)
        arr = np.zeros((fp.GetNumBits(),), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        scaffold_fps.append(arr)
    scaffold_fps = np.array(scaffold_fps)

    if len(scaffold_fps) > 30:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        tsne = TSNE(n_components=2, random_state=42, perplexity=20)

    xy = tsne.fit_transform(scaffold_fps)

    id_maps = []
    for unique_scaffold in unique_scaffolds:
        id_map = []
        for index, scaffold in enumerate(scaffolds):
            if unique_scaffold == scaffold:
                id_map.append(index)
        id_maps.append(id_map)

    group_similarity = []
    for maps in tqdm(id_maps, desc='mapping'):
        if len(maps) < 2:
            group_similarity.append(0)
        else:
            fps = []
            for index in maps:
                mol = Chem.MolFromSmiles(smiles_array[index])
                if mol:
                    fp = Chem.RDKFingerprint(mol, fpSize=1024)
                    fps.append(fp)
            similarity = []
            for i in range(len(fps)):
                similarity.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
            group_similarity.append(np.mean(similarity))

    map_length = [len(_map) for _map in id_maps]

    scaffold_info = pd.DataFrame({
        "Scaffold": unique_scaffolds,
        "tSNE1": xy[:, 0],
        "tSNE2": xy[:, 1],
        "group_similarity": group_similarity,
        "molecule_amount": map_length
    })

    return scaffold_info


def phychem_evaluation():
    source_set = pd.read_csv(SOURCE_SET_PATH)
    target_set = pd.read_csv(TARGET_SET_PATH)

    source_pos = source_set[source_set["Anti"] == 1][EXTEND_FEATURES]
    source_neg = source_set[source_set["Anti"] == 0][EXTEND_FEATURES]
    target_pos = target_set[target_set["Anti"] == 1][EXTEND_FEATURES]
    target_neg = target_set[target_set["Anti"] == 0][EXTEND_FEATURES]

    # Necessary evaluation
    pos = pd.concat([source_pos, target_pos], axis=0)
    neg = pd.concat([source_neg, target_neg], axis=0)
    pos_neg_overlap = run_parallel_overlap(pos, neg, EXTEND_FEATURES, group_size=3)

    # Rational evaluation
    source_all = pd.concat([source_pos, source_neg], axis=0)
    target_all = pd.concat([target_pos, target_neg], axis=0)
    source_target_overlap = run_parallel_overlap(source_all, target_all, EXTEND_FEATURES, group_size=3)

    # Categories evaluation
    data_list = [("sour_pos", source_pos), ("sour_neg", source_neg), ("tar_pos", target_pos), ("tar_neg", target_neg)]
    tsne_by_descriptors = tsne_by_categories(data_list)

    total_data = pd.concat([source_pos, source_neg, target_pos, target_neg], axis=0)
    phychem_diversity_analysis(total_data)
    sim_data, vague_sim_data = phychem_diversity_analysis(total_data)

    return pos_neg_overlap, source_target_overlap, tsne_by_descriptors, sim_data, vague_sim_data


def structure_evaluation(

):
    source_set = pd.read_csv(SOURCE_SET_PATH)
    target_set = pd.read_csv(TARGET_SET_PATH)

    source_smiles = source_set['smiles']
    target_smiles = target_set['smiles']
    total_data = pd.concat([source_smiles, target_smiles], axis=0)

    # diversity of total set evaluation.
    sim_data, vague_sim_data = structure_diversity_analysis(total_data)

    # scaffold analysis of total set.
    scaffold_info = structure_scaffold_analysis(total_data)

    # scaffold analysis of antibiotic set.
    source_data = pd.concat([source_smiles], axis=0)
    source_scaffold_info = structure_scaffold_analysis(source_data)

    # scaffold analysis of antibiotic set.
    target_data = pd.concat([target_smiles], axis=0)
    target_scaffold_info = structure_scaffold_analysis(target_data)

    return sim_data, vague_sim_data, scaffold_info, source_scaffold_info, target_scaffold_info


def set_evaluation(

):
    # # phychem_evaluation on training set.
    # pos_neg_overlap, source_target_overlap, tsne_by_descriptors, sim_data, vague_sim_data = phychem_evaluation()
    # merged_overlap = pd.concat([pos_neg_overlap.set_index("Feature")["Overlap"],
    #                             source_target_overlap.set_index("Feature")["Overlap"]],
    #                            axis=1, keys=["pos_neg", "source_target"]).loc[EXTEND_FEATURES]
    # merged_overlap.to_csv("phychem_overlap.csv")
    # for category, data in tsne_by_descriptors.items():
    #     data.to_csv(f"tsne_by_{category}.csv", index=False)
    # sim_data.to_csv("phychem_diversity.csv", index=False)
    # vague_sim_data.to_csv("phychem_diversity_vague.csv", index=False)

    # structure_evaluation on training set.
    sim_data, vague_sim_data, scaffold_info, source_scaffold_info, target_scaffold_info = structure_evaluation()
    sim_data.to_csv("structure_diversity.csv", index=False)
    vague_sim_data.to_csv("structure_diversity_vague.csv", index=False)
    scaffold_info.to_csv("structure_scaffold.csv", index=False)
    source_scaffold_info.to_csv("structure_scaffold_source.csv", index=False)
    target_scaffold_info.to_csv("structure_scaffold_target.csv", index=False)


if __name__ == '__main__':
    set_evaluation()
