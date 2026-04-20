import csv
import shutil
import tempfile
import os
from multiprocessing import Pool, cpu_count, get_context, TimeoutError as MP_TimeoutError
from pathlib import Path
from time import time

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdRGroupDecomposition
from rdkit.Chem.rdRGroupDecomposition import RGroupDecompositionParameters, RGroupCoreAlignment
from tqdm import tqdm
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
import gc

from scripts.constants import ANTIBIOTIC_PATH, DEFAULT_COLUMNS, ANTIBIOTIC_CORE, LABELED_FILE, UNLABELED_FILE

RDLogger.DisableLog('rdApp.*')

params = RGroupDecompositionParameters()
params.onlyMatchAtRGroups = False
params.allowNonTerminalRGroups = True
params.doTautomers = True
params.removeHydrogensPostMatch = True
params.alignment = RGroupCoreAlignment.MCS

width = shutil.get_terminal_size().columns
_shared_fingerprints = None


def _init_worker(fingerprints):
    global _shared_fingerprints
    _shared_fingerprints = fingerprints


def run_mcs(molecules):
    from rdkit.Chem import rdFMCS
    result = rdFMCS.FindMCS(molecules, timeout=120)
    return result.smartsString if not result.canceled else None


def ClusterByFp(fingerprints, threshold: float = 0.2):
    n_fps = len(fingerprints)
    args = range(1, n_fps)
    results = []
    n_proc = int(os.environ.get("LSB_DJOB_NUMPROC", cpu_count()))
    with Pool(processes=n_proc, initializer=_init_worker, initargs=(fingerprints,)) as pool:
        for result in tqdm(pool.imap(compute_distance, args, chunksize=64), total=len(args), desc="Distance computation"):
            results.append(result)

    distance_matrix = []
    for row in results:
        distance_matrix.extend(row)
    del results
    gc.collect()
    print(f"[INFO] Finished Distance Matrix Computation")
    clusters = Butina.ClusterData(data=distance_matrix, nPts=n_fps, distThresh=threshold, isDistData=True)
    return clusters


def compute_distance(i):
    fp_i = _shared_fingerprints[i]
    sims = DataStructs.BulkTanimotoSimilarity(fp_i, _shared_fingerprints[:i])
    return [1 - sim for sim in sims]


def prune_and_save(args):
    core, molecule_list = args
    smiles = Chem.MolToSmiles(core)
    temp_data = molecule_pruning(molecule_list=molecule_list, core_list=[smiles],
                                 category_id='Negative Samples', label_id='Unlabeled', loose_match=False)
    temp_path = Path(tempfile.gettempdir()) / f"temp_{hash(smiles)}.csv"
    temp_data.to_csv(temp_path, index=False)
    return temp_path


def calc_fingerprint(
        smiles: str
) -> Chem.DataStructs.ExplicitBitVect | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.RDKFingerprint(mol, minPath=1, fpSize=4096)


def wildcard_atom_replacement(mol: Chem.Mol, atom_index: int) -> Chem.Mol:
    mol = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(atom_index)
            atom.SetAtomMapNum(0)

    for atom in mol.GetAtoms():
        atom.SetIsAromatic(False)
    for bond in mol.GetBonds():
        bond.SetIsAromatic(False)

    mol = Chem.RemoveHs(mol)
    return mol


def molecule_pruning(
        molecule_list: pd.DataFrame = None,
        core_list: list = None,
        category_id: str = None,
        label_id: str = None,
        loose_match: bool = False

) -> pd.DataFrame:
    cid, category, cores, r_groups, smiles, labels, InChIKeys, activity = [], [], [], [], [], [], [], []
    core_mol = [mol for smiles in core_list if (mol := Chem.MolFromSmiles(smiles)) is not None]

    molecule_list_mol = []
    mol_index_map = []
    for idx, smile in enumerate(molecule_list['SMILES']):
        try:
            mol = Chem.MolFromSmiles(smile)
            try:
                Chem.RemoveStereochemistry(mol)
            except Exception as e:
                print(f"[Warning] RemoveStereochemistry failed for SMILES {smile}: {e}")
                continue
            if mol is not None:
                molecule_list_mol.append(mol)
                mol_index_map.append(idx)
            else:
                print(f"[Warning] Invalid SMILES skipped: {smile}")
        except Exception as e:
            print(f"[Warning] Failed to process SMILES {smile}: {e}")

    R_groups, fails = rdRGroupDecomposition.RGroupDecompose(cores=core_mol, mols=molecule_list_mol, options=params)
    matched_indices = list(set(range(len(molecule_list_mol))) - set(fails))
    original_indices = [mol_index_map[i] for i in matched_indices]

    for i, R_group in enumerate(R_groups):
        if not loose_match:
            if not Chem.MolFromSmiles(molecule_list['SMILES'][original_indices[i]]).HasSubstructMatch(
                    wildcard_atom_replacement(R_group['Core'], atom_index=1)):
                continue
        for label, mol in R_group.items():
            if label.startswith("R") and mol is not None:
                mol = wildcard_atom_replacement(mol=mol, atom_index=1)
                if not Chem.MolFromSmiles(molecule_list['SMILES'][original_indices[i]]).HasSubstructMatch(mol):
                    continue
                try:
                    frags = Chem.GetMolFrags(mol, asMols=True)
                except Chem.rdchem.KekulizeException as e:
                    print(f"[Warning] Molecule {molecule_list['Compound_CID'][original_indices[i]]} Failed")
                    print(f"[Warning] Fragment kekulization failed:{e}")
                    del mol
                    gc.collect()
                    continue
                for frag in frags:
                    frag = Chem.RemoveHs(frag)
                    try:
                        InChI = Chem.MolToInchi(frag)
                        InChIKey = Chem.InchiToInchiKey(InChI)
                    except Exception as e:
                        print(f"[Warning] InChI generation failed for fragment: {e}")
                        continue
                    if InChIKey not in InChIKeys:
                        r_group = Chem.MolToSmiles(frag)

                        cid.append(molecule_list['Compound_CID'][original_indices[i]])
                        category.append(category_id)
                        cores.append(Chem.MolToSmiles(wildcard_atom_replacement(R_group['Core'], atom_index=1)))
                        r_groups.append(r_group)
                        smiles.append(molecule_list['SMILES'][original_indices[i]])
                        labels.append(label_id)
                        InChIKeys.append(InChIKey)
                        activity.append(molecule_list['Activity'][original_indices[i]])

    ligands = pd.DataFrame({'Compound_CID': cid, 'Category': category, 'Core': cores, 'SMILES': r_groups,
                            'Source': smiles, 'Label': labels, 'InChIKeys': InChIKeys, 'Activity': activity})

    return ligands


def direct_pruning(
        root_path: Path = Path(ANTIBIOTIC_PATH / 'Unprocessed'),
        save_path: Path = Path(ANTIBIOTIC_PATH / 'Processed'),
) -> None:

    with open(save_path, mode='w', newline='', encoding='utf8') as results:
        writer = csv.writer(results)
        writer.writerow(['Compound_CID', 'Category', 'Core', 'SMILES', 'Source', 'Label', 'InChIKeys', 'Activity'])

    print(f'Starting processing labeled data...\n')
    for core_name, core_list in ANTIBIOTIC_CORE.items():
        ligands_amount = 0
        print(f"Core name: {core_name}, label: {LABELED_FILE.get(core_name)}")
        # Iterate over all CSV files in the specified directory.
        for label in LABELED_FILE.get(core_name) + UNLABELED_FILE:
            if label in LABELED_FILE.get(core_name):
                sub_folder = 'PositiveLabeled'
            else:
                sub_folder = 'PositiveUnlabeled'
            data_file = root_path / sub_folder / f'{label}.csv'

            if data_file.exists():
                molecule_list = pd.read_csv(data_file, usecols=DEFAULT_COLUMNS)
                uni_molecule = []
                for _, row in molecule_list.iterrows():
                    smiles = row['SMILES']
                    if '.' in smiles:
                        sub_molecules = smiles.split('.')
                        for sub_molecule in sub_molecules:
                            new_row = row.copy()
                            new_row['SMILES'] = sub_molecule
                            uni_molecule.append(new_row)
                    else:
                        uni_molecule.append(row)
                molecule_list = pd.DataFrame(uni_molecule).reset_index(drop=True)
                temp_data = molecule_pruning(molecule_list=molecule_list, core_list=core_list, category_id=core_name,
                                             label_id=label, loose_match=True)
                temp_data.to_csv(save_path, index=False, mode='a', header=False)
                print(f"\t({sub_folder} data)->{label}-labeled antibiotics: {len(temp_data)}")
                ligands_amount += len(temp_data)
            else:
                raise FileNotFoundError(f"Missing file: {data_file}")
        print(f"Total antibiotics ligands: {ligands_amount}")
        print("─" * width)


def indirect_pruning(
        data_file: Path = Path(ANTIBIOTIC_PATH / 'Unprocessed'),
        save_path: Path = Path(ANTIBIOTIC_PATH / 'Processed'),
) -> None:

    if data_file.exists():
        if data_file.is_dir():
            csv_files = list(data_file.glob('*.csv'))
            molecule_list = pd.concat([pd.read_csv(file, usecols=DEFAULT_COLUMNS) for file in csv_files], ignore_index=True)
        else:
            molecule_list = pd.read_csv(data_file, usecols=DEFAULT_COLUMNS)

        n_proc = int(os.environ.get("LSB_DJOB_NUMPROC", cpu_count()))

        with Pool(processes=n_proc) as pool:
            print(f"[INFO] Using {n_proc} cores for parallel fingerprint computation.")
            fingerprints = list(
                tqdm(pool.imap(calc_fingerprint, molecule_list['SMILES'], chunksize=128), total=len(molecule_list),
                     desc='Fingerprint Collecting'))
        Fingerprint = []
        valid_indices = []
        for idx, (smile, fp) in enumerate(zip(molecule_list['SMILES'], fingerprints)):
            if fp is not None:
                Fingerprint.append(fp)
                valid_indices.append(idx)

        del fingerprints
        gc.collect()
        molecule_list = molecule_list.iloc[valid_indices].reset_index(drop=True)

        print(f"[INFO] Using {n_proc} cores for parallel similarity matrix computation.")
        clusters = ClusterByFp(Fingerprint, threshold=0.2)
        del Fingerprint
        gc.collect()

        filtered_clusters = [tup for tup in clusters if len(tup) != 1]
        filtered_clusters = [tup for tup in filtered_clusters if len(tup) != 2]
        unique_group = []
        valid_clusters = []
        keys = set()
        start = time()

        print(f"[INFO] Starting MCS computation on {len(filtered_clusters)} clusters.")
        ctx = get_context("fork")
        with ctx.Pool(processes=n_proc) as pool:
            futures = []
            cluster_records = []
            for filtered_cluster in tqdm(filtered_clusters, desc="Preparing MCS Step"):
                mols = [Chem.MolFromSmiles(molecule_list['SMILES'].iloc[i]) for i in filtered_cluster]
                mols = [mol for mol in mols if mol is not None]
                if len(mols) >= 2:
                    futures.append(pool.apply_async(run_mcs, args=(mols,)))
                    cluster_records.append(filtered_cluster)

            for i, future in enumerate(tqdm(futures, desc="Finding MCS")):
                try:
                    smarts_string = future.get(timeout=120)
                except MP_TimeoutError:
                    print(f"[Timeout] MCS #{i} exceeded 2 minutes.")
                    continue
                except Exception as e:
                    print(f"[Error] MCS #{i} failed: {e}")
                    continue

                if smarts_string is None:
                    continue

                try:
                    mcs_mol = Chem.MolFromSmarts(smarts_string)
                    key = Chem.inchi.MolToInchiKey(mcs_mol)
                except Exception as e:
                    print(f"[InChIKey generation failed]: {e}")
                    continue
                if key not in keys:
                    unique_group.append(mcs_mol)
                    valid_clusters.append(cluster_records[i])
                    keys.add(key)
                del mcs_mol
                gc.collect()

        print(f"Time taken: {time() - start:.2f}s")
        del keys
        gc.collect()
        # Save unique_group to CSV with corresponding cluster members
        mcs_output_path = ANTIBIOTIC_PATH / 'Processed' / "unique_mcs_cores.csv"
        with open(mcs_output_path, mode='w', newline='', encoding='utf8') as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES", "InChIKeys", "ClusterMembers"])
            for mol, cluster in zip(unique_group, valid_clusters):
                smiles = Chem.MolToSmiles(mol)
                member_smiles = [molecule_list['SMILES'].iloc[i] for i in cluster]
                try:
                    inchi_key = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))
                except Exception as e:
                    inchi_key = f"Error: {e}"
                writer.writerow([smiles, inchi_key] + member_smiles)
        print(f"[INFO] Saved {len(unique_group)} unique MCS cores to {mcs_output_path}")
        del filtered_clusters, valid_clusters
        gc.collect()

        print(f"[INFO] Starting pruning on {len(unique_group)} Molecule cores. ")
        with Pool(n_proc) as pool:
            temp_paths = list(tqdm(pool.imap(prune_and_save, [(core, molecule_list) for core in unique_group], chunksize=8),
                                   total=len(unique_group), desc="Pruning ligands"))

        with open(save_path, mode='w', newline='', encoding='utf8') as results:
            writer = csv.writer(results)
            writer.writerow(['Compound_CID', 'Category', 'Core', 'SMILES', 'Source', 'Label', 'InChIKeys', 'Activity'])
        with open(save_path, mode='a', newline='', encoding='utf8') as out_file:
            for temp_path in temp_paths:
                df = pd.read_csv(temp_path)
                df.to_csv(out_file, index=False, header=False)
                temp_path.unlink()
    else:
        raise FileNotFoundError(f"Missing file: {data_file}")


def data_processing(
        root_path: Path = Path(ANTIBIOTIC_PATH / 'Unprocessed'),
        save_path: Path = Path(ANTIBIOTIC_PATH / 'Processed'),
) -> None:

    # Processing the positive sample data.
    file_name = save_path / 'Positive_Ligands.csv'
    direct_pruning(root_path=root_path, save_path=file_name)
    # file_name = save_path / 'Positive_Ligands_2.csv'
    # data_file = root_path / 'PositiveUnlabeled'
    # indirect_pruning(data_file=data_file, save_path=file_name)
    #
    # # Processing the negative sample data.
    # file_name = save_path / 'Negative_Ligands.csv'
    # data_file = root_path / 'NegativeUnlabeled' / f'NegativeSample.csv'
    # indirect_pruning(data_file=data_file, save_path=file_name)


if __name__ == '__main__':

    from tap import tapify
    tapify(data_processing)
