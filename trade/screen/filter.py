import csv
import gc
import os
import uuid
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.Draw import MolToImage
from rdkit.ML.Cluster import Butina
from tqdm import tqdm

from trade.constants import EMP_FILE, CLU_FILE, ORIGINAL_FILE, PHYCHEM_CLA_EMBEDDING_PATH, PHYCHEM_LAYER_PATH, \
    PHYCHEM_FILE, STRUCTURE_LAYER_PATH, STRUCTURE_FILE, \
    STRUCTURE_CLA_EMBEDDING_PATH, RANKING_LAYER_PATH, RANKING_FILE, STRUCTURE_REG_EMBEDDING_PATH
from trade.model_selector.featurizer import PhyChemEmbeddingModel, GraphEmbeddingModel


class Filter:
    """
    A class to apply multiple filtering layers on molecules based on various constraints.

    This class allows users to define and apply a series of filtering steps (such as empirical rules,
    graph convolutional networks, random forests, and clustering models) to process molecular datasets.
    """

    def __init__(
            self,
            set_path: Path,
            verbose: bool,
            replicate: bool
    ):
        """Creates the Generator.

        :param set_path: The path to the training set.
        :param verbose: Whether to print out additional statements during generation.
        :param replicate: Whether to apply replication strategies.
        """
        self.path = set_path
        self.verbose = verbose
        self.replicate = replicate
        self.chemical_constraints = []
        self.execution_orders = []

    @staticmethod
    def _init_worker(fingerprints):
        global _shared_fingerprints
        _shared_fingerprints = fingerprints

    @staticmethod
    def compute_distance(i):
        fp_i = _shared_fingerprints[i]
        sims = DataStructs.BulkTanimotoSimilarity(fp_i, _shared_fingerprints[:i])
        return [1 - sim for sim in sims]

    def ClusterByFp(self, fingerprints, threshold: float = 0.2):
        n_fps = len(fingerprints)
        args = range(1, n_fps)
        results = []
        n_proc = int(os.environ.get("LSB_DJOB_NUMPROC", cpu_count()))
        func = partial(self.compute_distance)
        with Pool(processes=n_proc, initializer=self._init_worker, initargs=(fingerprints,)) as pool:
            for result in tqdm(pool.imap(func, args, chunksize=64), total=len(args), desc="Distance computation"):
                results.append(result)

        distance_matrix = []
        for row in results:
            distance_matrix.extend(row)
        del results
        gc.collect()
        print(f"[INFO] Finished Distance Matrix Computation")
        clusters = Butina.ClusterData(data=distance_matrix, nPts=n_fps, distThresh=threshold, isDistData=True)
        return clusters

    @staticmethod
    def is_same_mcs(molecule1, molecule2) -> bool:
        RDKFingerprint1 = Chem.RDKFingerprint(molecule1, minPath=1, fpSize=8192)
        RDKFingerprint2 = Chem.RDKFingerprint(molecule2, minPath=1, fpSize=8192)
        if RDKFingerprint1 == RDKFingerprint2:
            return True
        else:
            return False

    @staticmethod
    def pool_run_emp(file_batch: list[Path], chemical_constraints: list):
        total_input, total_output = 0, 0
        for file in file_batch:
            chunk_size = 1000
            chunks = pd.read_csv(file, chunksize=chunk_size)
            small_dataframes = [chunk.reset_index(drop=True) for chunk in chunks]

            out_path = EMP_FILE / f'{file.name}'
            all_results = []

            for data in small_dataframes:
                total_input += len(data)
                for index in range(len(data)):
                    smiles = data.loc[index, 'smiles']
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None and any(mol.HasSubstructMatch(c) for c in chemical_constraints):
                        total_output += 1
                        all_results.append(data.loc[index].to_dict())

            if all_results:
                filtered_df = pd.DataFrame(all_results, columns=small_dataframes[0].columns)
                filtered_df.to_csv(out_path, index=False)
            del small_dataframes
            del all_results
            gc.collect()
        return total_input, total_output

    @staticmethod
    def pool_run_phychem(file_batch: list[Path], model):
        data_embedding = PhyChemEmbeddingModel(mode='classification')
        data_embedding.reload(path=PHYCHEM_CLA_EMBEDDING_PATH)
        data_embedding.load_data(paths=file_batch)
        data_embedding.embedding()
        X = data_embedding.processed_data
        total_input, total_output = 0, 0
        total_input += len(X)

        probs = model.predict(X)
        results = (probs >= 0.4).astype(int)

        total_input = len(results)
        total_output = int(results.sum())
        if total_output > 0:
            filtered_data = data_embedding.raw_data.loc[results == 1]
            filename = f"{uuid.uuid4().hex}.csv"
            file_path = PHYCHEM_FILE / filename

            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data_embedding.raw_data.columns)
                writer.writerows(filtered_data.values.tolist())
        del data_embedding, X, probs, results
        gc.collect()

        return total_input, total_output

    @staticmethod
    def pool_run_structure(path, model):
        data_embedding = GraphEmbeddingModel(mode='classification')
        data_embedding.reload(path=STRUCTURE_CLA_EMBEDDING_PATH)
        data_embedding.load_data(paths=path)
        data_embedding.embedding()
        X = data_embedding.processed_data

        total_input, total_output = 0, 0
        total_input += len(X)

        probs = model.predict(X)
        results = (probs >= 0.5).astype(int)
        # Store filtered molecules
        if results.sum() != 0:
            import uuid
            filename = STRUCTURE_FILE / f"{uuid.uuid4().hex}.csv"
            selected = data_embedding.raw_data[results == 1]

            selected.to_csv(filename, index=False)
            total_output = len(selected)

            del data_embedding, X, probs, results, selected
            gc.collect()
            return total_input, total_output
        else:
            return total_input, 0

    @staticmethod
    def pool_run_ranking(path, model):
        data_embedding = GraphEmbeddingModel(mode='regression')
        data_embedding.reload(path=STRUCTURE_REG_EMBEDDING_PATH)
        data_embedding.load_data(paths=path)
        data_embedding.embedding()
        X = data_embedding.processed_data

        total_input, total_output = 0, 0
        total_input += len(X)

        result = model.predict(X)

        data = data_embedding.raw_data.copy()
        data['pred_rank'] = result
        import uuid
        filename = f"{uuid.uuid4().hex}.csv"
        data.to_csv(RANKING_FILE / f'{filename}', index=False)

    def add_Emp_layer(self, input_path: Path | None = None, *args: str):
        for arg in args:
            if not isinstance(arg, str):
                raise TypeError(f"Expected str, but got {type(arg).__name__}: {arg}")
            else:
                self.chemical_constraints.append(Chem.MolFromSmarts(arg))
        if input_path is not None:
            self.execution_orders.append(partial(self.run_Emp_layer, input_path=input_path))
        else:
            self.execution_orders.append(partial(self.run_Emp_layer))

    def add_phychem_layer(self, input_path: Path | None = None, model_path: Path = None):
        """Adds a Graph Convolutional Network (GCN) layer for molecular classification.

        :param model_path:
        :param input_path: The path to the folder containing input CSV files. This is required unless `self.replicate` is True.
        """
        if self.replicate:

            with open(PHYCHEM_LAYER_PATH, 'rb') as f:
                model = joblib.load(f)
        else:
            with open(model_path, 'rb') as f:
                model = joblib.load(f)

        # Add the GCN layer execution to the pipeline
        if input_path is not None:
            self.execution_orders.append(partial(self.run_phychem_layer, model=model, input_path=input_path))
        else:
            self.execution_orders.append(partial(self.run_phychem_layer, model=model))

    def add_structure_layer(self, input_path: Path | None = None, model_path: Path = None):
        """Adds a Graph Convolutional Network (GCN) layer for molecular classification.

        :param model_path:
        :param input_path: The path to the folder containing input CSV files. This is required unless `self.replicate` is True.
        """
        if self.replicate:
            with open(STRUCTURE_LAYER_PATH, 'rb') as f:
                model = joblib.load(f)
        else:
            with open(model_path, 'rb') as f:
                model = joblib.load(f)

        # Add the GCN layer execution to the pipeline
        if input_path is not None:
            self.execution_orders.append(partial(self.run_structure_layer, model=model, input_path=input_path))
        else:
            self.execution_orders.append(partial(self.run_structure_layer, model=model))

    def add_ranking_layer(self, input_path: Path | None = None, model_path: Path = None):
        """Adds a Gradient Boosting layer to the molecular filtering pipeline.

        :param model_path:
        :param input_path: The path to the folder containing input CSV files. This is required unless `self.replicate` is True.
        """
        # Load a pre-trained GB model if replication mode is enabled
        if self.replicate:
            with open(RANKING_LAYER_PATH, 'rb') as f:
                model = joblib.load(f)
        else:
            with open(model_path, 'rb') as f:
                model = joblib.load(f)

        if input_path is not None:
            self.execution_orders.append(partial(self.run_ranking_layer, model=model, input_path=input_path))
        else:
            self.execution_orders.append(partial(self.run_ranking_layer, model=model))

    def add_Clustering_layer(self, input_path: Path | None = None):
        """Adds a Clustering layer to the molecular filtering pipeline.

        :param input_path: The path to the folder containing input CSV files. This is required unless `self.replicate` is True.
        """

        if input_path is not None:
            self.execution_orders.append(partial(self.run_Clustering_layer, input_path=input_path))
        else:
            self.execution_orders.append(partial(self.run_Clustering_layer))

    def run_Emp_layer(self, input_path: Path | None = None):
        """
        Runs the Empirical (Emp) filtering layer to filter out invalid molecules based on empirical rules.

        :param input_path: The path to the folder containing input CSV files. This is required unless `self.replicate` is True.
        """
        # Print status message if verbose mode is enabled
        if self.verbose:
            print('Starting Emp_layer filtering...')
        # Determine the folder from which to read data based on the replicate flag

        searching_file = ORIGINAL_FILE if self.replicate else input_path
        if searching_file is None:
            raise Exception("Input data path is required!")

        # Delete any existing files in the Emp_FILE directory
        for file in EMP_FILE.glob("*.csv"):
            file.unlink(missing_ok=True)

        # Initialize counters for the number of molecules processed
        files = list(searching_file.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {searching_file}")

        n_proc = int(os.environ.get("LSB_DJOB_NUMPROC", cpu_count()))
        if self.verbose:
            print(f"[INFO] Using {n_proc} cores for Emp_layer file-level parallel filtering.")

        batch_size = 4
        file_batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]

        func = partial(self.pool_run_emp, chemical_constraints=self.chemical_constraints)

        with Pool(processes=n_proc) as pool:
            results = list(tqdm(pool.imap(func, file_batches), total=len(file_batches), desc="Emp Filtering"))

        total_input = sum(tin for tin, _ in results)
        total_output = sum(tout for _, tout in results)

        if self.verbose:
            print(f"Searching Space number: {total_input}, possible molecule number by Emp: {total_output}")
            print(f"The proportion of filtration: {round((total_input - total_output) / total_input * 100, 2)}%")

    def run_phychem_layer(self, model, input_path: Path | None = None):
        """Runs the Graph Convolutional Network (GCN) filtering layer to evaluate molecules using a trained GCN model.

        :param model: A trained DeepChem Graph Convolutional Network model.
        :param input_path: The path to the folder containing input CSV files.
        """
        # Determine the input file directory
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)

        if self.verbose:
            print('Starting phychem_layer filtering...')
        if self.replicate:
            searching_file = EMP_FILE
        else:
            if input_path is None:
                raise Exception("Input data path is required!")
            else:
                searching_file = input_path

        for file in PHYCHEM_FILE.glob('*.csv'):
            try:
                file.unlink()
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")

        n_proc = int(os.environ.get("LSB_DJOB_NUMPROC", cpu_count()))
        if self.verbose:
            print(f"[INFO] Using {n_proc} cores for phychem_layer file-level parallel filtering.")

        files = list(searching_file.glob('*.csv'))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {searching_file}")

        batch_size = 5
        file_batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
        func = partial(self.pool_run_phychem, model=model)

        total_input, total_output = 0, 0
        with Pool(processes=n_proc) as pool:
            results = list(tqdm(pool.imap(func, file_batches), total=len(file_batches), desc="PhyChem Filtering"))

        for tin, tout in results:
            total_input += tin
            total_output += tout

        # Print the filtering statistics if verbose mode is enabled
        if self.verbose:
            print(f'Searching Space number: {total_input}, possible molecule number by phychem: {total_output}')
            print(f'The proportion of filtration: {round((total_input - total_output) / total_input * 100, 2)}%')
            print()

    def run_structure_layer(self, model, input_path: Path | None = None):
        """Runs the Graph Convolutional Network (GCN) filtering layer to evaluate molecules using a trained GCN model.

        :param model: A trained DeepChem Graph Convolutional Network model.
        :param input_path: The path to the folder containing input CSV files.
        """
        # Determine the input file directory
        if self.verbose:
            print('Starting structure_layer filtering...')
        if self.replicate:
            searching_file = PHYCHEM_FILE
        else:
            if input_path is None:
                raise Exception("Input data path is required!")
            else:
                searching_file = input_path

        # Remove previous clustering results
        for file in STRUCTURE_FILE.glob('*.csv'):
            try:
                file.unlink()
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")

        # Initialize counters for the number of molecules processed
        files = list(searching_file.glob('*.csv'))
        batch_size = 2
        file_batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]

        n_proc = int(os.environ.get("LSB_DJOB_NUMPROC", cpu_count()))
        func = partial(self.pool_run_structure, model=model)

        total_input, total_output = 0, 0
        with Pool(processes=n_proc) as pool:
            results = list(tqdm(pool.imap(func, file_batches), total=len(file_batches), desc="Structure Filtering"))

        for tin, tout in results:
            total_input += tin
            total_output += tout

        # Print the filtering statistics if verbose mode is enabled
        if self.verbose:
            print(f'Searching Space number: {total_input}, possible molecule number by structure: {total_output}')
            print(f'The proportion of filtration: {round((total_input - total_output) / total_input * 100, 2)}%')
            print()

    def run_ranking_layer(self, model, input_path: Path | None = None):
        """Runs the Gradient Boosting (GB) filtering layer to evaluate molecules using a trained GB model.

        :param model:
        :param input_path: The path to the folder containing input CSV files. This is required unless `self.replicate` is True.
        """
        # Print status message if verbose mode is enabled
        # Determine the input file directory
        if self.verbose:
            print('Starting ranking_layer filtering...')
        if self.replicate:
            searching_file = STRUCTURE_FILE
        else:
            if input_path is None:
                raise Exception("Input data path is required!")
            else:
                searching_file = input_path

        for file in RANKING_FILE.glob('*.csv'):
            try:
                file.unlink()
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")

        # Initialize counters for the number of molecules processed
        files = list(searching_file.glob('*.csv'))
        batch_size = 1
        file_batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]

        n_proc = int(os.environ.get("LSB_DJOB_NUMPROC", cpu_count()))
        func = partial(self.pool_run_ranking, model=model)
        with Pool(processes=n_proc) as pool:
            list(tqdm(pool.imap(func, file_batches), total=len(file_batches), desc="ranking Filtering"))

        all_files = list(RANKING_FILE.glob('*.csv'))
        if not all_files:
            print("[INFO] No ranking files found.")
            return
        df_list = [pd.read_csv(f) for f in all_files]
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df = combined_df.sort_values('pred_rank', ascending=True)
        top_df = combined_df.head(1000)

        for f in all_files:
            try:
                f.unlink()
            except Exception as e:
                print(f"[WARN] Failed to delete {f}: {e}")
        final_file = RANKING_FILE / 'ranking_top.csv'
        top_df.to_csv(final_file, index=False)
        print(f"[INFO] Saved top ranking molecules to {final_file}")

    def run_Clustering_layer(self, input_path: Path | None = None):
        """Runs the Clustering Layer to filter and group molecules based on structural similarities.

        :param input_path: Path to the input CSV file (defaults to the output of GB layer).
        """
        if self.verbose:
            print('Starting Clustering_layer filtering...')
        from rdkit.Chem import rdFMCS

        if self.replicate:
            searching_file = RANKING_FILE
        else:
            if input_path is None:
                raise Exception("Input data path is required!")
            else:
                searching_file = input_path

        folder = CLU_FILE
        try:
            shutil.rmtree(folder)
            print(f"Deleted all contents in {folder}")
        except Exception as e:
            print(f"Error deleting {folder}: {e}")
        folder.mkdir(parents=True, exist_ok=True)

        files = list(searching_file.glob('*.csv'))
        Clustering_data = []
        for file in files:
            data = pd.read_csv(file)
            if data.empty:
                continue
            Clustering_data.append(data)
        if Clustering_data:
            Clustering_data = pd.concat(Clustering_data, ignore_index=True)
        else:
            Clustering_data = pd.DataFrame()

        Clustering_smiles = np.array(Clustering_data[['smiles']])
        # Compute molecular fingerprints
        Fingerprint = []
        for smiles in tqdm(Clustering_smiles, desc='Clustering'):
            mol = Chem.MolFromSmiles(smiles[0])
            RDKFingerprint = Chem.RDKFingerprint(mol, minPath=1, fpSize=8192)
            Fingerprint.append(RDKFingerprint)

        clusters = self.ClusterByFp(Fingerprint, threshold=0.3)

        # Remove tuples whose second dimension length is 1 or 2, remove trivial clusters (singletons or pairs)
        filtered_clusters = [tup for tup in clusters if len(tup) != 1]
        print('[INFO] Finished cleaning one-cluster.')
        filtered_clusters = [tup for tup in filtered_clusters if len(tup) != 2]
        print('[INFO] Finished cleaning two-cluster.')

        unique_group = []
        iterator = tqdm(filtered_clusters, desc='FindingMCS',
                        bar_format="{l_bar}{bar}| {remaining}") if self.verbose else filtered_clusters

        for filtered_cluster in iterator:
            filtered_smiles = []
            for index in filtered_cluster:
                smile = Clustering_data['smiles'][index]
                filtered_smiles.append(Chem.MolFromSmiles(smile))
            pairs = [(filtered_smiles[i], filtered_smiles[j]) for i in range(len(filtered_smiles)) for j in
                     range(i + 1, len(filtered_smiles))]
            for pair in pairs:
                mcs = rdFMCS.FindMCS(pair)
                mcs = Chem.MolFromSmarts(mcs.smartsString)
                if all(not self.is_same_mcs(mcs, mol) for mol in unique_group):
                    unique_group.append(mcs)

        # Filter out those with only two.
        unique_group_filtered_1 = []
        for i in range(len(unique_group)):
            index = 0
            for Clustering_smile in Clustering_smiles:
                if Chem.MolFromSmiles(Clustering_smile[0]).HasSubstructMatch(unique_group[i]):
                    index += 1
            if index > 2:
                unique_group_filtered_1.append(unique_group[i])
        print('[INFO] Finished removed Minority MCS.')

        # Further remove basic structures
        unique_group_filtered_2 = []
        for unique_group1 in unique_group_filtered_1:
            index = 0
            for unique_group2 in unique_group_filtered_1:
                if unique_group2.HasSubstructMatch(unique_group1):
                    index += 1
            if index == 1:
                unique_group_filtered_2.append(unique_group1)
        print('[INFO] Finished removed basic structures.')

        if self.verbose:
            print(f'unique_group -> {len(unique_group_filtered_2)}')

        with open(CLU_FILE / rf"Molecular.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['SMILES', 'Grade'])

        with open(CLU_FILE / rf"MCS.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['SMILES', 'Final_Score', 'Mean', 'Var', 'Number'])

        unique_group_index = 0
        Mol_mcs = []  # Obtain the molecular chemical formula
        Mean_Grade = []  # The mean of the predicted scores
        Var_Grade = []  # The variance of predicted scores
        Number = []  # The number of contained molecules
        # Save MCS results and corresponding molecules
        for mcs in unique_group_filtered_2:
            unique_group_index += 1
            unique_group_file = CLU_FILE / rf"MCS{unique_group_index}"

            os.makedirs(unique_group_file, exist_ok=True)
            McsImage = MolToImage(mcs)
            McsImage.save(unique_group_file / rf'MCS{unique_group_index}.png')
            McsImage.save(CLU_FILE / rf'MCS{unique_group_index}.png')

            Grade = []
            if self.verbose:
                print()
                print(f'MCS: {Chem.MolToSmiles(mcs)}')
            has_mcs_index = -1
            for smiles in Clustering_smiles:
                has_mcs_index += 1
                if Chem.MolFromSmiles(smiles[0]).HasSubstructMatch(mcs):
                    if self.verbose:
                        print(smiles[0])
                    McsImage = MolToImage(Chem.MolFromSmiles(smiles[0]))
                    McsImage.save(unique_group_file / rf'{has_mcs_index + 1}.png')
                    Grade.append(Clustering_data['pred_rank'][has_mcs_index])

            Mol_mcs.append(mcs)
            Mean_Grade.append(np.mean(Grade))
            Var_Grade.append(np.var(Grade))
            Number.append(len(Grade))
        # Ranking based on Mean Grade, Variance, and Cluster Size
        Mean_sorted = np.array([sorted(Mean_Grade, reverse=True).index(x) for x in Mean_Grade])
        Var_sorted = np.array([sorted(Var_Grade, reverse=True).index(x) for x in Var_Grade])
        Number_sorted = np.array([sorted(Number).index(x) for x in Number])
        # Number_sorted with double weight.
        Average_Grade = np.sum([Mean_sorted, Mean_sorted, Var_sorted, Number_sorted, Number_sorted, Number_sorted], axis=0)
        # Save rankings to CSV
        for no in range(len(Clustering_smiles)):
            with open(CLU_FILE / rf"Molecular.csv", 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([Clustering_smiles[no][0], Clustering_data['pred_rank'][no]])
        print('[INFO] Molecular.csv finished writing.')

        for no in range(len(Average_Grade)):
            with open(CLU_FILE / rf"MCS.csv", 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [Chem.MolToSmiles(Mol_mcs[no]), Average_Grade[no], Mean_Grade[no], Var_Grade[no], Number[no]])
        print('[INFO] MCS.csv finished writing.')

    def run(self):
        """Executes all functions in `self.execution_orders` sequentially."""
        for execution_order in self.execution_orders:
            execution_order()
