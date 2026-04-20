import io
import time
import random
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

from funscrmol.constants import DESCRIPTORS_EMBEDDING_PATH
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from trade.constants import (DEFAULT_FEATURES, TEST_RATIO, SOURCE_SET_PATH, STRUCTURE_REG_EMBEDDING_PATH,
                             STRUCTURE_CLA_EMBEDDING_PATH, PHYCHEM_CLA_EMBEDDING_PATH, MODE, PHYCHEM_REG_EMBEDDING_PATH)
from contextlib import redirect_stdout, redirect_stderr
f = io.StringIO()
with redirect_stdout(f), redirect_stderr(f):
    import deepchem as dc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
featurizer = dc.feat.ConvMolFeaturizer()
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)


class DescriptorsEmbeddingModel:
    def __init__(self, mode: MODE | None, raw_data: pd.DataFrame = None, features: list = DEFAULT_FEATURES):
        """Creates the Generator.

        :param raw_data: The raw data that need to be processed.
        """
        self.mode = mode
        self.raw_data = raw_data
        self.features = features
        self.processed_data = None
        self.scaler = None
        self.pca = None

    def load_data(self, paths: list[Path]):
        merged_data = []
        for path in paths:
            data = pd.read_csv(path)
            merged_data.append(data)
        self.raw_data = pd.concat(merged_data, ignore_index=True)

    def save(self, path: Path) -> None:
        joblib.dump({
            'scaler': self.scaler,
            'pca': self.pca,
            'features': self.features
        }, path)

    def reload(self, path: Path) -> None:
        state = joblib.load(path)
        self.scaler = state['scaler']
        self.pca = state['pca']
        self.features = state['features']

    def fit_model(self, n_components: int):

        X = self.raw_data[self.features]
        # Apply PCA for feature dimensionality reduction
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.pca = PCA(n_components=n_components)
        self.pca.fit(X_scaled)

    def embedding(self) -> pd.DataFrame:

        if 'Anti' in self.raw_data.columns:
            X = self.raw_data[DEFAULT_FEATURES]
            X_scaled = self.scaler.transform(X)
            embeddings = self.pca.transform(X_scaled)
            X_pca = pd.DataFrame(embeddings, columns=[f"PC{i}" for i in range(embeddings.shape[1])])
            X_pca['Anti'] = self.raw_data[['Anti']].values
            X_pca['Activity'] = self.raw_data[['Activity']].values
            self.processed_data = X_pca
        else:
            X = self.raw_data[DEFAULT_FEATURES]
            X_scaled = self.scaler.transform(X)
            embeddings = self.pca.transform(X_scaled)
            X_pca = pd.DataFrame(embeddings, columns=[f"PC{i}" for i in range(embeddings.shape[1])])
            self.processed_data = X_pca
        return self.processed_data

    def split(self, split_rate: float = TEST_RATIO) -> tuple:
        TrainingSet_grouped = self.processed_data.groupby('Anti')
        X_P, Y_P, X_N, Y_N = None, None, None, None
        random_int = random.randint(1, 10)
        for group_names, grouped_data in TrainingSet_grouped:
            if str(group_names) == '1':
                X_P = grouped_data.drop(columns=['Anti', 'Activity'])
                if self.mode == 'classification':
                    Y_P = grouped_data[['Anti']]
                elif self.mode == 'regression':
                    Y_P = grouped_data[['Activity']]
                else:
                    raise ValueError(f"Unexpected mode: {group_names}")
            elif str(group_names) == '0':
                X_N = grouped_data.drop(columns=['Anti', 'Activity'])
                if self.mode == 'classification':
                    Y_N = grouped_data[['Anti']]
                elif self.mode == 'regression':
                    Y_N = grouped_data[['Activity']]
                else:
                    raise ValueError(f"Unexpected mode: {group_names}")
            else:
                raise ValueError(f"Unexpected Anti type: {group_names}")

        # Apply PCA for feature dimensionality reduction
        x_train_P, x_test_P, y_train_P, y_test_P = train_test_split(X_P, Y_P, test_size=split_rate,
                                                                    random_state=random_int)
        x_train_N, x_test_N, y_train_N, y_test_N = train_test_split(X_N, Y_N, test_size=split_rate,
                                                                    random_state=random_int)
        x_train = pd.concat([x_train_P, x_train_N], axis=0)
        y_train = pd.concat([y_train_P, y_train_N], axis=0)
        x_test = pd.concat([x_test_P, x_test_N], axis=0)
        y_test = pd.concat([y_test_P, y_test_N], axis=0)

        return x_train, y_train, x_test, y_test


class PhyChemEmbeddingModel(nn.Module):
    def __init__(self, mode: MODE, raw_data: pd.DataFrame = None, features: list = DEFAULT_FEATURES,
                 input_dim: int = 155, hidden_dim: int = 43, epochs: int = 100000, lr: float = 1e-3):
        """Creates the Generator.

        :param raw_data: The raw data that need to be processed.
        """
        super().__init__()
        self.mode = mode
        self.raw_data = raw_data
        self.features = features
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.processed_data = None
        self.encoder = None
        self.model = None

    def load_data(self, paths: list[Path]):
        merged_data = []
        for path in paths:
            data = pd.read_csv(path)
            if data.empty:
                continue
            merged_data.append(data)

        if merged_data:
            self.raw_data = pd.concat(merged_data, ignore_index=True)
        else:
            print("[WARNING] All files are empty!")
            self.raw_data = pd.DataFrame()

    def save(self, path: Path) -> None:
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'mode': self.mode,
            'features': self.features
        }, path)

    def reload(self, path: Path) -> None:
        checkpoint = torch.load(path)
        self.input_dim = checkpoint['input_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.mode = checkpoint['mode']
        self.features = checkpoint['features']

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, 1)
        )
        self.model = nn.Sequential(self.encoder, predictor)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def fit_model(self, n_components: int):
        self.hidden_dim = n_components

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, 1)
        )
        self.model = nn.Sequential(self.encoder, predictor)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=200
        )

        X = self.raw_data[DEFAULT_FEATURES]
        if self.mode == 'classification':
            Y = self.raw_data['Anti']
            loss_fn = nn.BCEWithLogitsLoss()
        elif self.mode == 'regression':
            Y = self.raw_data['Activity']
            loss_fn = nn.MSELoss()
        else:
            raise ValueError("Error type of the model.")

        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        Y_tensor = torch.tensor(Y.values, dtype=torch.float32).view(-1, 1)

        mask = torch.isfinite(X_tensor).all(dim=1) & torch.isfinite(Y_tensor).squeeze()
        X_tensor = X_tensor[mask]
        Y_tensor = Y_tensor[mask]

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = loss_fn(output, Y_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

    def embedding(self) -> pd.DataFrame:

        if self.model is None:
            raise RuntimeError("Encoder has not been trained. Please call fit_model() before embedding.")

        if 'Anti' in self.raw_data.columns:
            X = self.raw_data[DEFAULT_FEATURES]
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
            self.encoder.eval()
            with torch.no_grad():
                embeddings = self.encoder(X_tensor).numpy()
            self.processed_data = pd.DataFrame(embeddings, columns=[f"PC{i}" for i in range(embeddings.shape[1])])
            self.processed_data["Anti"] = self.raw_data["Anti"].values
            self.processed_data["Activity"] = self.raw_data["Activity"].values
            return self.processed_data
        else:
            X = self.raw_data[DEFAULT_FEATURES]
            X_numeric = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
            X_tensor = torch.tensor(X_numeric.values, dtype=torch.float32)
            self.encoder.eval()
            with torch.no_grad():
                embeddings = self.encoder(X_tensor).numpy()
            self.processed_data = pd.DataFrame(embeddings, columns=[f"PC{i}" for i in range(embeddings.shape[1])])
            return self.processed_data

    def split(self, split_rate: float = TEST_RATIO) -> tuple:
        TrainingSet_grouped = self.processed_data.groupby('Anti')
        X_P, Y_P, X_N, Y_N = None, None, None, None
        random_int = random.randint(1, 10)
        for group_names, grouped_data in TrainingSet_grouped:
            if str(group_names) == '1':
                X_P = grouped_data.drop(columns=['Anti', 'Activity'])
                if self.mode == 'classification':
                    Y_P = grouped_data[['Anti']]
                elif self.mode == 'regression':
                    Y_P = grouped_data[['Activity']]
                else:
                    raise ValueError(f"Unexpected mode: {group_names}")
            elif str(group_names) == '0':
                X_N = grouped_data.drop(columns=['Anti', 'Activity'])
                if self.mode == 'classification':
                    Y_N = grouped_data[['Anti']]
                elif self.mode == 'regression':
                    Y_N = grouped_data[['Activity']]
                else:
                    raise ValueError(f"Unexpected mode: {group_names}")
        # Apply PCA for feature dimensionality reduction
        x_train_P, x_test_P, y_train_P, y_test_P = train_test_split(X_P, Y_P, test_size=split_rate,
                                                                    random_state=random_int)
        x_train_N, x_test_N, y_train_N, y_test_N = train_test_split(X_N, Y_N, test_size=split_rate,
                                                                    random_state=random_int)
        x_train = pd.concat([x_train_P, x_train_N], axis=0)
        y_train = pd.concat([y_train_P, y_train_N], axis=0)
        x_test = pd.concat([x_test_P, x_test_N], axis=0)
        y_test = pd.concat([y_test_P, y_test_N], axis=0)

        return x_train, y_train, x_test, y_test


class FingerprintEmbeddingModel:
    FINGERPRINT_FUNCS = {
        "RDK": lambda mol, fpSize: Chem.RDKFingerprint(mol, fpSize=fpSize),
        "Morgan": lambda mol, fpSize: AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fpSize),
        "MACCS": lambda mol, fpSize=167: MACCSkeys.GenMACCSKeys(mol),
        "AtomPair": lambda mol, fpSize: AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=fpSize),
    }

    def __init__(self, mode: MODE, raw_data: pd.DataFrame = None, ):

        self.mode = mode
        self.raw_data = raw_data
        self.processed_data = None

    @staticmethod
    def dataset_transfer(molecules: list[str], labels: list = None) -> tuple:
        # Training mode
        if labels is not None:
            molecule_list = []
            valid_labels = []
            for mol, label in zip([Chem.MolFromSmiles(s) for s in molecules], labels):
                if mol is not None:
                    molecule_list.append(mol)
                    valid_labels.append(label)
            return molecule_list, valid_labels
        # Screening mode
        else:
            valid_molecules = []
            valid_indices = []
            for index, smiles in enumerate(molecules):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_molecules.append(mol)
                    valid_indices.append(index)

            return valid_molecules, valid_indices

    def load_data(self, paths: list[Path]):
        merged_data = []
        for path in paths:
            data = pd.read_csv(path)
            merged_data.append(data)
        self.raw_data = pd.concat(merged_data, ignore_index=True)

    def embedding(self, fpSize: int, fingerprint: str = 'RDK') -> pd.DataFrame | tuple:

        if fingerprint not in self.FINGERPRINT_FUNCS:
            raise ValueError(f"Unsupported fingerprint: {fingerprint}")

        fp_function = self.FINGERPRINT_FUNCS[fingerprint]

        if 'Anti' in self.raw_data.columns:
            if self.mode == 'classification':
                dataset, labels = self.dataset_transfer(molecules=self.raw_data['smiles'], labels=self.raw_data['Anti'])
            elif self.mode == 'regression':
                dataset, labels = self.dataset_transfer(molecules=self.raw_data['smiles'],
                                                        labels=self.raw_data['Activity'])
            else:
                raise ValueError("Error type of the model.")
            embeddings = []
            for mol in dataset:
                fp = fp_function(mol, fpSize=fpSize)
                fp_list = list(fp)
                embeddings.append(fp_list)
            self.processed_data = pd.DataFrame(embeddings, columns=[f"fp{i}" for i in range(fpSize)])
            self.processed_data["Anti"] = self.raw_data["Anti"].values[:len(dataset)]
            self.processed_data["Activity"] = self.raw_data["Activity"].values[:len(dataset)]

            return self.processed_data, None
        else:
            dataset, _ = self.dataset_transfer(molecules=self.raw_data['smiles'])
            embeddings = []
            for mol in dataset:
                fp = fp_function(mol, fpSize=fpSize)
                fp_list = list(fp)
                embeddings.append(fp_list)
            self.processed_data = pd.DataFrame(embeddings, columns=[f"fp{i}" for i in range(fpSize)])
            return self.processed_data

    def split(self, split_rate: float = TEST_RATIO) -> tuple:
        TrainingSet_grouped = self.processed_data.groupby('Anti')
        X_P, Y_P, X_N, Y_N = None, None, None, None
        random_int = random.randint(1, 10)
        for group_names, grouped_data in TrainingSet_grouped:
            if str(group_names) == '1':
                X_P = grouped_data.drop(columns=['Anti', 'Activity'])
                if self.mode == 'classification':
                    Y_P = grouped_data[['Anti']]
                elif self.mode == 'regression':
                    Y_P = grouped_data[['Activity']]
                else:
                    raise ValueError(f"Unexpected mode: {group_names}")
            elif str(group_names) == '0':
                X_N = grouped_data.drop(columns=['Anti', 'Activity'])
                if self.mode == 'classification':
                    Y_N = grouped_data[['Anti']]
                elif self.mode == 'regression':
                    Y_N = grouped_data[['Activity']]
                else:
                    raise ValueError(f"Unexpected mode: {group_names}")

        # Apply PCA for feature dimensionality reduction
        x_train_P, x_test_P, y_train_P, y_test_P = train_test_split(X_P, Y_P, test_size=split_rate,
                                                                    random_state=random_int)
        x_train_N, x_test_N, y_train_N, y_test_N = train_test_split(X_N, Y_N, test_size=split_rate,
                                                                    random_state=random_int)
        x_train = pd.concat([x_train_P, x_train_N], axis=0)
        y_train = pd.concat([y_train_P, y_train_N], axis=0)
        x_test = pd.concat([x_test_P, x_test_N], axis=0)
        y_test = pd.concat([y_test_P, y_test_N], axis=0)

        return x_train, y_train, x_test, y_test


class GraphEmbeddingModel(dc.models.GraphConvModel):
    def __init__(self, n_tasks=1, mode: MODE = None, raw_data: pd.DataFrame = None, epochs: int = 500,
                 **kwargs):
        super().__init__(n_tasks=n_tasks, mode=str(mode), graph_conv_layers=[64, 64, 64, 128, 128],
                         dense_layer_size=256, dropout=0.1, **kwargs)
        self.mode = mode
        self.raw_data = raw_data
        self.epochs = epochs
        self.processed_data = None

    @staticmethod
    def dataset_transfer(molecules: list[str], labels: list = None) -> dc.data.NumpyDataset | tuple:
        # Training mode
        if labels is not None:
            molecule_list = []
            valid_labels = []
            for mol, label in zip([Chem.MolFromSmiles(s) for s in molecules], labels):
                if mol is not None:
                    molecule_list.append(mol)
                    valid_labels.append(label)
            features = featurizer.featurize(molecules)
            dataset = dc.data.NumpyDataset(X=features, y=np.array(valid_labels))
            return dataset
        # Screening mode
        else:
            valid_molecules = []
            valid_indices = []
            for index, smiles in enumerate(molecules):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_molecules.append(mol)
                    valid_indices.append(index)

            features = featurizer.featurize(valid_molecules)
            dataset = dc.data.NumpyDataset(X=features)
            return dataset, valid_indices

    def load_data(self, paths: list[Path]):
        merged_data = []
        for path in paths:
            data = pd.read_csv(path)
            if data.empty:
                continue
            merged_data.append(data)

        if merged_data:
            self.raw_data = pd.concat(merged_data, ignore_index=True)
        else:
            print("[WARNING] All files are empty!")
            self.raw_data = pd.DataFrame()

    def save(self, path: Path) -> None:
        """Saves model to disk using joblib."""
        joblib.dump(self.model, path)

    def reload(self, path: Path) -> None:
        """Loads model from joblib file on disk."""
        self.model = joblib.load(path)

    def fit_model(self) -> None:
        if self.mode == 'classification':
            dataset = self.dataset_transfer(molecules=self.raw_data['smiles'], labels=self.raw_data['Anti'])
            self.loss = dc.models.losses.SigmoidCrossEntropy()
        elif self.mode == 'regression':
            dataset = self.dataset_transfer(molecules=self.raw_data['smiles'], labels=self.raw_data['Activity'])
            self.loss = dc.models.losses.L2Loss()
        else:
            raise ValueError("Error type of the model.")

        losses = []
        for epoch in range(1, self.epochs + 1):
            loss = self.fit(dataset, nb_epoch=1)
            losses.append(loss)
        self.model.summary()

    def embedding(self) -> tuple:
        if not self.model.built:
            raise RuntimeError("Model has not been trained. Please call fit_model() before embedding.")

        if 'Anti' in self.raw_data.columns:
            if self.mode == 'classification':
                dataset = self.dataset_transfer(molecules=self.raw_data['smiles'], labels=self.raw_data['Anti'])
            elif self.mode == 'regression':
                dataset = self.dataset_transfer(molecules=self.raw_data['smiles'], labels=self.raw_data['Activity'])
            else:
                raise ValueError("Error type of the model.")

            embeddings = self.predict_embedding(dataset)
            embeddings = embeddings[:len(dataset)]
            self.processed_data = pd.DataFrame(embeddings, columns=[f"PC{i}" for i in range(embeddings.shape[1])])
            self.processed_data["Anti"] = self.raw_data["Anti"].values[:len(dataset)]
            self.processed_data["Activity"] = self.raw_data["Activity"].values[:len(dataset)]
            return self.processed_data, None
        else:
            dataset, valid_indices = self.dataset_transfer(molecules=self.raw_data['smiles'])
            embeddings = self.predict_embedding(dataset)
            embeddings = embeddings[:len(dataset)]
            self.processed_data = pd.DataFrame(embeddings, columns=[f"PC{i}" for i in range(embeddings.shape[1])])
            return self.processed_data, valid_indices

    def split(self, split_rate: float = TEST_RATIO) -> tuple:
        TrainingSet_grouped = self.processed_data.groupby('Anti')
        X_P, Y_P, X_N, Y_N = None, None, None, None
        random_int = random.randint(1, 10)
        for group_names, grouped_data in TrainingSet_grouped:
            if str(group_names) == '1':
                X_P = grouped_data.drop(columns=['Anti', 'Activity'])
                if self.mode == 'classification':
                    Y_P = grouped_data[['Anti']]
                elif self.mode == 'regression':
                    Y_P = grouped_data[['Activity']]
                else:
                    raise ValueError(f"Unexpected mode: {group_names}")
            elif str(group_names) == '0':
                X_N = grouped_data.drop(columns=['Anti', 'Activity'])
                if self.mode == 'classification':
                    Y_N = grouped_data[['Anti']]
                elif self.mode == 'regression':
                    Y_N = grouped_data[['Activity']]
                else:
                    raise ValueError(f"Unexpected mode: {group_names}")

        # Apply PCA for feature dimensionality reduction
        x_train_P, x_test_P, y_train_P, y_test_P = train_test_split(X_P, Y_P, test_size=split_rate,
                                                                    random_state=random_int)
        x_train_N, x_test_N, y_train_N, y_test_N = train_test_split(X_N, Y_N, test_size=split_rate,
                                                                    random_state=random_int)
        x_train = pd.concat([x_train_P, x_train_N], axis=0)
        y_train = pd.concat([y_train_P, y_train_N], axis=0)
        x_test = pd.concat([x_test_P, x_test_N], axis=0)
        y_test = pd.concat([y_test_P, y_test_N], axis=0)

        return x_train, y_train, x_test, y_test


def feature_engineering():

    data_embedding = DescriptorsEmbeddingModel(mode=None)
    data_embedding.load_data(paths=[SOURCE_SET_PATH])
    data_embedding.fit_model(n_components=43)
    data_embedding.save(path=DESCRIPTORS_EMBEDDING_PATH)
    print(f'PhyChemEmbeddingModel (n={43}) saved.')

    # Classification-embedding model fitting.
    print(f'PhyChemEmbeddingModel starting...')
    data_embedding = PhyChemEmbeddingModel(mode='classification', input_dim=155, hidden_dim=43, epochs=10000, lr=1e-4)
    data_embedding.load_data(paths=[SOURCE_SET_PATH])
    data_embedding.fit_model(n_components=43)
    data_embedding.save(path=PHYCHEM_CLA_EMBEDDING_PATH)
    print(f'PhyChemEmbeddingModel (n={43}) saved.')

    # Regression-embedding model fitting.
    data_embedding = PhyChemEmbeddingModel(mode='regression', input_dim=155, hidden_dim=43, epochs=40000, lr=1e-3)
    data_embedding.load_data(paths=[SOURCE_SET_PATH])
    data_embedding.fit_model(n_components=43)
    data_embedding.save(path=PHYCHEM_REG_EMBEDDING_PATH)
    print(f'PhyChemEmbeddingModel (n={43}) saved.')

    # Classification-embedding model fitting.
    print(f'GraphEmbeddingModel starting...')
    data_embedding = GraphEmbeddingModel(mode='classification')
    data_embedding.load_data(paths=[SOURCE_SET_PATH])
    data_embedding.fit_model()
    data_embedding.save(path=STRUCTURE_CLA_EMBEDDING_PATH)
    print('GraphEmbeddingModel saved.')

    # Regression-embedding model fitting.
    print(f'GraphEmbeddingModel starting...')
    data_embedding = GraphEmbeddingModel(mode='regression')
    data_embedding.load_data(paths=[SOURCE_SET_PATH])
    data_embedding.fit_model()
    data_embedding.save(path=STRUCTURE_REG_EMBEDDING_PATH)
    print('GraphEmbeddingModel saved.')


if __name__ == '__main__':

    from tap import tapify
    tapify(feature_engineering)
