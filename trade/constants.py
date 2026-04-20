"""Contains constants shared throughout trade."""
from importlib import resources
from typing import Literal
from pathlib import Path

N_TASKS = 1
MODE = Literal['classification', 'regression']
INPUT_TYPE = Literal['phychem', 'structure']
DROPOUT = 0.2
GRAPH_CONV_LAYER = [64, 64]
DENSE_LAYER_SIZE = 128
NUMBER_ATOM_FEATURES = 27
N_CLASSES = 2
SEED = 0
TRAIN_RATIO, TEST_RATIO = 0.7, 0.3
TASK = ['Anti', 'MIC']
FEATURE_FIELD = "smiles"
DEFAULT_THRESHOLD = 0.5
DEFAULT_NB_EPOCH = 1000
DEFAULT_ROLLS = 12
DEFAULT_FEATURES = ['MaxAbsEStateIndex', 'MaxEStateIndex',
                    'MinAbsEStateIndex', 'MinEStateIndex', 'qed', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',
                    'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge',
                    'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'BCUT2D_MWHI', 'BCUT2D_MWLOW',
                    'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW',
                    'AvgIpc', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n',
                    'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3',
                    'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14',
                    'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8',
                    'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6',
                    'SMR_VSA7', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2',
                    'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'EState_VSA1',
                    'EState_VSA10', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6',
                    'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2',
                    'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8',
                    'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticHeterocycles',
                    'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings',
                    'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'MolMR',
                    'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH',
                    'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_NH0', 'fr_NH1', 'fr_NH2',
                    'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_alkyl_halide', 'fr_amide',
                    'fr_aniline', 'fr_aryl_methyl', 'fr_benzene', 'fr_bicyclic', 'fr_ester', 'fr_ether', 'fr_guanido',
                    'fr_halogen', 'fr_hdrzine', 'fr_imidazole', 'fr_methoxy', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom',
                    'fr_nitro_arom_nonortho', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond',
                    'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_thiazole']
DEFAULT_N_COMPONENTS = 40
DEFAULT_CCP_ALPHA = 0
DEFAULT_MAX_FEATURES = 'sqrt'
DEFAULT_MAX_SAMPLES = 0.7
DEFAULT_MIN_IMPURITY_DECREASE = 0
DEFAULT_MIN_SAMPLES_SPLIT = 3
DEFAULT_MIN_WEIGHT_FRACTION_LEAF = 0
DEFAULT_N_ESTIMATORS = 5000
DEFAULT_OOB_SCORE = True
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_PRE_LR = 1e-4
DEFAULT_FINETUNE_LR = 1e-4
DEFAULT_MAX_DEPTH = 7
DEFAULT_GAMMA = 0.5
DEFAULT_MIN_CHILD_WEIGHT = 1
DEFAULT_COLSAMPLE_BYTREE = 1
DEFAULT_COLSAMPLE_BYLEVEL = 0.6
DEFAULT_COLSAMPLE_BYNODE = 1
DEFAULT_DELTA_STEP = 0
DEFAULT_REG_ALPHA = 1
DEFAULT_REG_LAMBDA = 1
DEFAULT_RANDOM_STATE = 0
DEFAULT_BASE_SCORE = 0.39534
DEFAULT_CALLBACKS = None
DEFAULT_EARLY_STOPPING_ROUNDS = None
DEFAULT_ENABLE_CATEGORICAL = False
DEFAULT_GROW_POLICY = None
DEFAULT_IMPORTANCE_TYPE = None
DEFAULT_MAX_BIN = None
DEFAULT_MAX_CAT_THRESHOLD = None
DEFAULT_MAX_CAT_TO_ONEHOT = None
DEFAULT_MAX_LEAVES = None
DEFAULT_MULTI_STRATEGY = None
DEFAULT_NUM_PARALLEL_TREE = None
DEFAULT_FIT_INTERCEPT = False
DEFAULT_KERNEL = 'rbf'
DEFAULT_EPSILON = 0.1
DEFAULT_ACTIVATION_FUNCTION = 'relu'
DEFAULT_UNIT = 64
DEFAULT_OPTIMIZER = 'adam'
DEFAULT_LOSS = 'mean_squared_error'
DEFAULT_BATCH_SIZE = 16
DEFAULT_N_NEIGHBORS = 5
DEFAULT_ALGORITHM = 'SAMME.R'
DEFAULT_EPOCHS = 50

# Path where the data files are stored
with resources.path('trade', 'Data') as resources_dir:
    DATA_DIR = resources_dir

SOURCE_SET_PATH = Path(DATA_DIR / 'TrainingSet' / 'SourceDataset.csv')
TARGET_SET_PATH = Path(DATA_DIR / 'TrainingSet' / 'TargetDataset.csv')

DOWNLOAD_FILE = Path(DATA_DIR / 'Download_data')
ORIGINAL_FILE = Path(DATA_DIR / 'Original_space')
EMP_FILE = Path(DATA_DIR / 'After_Emp')
PHYCHEM_FILE = Path(DATA_DIR / 'After_phychem')
STRUCTURE_FILE = Path(DATA_DIR / 'After_structure')
RANKING_FILE = Path(DATA_DIR / 'After_ranking')
CLU_FILE = Path(DATA_DIR / 'After_Clustering')

DESCRIPTORS_EMBEDDING_PATH = Path(DATA_DIR / 'Models' / 'Descriptors_embedding.pkl')
PHYCHEM_CLA_EMBEDDING_PATH = Path(DATA_DIR / 'Models' / 'phychem_embedding(cla).pkl')
STRUCTURE_CLA_EMBEDDING_PATH = Path(DATA_DIR / 'Models' / 'structure_embedding(cla).pkl')
PHYCHEM_REG_EMBEDDING_PATH = Path(DATA_DIR / 'Models' / 'phychem_embedding(reg).pkl')
STRUCTURE_REG_EMBEDDING_PATH = Path(DATA_DIR / 'Models' / 'structure_embedding(reg).pkl')

PHYCHEM_LAYER_PATH = Path(DATA_DIR / 'Models' / 'phychem_layer.pkl')
STRUCTURE_LAYER_PATH = Path(DATA_DIR / 'Models' / 'structure_layer.pkl')
RANKING_LAYER_PATH = Path(DATA_DIR / 'Models' / 'ranking_layer.pkl')
