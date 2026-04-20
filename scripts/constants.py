from trade import constants

DOWNLOAD_PATH = constants.DOWNLOAD_FILE
DEFAULT_FEATURES = constants.DEFAULT_FEATURES
DATA_DIR = constants.DATA_DIR
SHAP_DIR = DATA_DIR / 'SHAP.csv'
ANTIBIOTIC_PATH = DATA_DIR / 'Antibiotic_data'
DEFAULT_COLUMNS = ['Compound_CID', 'SMILES', 'Activity']

UNLABELED_FILE = ['Anti-Infective Agent', 'Antibiotic', 'Antimicrobial', 'J01']
LABELED_FILE = {
    'Tetracyclines': ['Tetracyclines', 'J01A'],
    'Amphenicols': ['Amphenicols', 'J01B'],
    'Penicillins': ['Penicillins', 'Beta-lactam', 'J01C', 'J01D'],
    'Carbapenems': ['Carbapenems', 'Beta-lactam', 'J01C', 'J01D'],
    'Cephalosporins': ['Cephalosporins', 'Beta-lactam', 'J01C', 'J01D'],
    'DHFR_Inhibitor': ['DHFR_Inhibitor', 'J01E'],
    'Sulfonamides': ['Sulfonamides', 'J01E'],
    'Macrolides': ['Macrolides', 'J01F'],
    'Lincosamides': ['Lincosamides', 'J01F'],
    'Aminoglycoside': ['Aminoglycoside', 'J01G'],
    'Quinolines': ['Quinolines', 'J01M'],
    'Pleuromutilin': ['Pleuromutilin', 'J01X'],
    'Oxazolidinone': ['Oxazolidinone', 'J01X'],
    'Nitroimidazole': ['Nitroimidazole', 'J01X'],
    'Nitrofuran': ['Nitrofuran', 'Furan', 'J01X'],
    'Rifamycin': ['Rifamycin', 'J01X'],
}

ANTIBIOTIC_CORE = {
    'Tetracyclines': ['CN(C1CC(C(N)=O)=CC(C1CC2C3)CC2=CC4=C3C=CC=C4)C',
                      'O=C(N)C1=CC(N(C)C)C2CC3CC(C=CC=C4)=C4CC3=CC2C1',
                      'O=C(C)C1=CC(C2)C(C(N)C1)CC3=C2C=C4C=CC=CC4=C3'],
    'Amphenicols': ['CC(NC=O)C(O)C1=CC=CC=C1'],
    'Penicillins': ['O=CC1N2C(CC2SC1)=O', 'NC1CN2C(C=O)CSC12', 'O=CC(CCC1C2)N1C2=O'],
    'Carbapenems': ['OC(C1=CSC2CC(N21)=O)=O', 'O=C1N2C(C(O)=O)=CCC2C1'],
    'Cephalosporins': ['OC(C1=CCSC(C2)N1C2=O)=O', 'OC(C1=CCCC2CC(N21)=O)=O', 'O=C1CC2N1C(C(O)=O)=CCO2'],
    'DHFR_Inhibitor': ['NC1=NC(N)=NC=C1'],
    'Sulfonamides': ['NC1=CC=C(C=C1)S(N)(=O)=O', 'NC1=CC=C(C=C1)S(=O)=O'],
    'Macrolides': ['O=C1CCCCCCCCCCCCO1', 'O=C1CCCCCCCCC=CCCCCO1', 'O=C1CCCCCCCCC=CC=CCCO1', 'O=C1C=CCCCCCCC=CCCCCO1',
                   'O=C1CCC=CC=CCCC=CC=CCCCO1', 'O=C1CCCCCCCCCCCCCO1', 'O=C1OCCCCNCCCCCCCC1'],
    'Lincosamides': ['O=C(C1NCCC1)NCC2OCCC(C2O)O', 'O=C(NCC1C(O)C(O)CCO1)C2NCCCC2'],
    'Aminoglycoside': ['NC1C(C(C(C(C1)N)O)O)O'],
    'Quinolines': ['O=C1C2=CC=CNC2NC=C1C(O)=O', 'O=C1C2=CC=CC=C2NC=C1C(O)=O', 'C12=CC=CC=C1N=CC=C2',
                   'C12=C(NCCC2)C=CC=C1', 'O=C1C2=CC=CN=C2NC=C1C(O)=O', 'C12=CC=CN=C1NC=CC2'],
    'Pleuromutilin': ['CC1CCC23CCC(C2C1(C)C(OC(C)=O)CC(C)(C=C)C(O)C3C)=O',
                      'CC1CCC23CCC(C2C1(C)C(OC(C)=O)CC(C)(CC)C(O)C3C)=O',
                      'CC1(C2=CC(CC32CCC1C)=O)C(CC(C)(C(C3C)O)C=C)O'],
    'Oxazolidinone': ['O=C1NCCO1'],
    'Nitroimidazole': ['O=N(C1=CN=CN1)=O', 'O=[N+]([O-])C1=CN=CN1'],
    'Nitrofuran': ['O=N(C1=CC=CO1)=O', 'O=[N+]([O-])C1=CC=CO1', 'CC1=CC=CO1', 'O=C1CC=CO1', 'O[C@H]1CCOC1',
                   'O=C1C=CCO1'],
    'Rifamycin': ['CC1C=CC=C(C)C(NC2=CC(O)=C(C3=C2O)C4=C(OC(C4=O)(OC=CC(OC)C(C)C(OC(C)=O)C(C)C(O)C(C)C1O)C)C(C)=C3O)=O',
                  'CC1C=CC=C(C)C(NC2=CC=C(C3=C2O)C4=C(OC(C4=O)(OC=CC(OC)C(C)C(OC(C)=O)C(C)C(O)C(C)C1O)C)C(C)=C3O)=O'],
}
BASIC_FEATURES = ['Name', 'smiles', 'Added', 'Availability', 'Since', 'Mwt', 'logP', 'Mol Formula', 'Rings',
                  'Heavy Atoms', 'Hetero Atoms', 'Fraction sp3', 'Tranche', 'pH range', 'Net charge', 'H-bond donors',
                  'H-bond acceptors', 'tPSA', 'Rotatable bonds', 'Apolardesolvationkcal/mol',
                  'Polardesolvationkcal/mol',
                  'Anti']
EXTEND_FEATURES = ['MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'qed', 'MolWt',
                   'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'NumRadicalElectrons', 'MaxPartialCharge',
                   'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FpDensityMorgan1',
                   'FpDensityMorgan2', 'FpDensityMorgan3', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI',
                   'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'AvgIpc',
                   'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n',
                   'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA',
                   'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2',
                   'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9',
                   'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7',
                   'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2',
                   'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9',
                   'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4',
                   'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1',
                   'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6',
                   'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount',
                   'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings',
                   'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors',
                   'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles',
                   'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO',
                   'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO',
                   'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2',
                   'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde',
                   'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline',
                   'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
                   'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan',
                   'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan',
                   'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy',
                   'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso',
                   'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond',
                   'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd',
                   'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene',
                   'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea'
                   ]
SHAP_FEATURES = {
    'NP Characters': ['Intensity Mean', 'Number Mean', 'ZP', 'Au rate', 'Ligand rate'],

    'Physical Property': ['HeavyAtomMolWt', 'ExactMolWt', 'MolWt', 'TPSA', 'MolLogP', 'MolMR', 'NumValenceElectrons',
                          'NumHAcceptors', 'NumHDonors', 'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge',
                          'MinAbsPartialCharge', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3'],

    'Structure Property': ['RingCount', 'HeavyAtomCount', 'FractionCSP3', 'NHOHCount', 'NOCount',
                           'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings',
                           'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings',
                           'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumHeteroatoms',
                           'NumRotatableBonds', 'NumRadicalElectrons'],

    'Topological Property': ['Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v',
                             'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'BalabanJ',
                             'BertzCT', 'AvgIpc'],

    'Surface Area': ['LabuteASA', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6',
                     'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12',
                     'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8',
                     'SlogP_VSA9', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3',
                     'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9',
                     'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2',
                     'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9'],

    'E-State': ['MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'VSA_EState1',
                'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8',
                'VSA_EState9', 'VSA_EState10'],

    'Fragments': ['fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH',
                  'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1',
                  'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde',
                  'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline',
                  'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
                  'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan',
                  'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan',
                  'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy',
                  'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso',
                  'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond',
                  'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd',
                  'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene',
                  'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea',
                  ],

    'Drug Descriptors': ['BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI',
                         'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'qed']
}
