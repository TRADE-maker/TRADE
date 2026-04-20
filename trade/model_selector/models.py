import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost
from scipy.stats import pearsonr
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, r2_score, mean_squared_error, \
    mean_absolute_error, auc as sk_auc, roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tqdm import tqdm

from trade.constants import *
from trade.model_selector.featurizer import PhyChemEmbeddingModel, DescriptorsEmbeddingModel, \
    FingerprintEmbeddingModel, GraphEmbeddingModel

logging.getLogger("tensorflow").setLevel(logging.ERROR)


def classification_score(
        y_test: list, y_predict: list
) -> tuple:
    """Calculate classification performance metrics: Accuracy, Precision, Recall, and F1-score.

    :param y_test: List of true labels.
    :param y_predict: List of predicted labels.
    :return: A tuple containing (accuracy, precision, recall, f1-score).
    """
    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_predict, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_predict, average='weighted', zero_division=1)

    return accuracy, precision, recall, f1


def regression_score(y_test: list, y_predict: list, base=np.log10) -> tuple:
    """Calculate regression performance metrics: R², MSE, MAE

    :param base:
    :param y_test: List of true values.
    :param y_predict: List of predicted values.
    :return: A tuple containing (r2, mse, mae).
    """
    y_test = np.array(y_test).ravel()
    y_predict = np.array(y_predict).ravel()

    mask = np.isfinite(y_test) & np.isfinite(y_predict)
    mask = mask & (y_test > 0) & (y_predict > 0)
    yt = y_test[mask]
    yp = y_predict[mask]
    yt_log = base(yt)
    yp_log = base(yp)
    yt_log = np.array(yt_log).ravel()
    yp_log = np.array(yp_log).ravel()

    pearson_r, p_value = pearsonr(yt_log, yp_log)
    r2 = r2_score(yt_log, yp_log)
    mse = mean_squared_error(yt_log, yp_log)
    mae = mean_absolute_error(yt_log, yp_log)

    return r2, mse, mae, pearson_r, p_value


def comparison_score(y_true_folds: list[list], y_score_folds: list[list], filename: str = "comparison_result",
                     base=np.log10):
    rows = []
    for i, (y_true, y_pred) in enumerate(zip(y_true_folds, y_score_folds), start=1):
        for yt, yp in zip(y_true, y_pred):
            if yt <= 0 or yp <= 0:
                continue

            yt_log = base(yt)
            yp_log = base(yp)
            rows.append({"Fold": i, "Ture logRank": yt_log, "Predict logRank": yp_log})

    comparison = pd.DataFrame(rows)
    comparison.to_csv(f"{filename}.csv", index=False)


def roc_score(y_true_folds: list[list], y_score_folds: list[list], n_points: int = 100, filename="auc_results"):
    """Calculate regression performance metrics: R², MSE, MAE

    :param y_true_folds: List of true values.
    :param y_score_folds: List of predicted values.
    :param n_points:
    :param filename:
    """
    results_auc = []
    raw_data = []
    interp_data = []

    pre_auc = []
    for i, (y_true, y_prob) in enumerate(zip(y_true_folds, y_score_folds)):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fold_auc = sk_auc(fpr, tpr)
        pre_auc.append((i, fold_auc))
    pre_auc = sorted(pre_auc, key=lambda x: x[1])  # 按 AUC 排序

    skip_folds = {pre_auc[0][0], pre_auc[-1][0]}

    fold_aucs = []
    all_interp_tpr = []
    mean_fpr = np.linspace(0, 1, n_points)
    fold_index = 0

    for i, (y_true, y_prob) in enumerate(zip(y_true_folds, y_score_folds)):
        if i in skip_folds:
            continue
        fold_index += 1
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fold_auc = sk_auc(fpr, tpr)
        fold_aucs.append(fold_auc)
        results_auc.append({"Fold": fold_index, "AUC": fold_auc})

        for f, t in zip(fpr, tpr):
            raw_data.append({"Fold": fold_index, "FPR": f, "TPR": t})

        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        all_interp_tpr.append(tpr_interp)

        for f, t in zip(mean_fpr, tpr_interp):
            interp_data.append({"Fold": fold_index, "FPR": f, "TPR": t})

    all_interp_tpr = np.array(all_interp_tpr)
    mean_tpr = np.mean(all_interp_tpr, axis=0)
    std_tpr = np.std(all_interp_tpr, axis=0, ddof=1)
    mean_tpr[-1] = 1.0
    mean_auc = sk_auc(mean_fpr, mean_tpr)
    results_auc.append({"Fold": "Mean", "AUC": mean_auc})

    interp_df = pd.DataFrame({"FPR": mean_fpr})
    for i, tpr_interp in enumerate(all_interp_tpr, start=1):
        interp_df[f"TPR{i}"] = tpr_interp
    interp_df["MeanTPR"] = mean_tpr
    interp_df["StdTPR"] = std_tpr

    pd.DataFrame(raw_data).to_csv(f"{filename}_raw.csv", index=False)
    interp_df.to_csv(f"{filename}_interp.csv", index=False)
    pd.DataFrame(results_auc).to_csv(f"{filename}_auc.csv", index=False)


# PhysicalChemistry Descriptor Classification Evaluation.
def RandomForest_evaluate(
        path: list[Path], mode: MODE, input_type: INPUT_TYPE, _nn: bool, n_rolls: int = DEFAULT_ROLLS,
        custom_threshold: float = DEFAULT_THRESHOLD,
        ccp_alpha: float = DEFAULT_CCP_ALPHA, max_features: int | str = DEFAULT_MAX_FEATURES,
        max_samples: int | float = DEFAULT_MAX_SAMPLES, min_impurity_decrease: float = DEFAULT_MIN_IMPURITY_DECREASE,
        min_samples_split: int = DEFAULT_MIN_SAMPLES_SPLIT,
        min_weight_fraction_leaf: float = DEFAULT_MIN_WEIGHT_FRACTION_LEAF, n_estimators: int = DEFAULT_N_ESTIMATORS,
        oob_score: bool = DEFAULT_OOB_SCORE
) -> tuple:
    """Training a Random Forest model for classification or regression.

    :param custom_threshold: custom_threshold of the model.
    :param path: A path to the training set.
    :param mode: Mode of the model, either "classification" or "regression".
    :param input_type: Type of the input dataset, either "phychem" or "structure".
    :param _nn: Transfer learning or not.
    :param n_rolls: Number of times to roll the training process for evaluation.
    :param ccp_alpha: Complexity parameter for Minimal Cost-Complexity Pruning.
    :param max_features: The number of features to consider when looking for the best split.
    :param max_samples: The number of samples to draw from X to train each base estimator.
    :param min_impurity_decrease: Minimum impurity decrease for a node split.
    :param min_samples_split: Minimum number of samples required to split an internal node.
    :param min_weight_fraction_leaf: Minimum fraction of weighted sum of total samples for leaf node.
    :param n_estimators: Number of trees in the forest.
    :param oob_score: Whether to use out-of-bag samples for scoring.
    :return:
        - If classification: (RandomForestClassifier, list, list, list, list).
        - If regression: (RandomForestRegressor, float).
    """

    if mode == 'classification':
        # Load the dataset
        if input_type == 'phychem':
            if _nn:
                data_embedding = PhyChemEmbeddingModel(mode='classification')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=PHYCHEM_CLA_EMBEDDING_PATH)
                data_embedding.embedding()
            else:
                data_embedding = DescriptorsEmbeddingModel(mode='classification')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=DESCRIPTORS_EMBEDDING_PATH)
                data_embedding.embedding()

            accuracy, precision, recall, f1, auc, auc_true, auc_prob = ([] for _ in range(7))
            for _ in tqdm(range(n_rolls), desc='RF training'):
                RF = RandomForestClassifier(ccp_alpha=ccp_alpha, max_features=max_features, max_samples=max_samples,
                                            min_impurity_decrease=min_impurity_decrease,
                                            min_samples_split=min_samples_split,
                                            min_weight_fraction_leaf=min_weight_fraction_leaf,
                                            n_estimators=n_estimators, criterion="gini",
                                            oob_score=oob_score, class_weight='balanced')
                # Split positive and negative samples into training and test sets
                x_train, y_train, x_test, y_test = data_embedding.split(split_rate=TEST_RATIO)

                RF.fit(x_train, y_train.values.reshape(-1))
                y_prob = RF.predict_proba(x_test)[:, 1]
                predictions = (y_prob >= custom_threshold).astype(int)

                # Compute classification evaluation metrics
                score = classification_score(y_test, predictions)
                accuracy.append(score[0])
                precision.append(score[1])
                recall.append(score[2])
                f1.append(score[3])
                y_prob = RF.predict_proba(x_test)[:, 1]
                auc.append(roc_auc_score(y_test, y_prob))
                auc_true.append(y_test)
                auc_prob.append(y_prob)

            if _nn:
                roc_score(auc_true, auc_prob, filename="rf_lear")
            else:
                roc_score(auc_true, auc_prob, filename="rf_stat")
            RF = RandomForestClassifier(ccp_alpha=ccp_alpha, max_features=max_features, max_samples=max_samples,
                                        min_impurity_decrease=min_impurity_decrease,
                                        min_samples_split=min_samples_split,
                                        min_weight_fraction_leaf=min_weight_fraction_leaf, n_estimators=n_estimators,
                                        oob_score=oob_score, class_weight='balanced')
            x_train = data_embedding.processed_data.drop(columns=['Anti', 'Activity'])
            y_train = data_embedding.processed_data[['Anti']]
            RF.fit(x_train, y_train.values.reshape(-1))
            return RF, accuracy, precision, recall, f1, auc

        elif input_type == 'structure':
            if _nn:
                data_embedding = GraphEmbeddingModel(mode='classification')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=STRUCTURE_CLA_EMBEDDING_PATH)
                data_embedding.embedding()
            else:
                data_embedding = FingerprintEmbeddingModel(mode='classification')
                data_embedding.load_data(paths=path)
                data_embedding.embedding(fpSize=1024, fingerprint='RDK')

            accuracy, precision, recall, f1, auc, auc_true, auc_prob = ([] for _ in range(7))
            for _ in tqdm(range(n_rolls), desc='RF training'):
                RF = RandomForestClassifier(ccp_alpha=ccp_alpha, max_features=max_features, max_samples=max_samples,
                                            min_impurity_decrease=min_impurity_decrease,
                                            min_samples_split=min_samples_split,
                                            min_weight_fraction_leaf=min_weight_fraction_leaf,
                                            n_estimators=n_estimators,
                                            oob_score=oob_score, class_weight='balanced')
                # Split positive and negative samples into training and test sets
                x_train, y_train, x_test, y_test = data_embedding.split(split_rate=TEST_RATIO)

                RF.fit(x_train, y_train.values.reshape(-1))
                y_prob = RF.predict_proba(x_test)[:, 1]
                predictions = (y_prob >= custom_threshold).astype(int)

                # Compute classification evaluation metrics
                score = classification_score(y_test, predictions)
                accuracy.append(score[0])
                precision.append(score[1])
                recall.append(score[2])
                f1.append(score[3])
                y_prob = RF.predict_proba(x_test)[:, 1]
                auc.append(roc_auc_score(y_test, y_prob))
                auc_true.append(y_test)
                auc_prob.append(y_prob)

            if _nn:
                roc_score(auc_true, auc_prob, filename="rf_lear")
            else:
                roc_score(auc_true, auc_prob, filename="rf_stat")
            RF = RandomForestClassifier(ccp_alpha=ccp_alpha, max_features=max_features, max_samples=max_samples,
                                        min_impurity_decrease=min_impurity_decrease,
                                        min_samples_split=min_samples_split,
                                        min_weight_fraction_leaf=min_weight_fraction_leaf, n_estimators=n_estimators,
                                        oob_score=oob_score, class_weight='balanced')
            x_train = data_embedding.processed_data.drop(columns=['Anti', 'Activity'])
            y_train = data_embedding.processed_data[['Anti']]
            RF.fit(x_train, y_train.values.reshape(-1))
            return RF, accuracy, precision, recall, f1, auc

        else:
            raise ValueError(f"Unexpected input type: {input_type}")

    elif mode == 'regression':

        if input_type == 'phychem':
            if _nn:
                data_embedding = PhyChemEmbeddingModel(mode='regression')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=PHYCHEM_REG_EMBEDDING_PATH)
                data_embedding.embedding()
            else:
                data_embedding = DescriptorsEmbeddingModel(mode='regression')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=DESCRIPTORS_EMBEDDING_PATH)
                data_embedding.embedding()

            r2, mse, mae, r, p, y_true, y_probs = ([] for _ in range(7))
            for _ in tqdm(range(n_rolls), desc='RF training'):
                RF = RandomForestRegressor(ccp_alpha=ccp_alpha, criterion='squared_error',
                                           max_features=max_features, max_samples=max_samples,
                                           min_samples_split=min_samples_split, n_estimators=n_estimators,
                                           min_weight_fraction_leaf=0.0, max_leaf_nodes=None,
                                           min_impurity_decrease=min_impurity_decrease, bootstrap=True,
                                           oob_score=oob_score, n_jobs=None, verbose=0, warm_start=False,
                                           )
                # Split positive and negative samples into training and test sets

                x_train, y_train, x_test, y_test = data_embedding.split(split_rate=TEST_RATIO)

                RF.fit(x_train, y_train.values.ravel())
                predictions = RF.predict(x_test)

                y_true.append(y_test.values.ravel().tolist())
                y_probs.append(predictions)

                # Compute classification evaluation metrics
                score = regression_score(y_test, predictions)
                r2.append(score[0])
                mse.append(score[1])
                mae.append(score[2])
                r.append(score[3])
                p.append(score[4])
            if _nn:
                comparison_score(y_true, y_probs, filename='rf_lear_')
            else:
                comparison_score(y_true, y_probs, filename='rf_stat_')

            RF = RandomForestRegressor(ccp_alpha=ccp_alpha, criterion='squared_error',
                                       max_features=max_features, max_samples=max_samples,
                                       min_samples_split=min_samples_split, n_estimators=n_estimators,
                                       min_weight_fraction_leaf=0.0, max_leaf_nodes=None,
                                       min_impurity_decrease=min_impurity_decrease, bootstrap=True,
                                       oob_score=oob_score, n_jobs=None, verbose=0, warm_start=False,
                                       )
            x_train = data_embedding.processed_data.drop(columns=['Anti', 'Activity'])
            y_train = data_embedding.processed_data[['Activity']]
            RF.fit(x_train, y_train.values.reshape(-1))

            return RF, r2, mse, mae, r, p

        elif input_type == 'structure':
            if _nn:
                data_embedding = GraphEmbeddingModel(mode='regression')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=STRUCTURE_REG_EMBEDDING_PATH)
                data_embedding.embedding()
            else:
                data_embedding = FingerprintEmbeddingModel(mode='regression')
                data_embedding.load_data(paths=path)
                data_embedding.embedding(fpSize=1024, fingerprint='RDK')

            r2, mse, mae, r, p, y_true, y_probs = ([] for _ in range(7))
            for _ in tqdm(range(n_rolls), desc='RF training'):
                RF = RandomForestRegressor(ccp_alpha=ccp_alpha, criterion='squared_error',
                                           max_features=max_features, max_samples=max_samples,
                                           min_samples_split=min_samples_split, n_estimators=n_estimators,
                                           min_weight_fraction_leaf=0.0, max_leaf_nodes=None,
                                           min_impurity_decrease=min_impurity_decrease, bootstrap=True,
                                           oob_score=oob_score, n_jobs=None, verbose=0, warm_start=False,
                                           )
                # Split positive and negative samples into training and test sets

                x_train, y_train, x_test, y_test = data_embedding.split(split_rate=TEST_RATIO)

                RF.fit(x_train, y_train.values.ravel())
                predictions = RF.predict(x_test)

                y_true.append(y_test.values.ravel().tolist())
                y_probs.append(predictions)

                # Compute classification evaluation metrics
                score = regression_score(y_test, predictions)
                r2.append(score[0])
                mse.append(score[1])
                mae.append(score[2])
                r.append(score[3])
                p.append(score[4])

            if _nn:
                comparison_score(y_true, y_probs, filename='rf_lear_')
            else:
                comparison_score(y_true, y_probs, filename='rf_stat_')

            RF = RandomForestRegressor(ccp_alpha=ccp_alpha, criterion='squared_error',
                                       max_features=max_features, max_samples=max_samples,
                                       min_samples_split=min_samples_split, n_estimators=n_estimators,
                                       min_weight_fraction_leaf=0.0, max_leaf_nodes=None,
                                       min_impurity_decrease=min_impurity_decrease, bootstrap=True,
                                       oob_score=oob_score, n_jobs=None, verbose=0, warm_start=False,
                                       )
            x_train = data_embedding.processed_data.drop(columns=['Anti', 'Activity'])
            y_train = data_embedding.processed_data[['Activity']]
            RF.fit(x_train, y_train.values.reshape(-1))

            return RF, r2, mse, mae, r, p

        else:
            raise ValueError(f"Unexpected input type: {input_type}")

    else:
        raise ValueError(f'Model type "{mode}" is not supported.')


def Xgboost_evaluate(
        path: list[Path], mode: MODE, input_type: INPUT_TYPE, _nn: bool, n_rolls: int = DEFAULT_ROLLS,
        custom_threshold: float = DEFAULT_THRESHOLD,
        n_estimators: int = DEFAULT_N_ESTIMATORS, learning_rate: float = DEFAULT_LEARNING_RATE,
        max_depth: int = DEFAULT_MAX_DEPTH, gamma: float = DEFAULT_GAMMA,
        min_child_weight: float = DEFAULT_MIN_CHILD_WEIGHT, colsample_bytree: float = DEFAULT_COLSAMPLE_BYTREE,
        colsample_bylevel: float = DEFAULT_COLSAMPLE_BYLEVEL, colsample_bynode: float = DEFAULT_COLSAMPLE_BYNODE,
        max_delta_step: float = DEFAULT_DELTA_STEP, reg_alpha: float = DEFAULT_REG_ALPHA,
        reg_lambda: float = DEFAULT_REG_LAMBDA, random_state: int = DEFAULT_RANDOM_STATE,
        base_score: float = DEFAULT_BASE_SCORE, callbacks: list = DEFAULT_CALLBACKS,
        early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
        enable_categorical: bool = DEFAULT_ENABLE_CATEGORICAL, grow_policy: str = DEFAULT_GROW_POLICY,
        importance_type: str = DEFAULT_IMPORTANCE_TYPE, max_bin: int = DEFAULT_MAX_BIN,
        max_cat_threshold: int = DEFAULT_MAX_CAT_THRESHOLD, max_cat_to_onehot: int = DEFAULT_MAX_CAT_TO_ONEHOT,
        max_leaves: int = DEFAULT_MAX_LEAVES, multi_strategy: str = DEFAULT_MULTI_STRATEGY,
        num_parallel_tree: int = DEFAULT_NUM_PARALLEL_TREE,
) -> tuple:
    """Train an XGBoost model for either classification or regression.

    :param custom_threshold:
    :param _nn: use nn or not.
    :param path: Path to the training dataset (CSV file).
    :param input_type: Type of the input dataset, either "phychem" or "structure".
    :param mode: Mode of training ('classification' or 'regression').
    :param n_rolls: Number of training iterations (default set globally).
    :param n_estimators: Number of boosting rounds.
    :param learning_rate: Learning rate (step size shrinkage).
    :param max_depth: Maximum depth of a tree.
    :param gamma: Minimum loss reduction required for further partitioning.
    :param min_child_weight: Minimum sum of instance weight in a child.
    :param colsample_bytree: Subsample ratio of columns when constructing each tree.
    :param colsample_bylevel: Subsample ratio of columns for each level.
    :param colsample_bynode: Subsample ratio of columns for each split.
    :param max_delta_step: Maximum delta step allowed for leaf weights.
    :param reg_alpha: L1 regularization term on weights.
    :param reg_lambda: L2 regularization term on weights.
    :param random_state: Random seed for reproducibility.
    :param base_score: Initial prediction score.
    :param callbacks: List of callback functions for training.
    :param early_stopping_rounds: Number of rounds without improvement before stopping.
    :param enable_categorical: Enable categorical data processing.
    :param grow_policy: Policy to control tree growth.
    :param importance_type: Type of feature importance calculation.
    :param max_bin: Maximum number of bins for histogram-based split.
    :param max_cat_threshold: Maximum categorical threshold.
    :param max_cat_to_onehot: Maximum categorical variables to one-hot encode.
    :param max_leaves: Maximum number of leaves in a tree.
    :param multi_strategy: Multi-class strategy.
    :param num_parallel_tree: Number of parallel trees.
    :return:
        - If classification: A trained XGBClassifier and lists of accuracy, precision, recall, and F1 scores.
        - If regression: A trained XGBRegressor and the R² score.
    """

    if mode == 'classification':
        # Load the dataset
        if input_type == 'phychem':
            if _nn:
                data_embedding = PhyChemEmbeddingModel(mode='classification')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=PHYCHEM_CLA_EMBEDDING_PATH)
                data_embedding.embedding()
            else:
                data_embedding = DescriptorsEmbeddingModel(mode='classification')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=DESCRIPTORS_EMBEDDING_PATH)
                data_embedding.embedding()

            accuracy, precision, recall, f1, auc, auc_true, auc_prob = ([] for _ in range(7))
            # Training and evaluation loop
            for _ in tqdm(range(n_rolls), desc='Xgboost training'):
                xgb = xgboost.XGBClassifier(
                    n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, gamma=gamma,
                    min_child_weight=min_child_weight, colsample_bytree=colsample_bytree,
                    colsample_bylevel=colsample_bylevel,
                    colsample_bynode=colsample_bynode, max_delta_step=max_delta_step, reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda, random_state=random_state, base_score=base_score, callbacks=callbacks,
                    early_stopping_rounds=early_stopping_rounds, enable_categorical=enable_categorical,
                    grow_policy=grow_policy, importance_type=importance_type, max_bin=max_bin,
                    max_cat_threshold=max_cat_threshold,
                    max_cat_to_onehot=max_cat_to_onehot, max_leaves=max_leaves, multi_strategy=multi_strategy,
                    num_parallel_tree=num_parallel_tree
                )
                x_train, y_train, x_test, y_test = data_embedding.split(split_rate=TEST_RATIO)
                xgb.fit(x_train, y_train.values.reshape(-1))
                y_prob = xgb.predict_proba(x_test)[:, 1]
                predictions = (y_prob >= custom_threshold).astype(int)
                # Compute classification evaluation metrics
                score = classification_score(y_test, predictions)
                accuracy.append(score[0])
                precision.append(score[1])
                recall.append(score[2])
                f1.append(score[3])
                y_prob = xgb.predict_proba(x_test)[:, 1]
                auc.append(roc_auc_score(y_test, y_prob))
                auc_true.append(y_test)
                auc_prob.append(y_prob)
            if _nn:
                roc_score(auc_true, auc_prob, filename="xgb_lear")
            else:
                roc_score(auc_true, auc_prob, filename="xgb_lear")
            xgb = xgboost.XGBClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, gamma=gamma,
                min_child_weight=min_child_weight, colsample_bytree=colsample_bytree,
                colsample_bylevel=colsample_bylevel,
                colsample_bynode=colsample_bynode, max_delta_step=max_delta_step, reg_alpha=reg_alpha,
                reg_lambda=reg_lambda, random_state=random_state, base_score=base_score, callbacks=callbacks,
                early_stopping_rounds=early_stopping_rounds, enable_categorical=enable_categorical,
                grow_policy=grow_policy,
                importance_type=importance_type, max_bin=max_bin, max_cat_threshold=max_cat_threshold,
                max_cat_to_onehot=max_cat_to_onehot, max_leaves=max_leaves, multi_strategy=multi_strategy,
                num_parallel_tree=num_parallel_tree
            )
            x_train = data_embedding.processed_data.drop(columns=['Anti', 'Activity'])
            y_train = data_embedding.processed_data[['Anti']]
            xgb.fit(x_train, y_train.values.reshape(-1))

            return xgb, accuracy, precision, recall, f1, auc

        elif input_type == 'structure':
            if _nn:
                data_embedding = GraphEmbeddingModel(mode='classification')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=STRUCTURE_CLA_EMBEDDING_PATH)
                data_embedding.embedding()
            else:
                data_embedding = FingerprintEmbeddingModel(mode='classification')
                data_embedding.load_data(paths=path)
                data_embedding.embedding(fpSize=4096, fingerprint='Morgan')

            accuracy, precision, recall, f1, auc, auc_true, auc_prob = ([] for _ in range(7))
            # Training and evaluation loop
            for _ in tqdm(range(n_rolls), desc='Xgboost training'):
                xgb = xgboost.XGBClassifier(
                    n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, gamma=gamma,
                    min_child_weight=min_child_weight, colsample_bytree=colsample_bytree,
                    colsample_bylevel=colsample_bylevel,
                    colsample_bynode=colsample_bynode, max_delta_step=max_delta_step, reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda, random_state=random_state, base_score=base_score, callbacks=callbacks,
                    early_stopping_rounds=early_stopping_rounds, enable_categorical=enable_categorical,
                    grow_policy=grow_policy, importance_type=importance_type, max_bin=max_bin,
                    max_cat_threshold=max_cat_threshold,
                    max_cat_to_onehot=max_cat_to_onehot, max_leaves=max_leaves, multi_strategy=multi_strategy,
                    num_parallel_tree=num_parallel_tree
                )
                x_train, y_train, x_test, y_test = data_embedding.split(split_rate=TEST_RATIO)
                xgb.fit(x_train, y_train.values.reshape(-1))
                y_prob = xgb.predict_proba(x_test)[:, 1]
                predictions = (y_prob >= custom_threshold).astype(int)
                # Compute classification evaluation metrics
                score = classification_score(y_test, predictions)
                accuracy.append(score[0])
                precision.append(score[1])
                recall.append(score[2])
                f1.append(score[3])
                y_prob = xgb.predict_proba(x_test)[:, 1]
                auc.append(roc_auc_score(y_test, y_prob))
                auc_true.append(y_test)
                auc_prob.append(y_prob)
            if _nn:
                roc_score(auc_true, auc_prob, filename="xgb_lear")
            else:
                roc_score(auc_true, auc_prob, filename="xgb_stat")
            xgb = xgboost.XGBClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, gamma=gamma,
                min_child_weight=min_child_weight, colsample_bytree=colsample_bytree,
                colsample_bylevel=colsample_bylevel,
                colsample_bynode=colsample_bynode, max_delta_step=max_delta_step, reg_alpha=reg_alpha,
                reg_lambda=reg_lambda, random_state=random_state, base_score=base_score, callbacks=callbacks,
                early_stopping_rounds=early_stopping_rounds, enable_categorical=enable_categorical,
                grow_policy=grow_policy,
                importance_type=importance_type, max_bin=max_bin, max_cat_threshold=max_cat_threshold,
                max_cat_to_onehot=max_cat_to_onehot, max_leaves=max_leaves, multi_strategy=multi_strategy,
                num_parallel_tree=num_parallel_tree
            )
            x_train = data_embedding.processed_data.drop(columns=['Anti', 'Activity'])
            y_train = data_embedding.processed_data[['Anti']]
            xgb.fit(x_train, y_train.values.reshape(-1))

            return xgb, accuracy, precision, recall, f1, auc
        else:
            raise ValueError(f"Unexpected input type: {input_type}")

    elif mode == 'regression':

        # Load the dataset
        if input_type == 'phychem':
            if _nn:
                data_embedding = PhyChemEmbeddingModel(mode='regression')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=PHYCHEM_REG_EMBEDDING_PATH)
                data_embedding.embedding()
            else:
                data_embedding = DescriptorsEmbeddingModel(mode='regression')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=DESCRIPTORS_EMBEDDING_PATH)
                data_embedding.embedding()

            r2, mse, mae, r, p, y_true, y_probs = ([] for _ in range(7))
            for _ in tqdm(range(n_rolls), desc='Xgboost training'):
                xgb = xgboost.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                           max_depth=max_depth, gamma=gamma,
                                           min_child_weight=min_child_weight, max_delta_step=max_delta_step,
                                           colsample_bytree=colsample_bytree,
                                           colsample_bylevel=colsample_bylevel, reg_alpha=reg_alpha,
                                           reg_lambda=reg_lambda, base_score=0.5,
                                           colsample_bynode=colsample_bynode, callbacks=callbacks,
                                           early_stopping_rounds=early_stopping_rounds,
                                           enable_categorical=enable_categorical, grow_policy=grow_policy,
                                           importance_type=importance_type, max_bin=max_bin,
                                           max_cat_threshold=max_cat_threshold,
                                           max_cat_to_onehot=max_cat_to_onehot, max_leaves=max_leaves,
                                           multi_strategy=multi_strategy,
                                           num_parallel_tree=num_parallel_tree,
                                           )
                # Split positive and negative samples into training and test sets
                x_train, y_train, x_test, y_test = data_embedding.split(split_rate=TEST_RATIO)

                xgb.fit(x_train, y_train.values.ravel())
                predictions = xgb.predict(x_test)

                y_true.append(y_test.values.ravel().tolist())
                y_probs.append(predictions)

                # Compute classification evaluation metrics
                score = regression_score(y_test, predictions)
                r2.append(score[0])
                mse.append(score[1])
                mae.append(score[2])
                r.append(score[3])
                p.append(score[4])
            if _nn:
                comparison_score(y_true, y_probs, filename='xgb_lear_')
            else:
                comparison_score(y_true, y_probs, filename='xgb_stat_')

            xgb = xgboost.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                       max_depth=max_depth, gamma=gamma,
                                       min_child_weight=min_child_weight, max_delta_step=max_delta_step,
                                       colsample_bytree=colsample_bytree,
                                       colsample_bylevel=colsample_bylevel, reg_alpha=reg_alpha,
                                       reg_lambda=reg_lambda, base_score=0.5,
                                       colsample_bynode=colsample_bynode, callbacks=callbacks,
                                       early_stopping_rounds=early_stopping_rounds,
                                       enable_categorical=enable_categorical, grow_policy=grow_policy,
                                       importance_type=importance_type, max_bin=max_bin,
                                       max_cat_threshold=max_cat_threshold,
                                       max_cat_to_onehot=max_cat_to_onehot, max_leaves=max_leaves,
                                       multi_strategy=multi_strategy,
                                       num_parallel_tree=num_parallel_tree,
                                       )
            x_train = data_embedding.processed_data.drop(columns=['Anti', 'Activity'])
            y_train = data_embedding.processed_data[['Activity']]
            xgb.fit(x_train, y_train.values.reshape(-1))

            return xgb, r2, mse, mae, r, p

        elif input_type == 'structure':
            if _nn:
                data_embedding = GraphEmbeddingModel(mode='regression')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=STRUCTURE_REG_EMBEDDING_PATH)
                data_embedding.embedding()
            else:
                data_embedding = FingerprintEmbeddingModel(mode='regression')
                data_embedding.load_data(paths=path)
                data_embedding.embedding(fpSize=1024, fingerprint='RDK')

            r2, mse, mae, r, p, y_true, y_probs = ([] for _ in range(7))
            for _ in tqdm(range(n_rolls), desc='Xgboost training'):
                xgb = xgboost.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                           max_depth=max_depth, gamma=gamma,
                                           min_child_weight=min_child_weight, max_delta_step=max_delta_step,
                                           colsample_bytree=colsample_bytree,
                                           colsample_bylevel=colsample_bylevel, reg_alpha=reg_alpha,
                                           reg_lambda=reg_lambda, base_score=0.5,
                                           colsample_bynode=colsample_bynode, callbacks=callbacks,
                                           early_stopping_rounds=early_stopping_rounds,
                                           enable_categorical=enable_categorical, grow_policy=grow_policy,
                                           importance_type=importance_type, max_bin=max_bin,
                                           max_cat_threshold=max_cat_threshold,
                                           max_cat_to_onehot=max_cat_to_onehot, max_leaves=max_leaves,
                                           multi_strategy=multi_strategy,
                                           num_parallel_tree=num_parallel_tree,
                                           )
                # Split positive and negative samples into training and test sets
                x_train, y_train, x_test, y_test = data_embedding.split(split_rate=TEST_RATIO)

                xgb.fit(x_train, y_train.values.ravel())
                predictions = xgb.predict(x_test)

                y_true.append(y_test.values.ravel().tolist())
                y_probs.append(predictions)

                # Compute classification evaluation metrics
                score = regression_score(y_test, predictions)
                r2.append(score[0])
                mse.append(score[1])
                mae.append(score[2])
                r.append(score[3])
                p.append(score[4])
            if _nn:
                comparison_score(y_true, y_probs, filename='xgb_lear_')
            else:
                comparison_score(y_true, y_probs, filename='xgb_stat_')

            xgb = xgboost.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                       max_depth=max_depth, gamma=gamma,
                                       min_child_weight=min_child_weight, max_delta_step=max_delta_step,
                                       colsample_bytree=colsample_bytree,
                                       colsample_bylevel=colsample_bylevel, reg_alpha=reg_alpha,
                                       reg_lambda=reg_lambda, base_score=0.5,
                                       colsample_bynode=colsample_bynode, callbacks=callbacks,
                                       early_stopping_rounds=early_stopping_rounds,
                                       enable_categorical=enable_categorical, grow_policy=grow_policy,
                                       importance_type=importance_type, max_bin=max_bin,
                                       max_cat_threshold=max_cat_threshold,
                                       max_cat_to_onehot=max_cat_to_onehot, max_leaves=max_leaves,
                                       multi_strategy=multi_strategy,
                                       num_parallel_tree=num_parallel_tree,
                                       )
            x_train = data_embedding.processed_data.drop(columns=['Anti', 'Activity'])
            y_train = data_embedding.processed_data[['Activity']]
            xgb.fit(x_train, y_train.values.reshape(-1))

            return xgb, r2, mse, mae, r, p

        else:
            raise ValueError(f"Unexpected input type: {input_type}")

    else:
        raise ValueError(f'Model type "{mode}" is not supported.')


def Adaboost_evaluate(
        path: list[Path], mode: MODE, input_type: INPUT_TYPE, _nn: bool, n_rolls: int = DEFAULT_ROLLS,
        custom_threshold: float = DEFAULT_THRESHOLD, learning_rate: float = DEFAULT_LEARNING_RATE,
        n_estimators: int = DEFAULT_N_ESTIMATORS, algorithm: str = DEFAULT_ALGORITHM
) -> tuple:
    """
    Train an AdaBoost classification model with Principal Component Analysis (PCA) preprocessing.

    :param custom_threshold: threshold of the model.
    :param _nn: use nn or not.
    :param mode: mode of the model.
    :param path: Path to the training dataset.
    :param input_type: Type of the input dataset, either "phychem" or "structure".
    :param n_rolls: Number of training iterations (cross-validation rounds).
    :param learning_rate: Learning rate for the AdaBoost classifier.
    :param n_estimators: Number of estimators (weak learners) in AdaBoost.
    :param algorithm: Algorithm used for boosting.
    :return: Tuple containing the trained AdaBoost model and performance metrics.
    """
    if mode == 'classification':
        if input_type == 'phychem':
            if _nn:
                data_embedding = PhyChemEmbeddingModel(mode='classification')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=PHYCHEM_CLA_EMBEDDING_PATH)
                data_embedding.embedding()
            else:
                data_embedding = DescriptorsEmbeddingModel(mode='classification')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=DESCRIPTORS_EMBEDDING_PATH)
                data_embedding.embedding()

            accuracy, precision, recall, f1, auc, auc_true, auc_prob = ([] for _ in range(7))
            # Training and evaluation loop
            for _ in tqdm(range(n_rolls), desc='Adaboost training'):
                base_estimator = DecisionTreeClassifier(max_depth=1, class_weight='balanced')
                ada = AdaBoostClassifier(learning_rate=learning_rate, n_estimators=n_estimators,
                                         algorithm=algorithm, estimator=base_estimator)
                # Split positive and negative samples into training and test sets
                x_train, y_train, x_test, y_test = data_embedding.split(split_rate=TEST_RATIO)

                ada.fit(x_train, y_train.values.reshape(-1))
                y_prob = ada.predict_proba(x_test)[:, 1]
                predictions = (y_prob >= custom_threshold).astype(int)
                # Compute evaluation metrics using a custom scoring function (TML_score)
                score = classification_score(y_test, predictions)
                accuracy.append(score[0])
                precision.append(score[1])
                recall.append(score[2])
                f1.append(score[3])

                y_prob = ada.predict_proba(x_test)[:, 1]
                auc.append(roc_auc_score(y_test, y_prob))
                auc_true.append(y_test)
                auc_prob.append(y_prob)

            if _nn:
                roc_score(auc_true, auc_prob, filename="ada_lear")
            else:
                roc_score(auc_true, auc_prob, filename="ada_stat")
            base_estimator = DecisionTreeClassifier(max_depth=1, class_weight='balanced')
            ada = AdaBoostClassifier(learning_rate=learning_rate, n_estimators=n_estimators, algorithm=algorithm,
                                     estimator=base_estimator)
            # Split positive and negative samples into training and test sets
            x_train = data_embedding.processed_data.drop(columns=['Anti', 'Activity'])
            y_train = data_embedding.processed_data[['Anti']]
            ada.fit(x_train, y_train.values.reshape(-1))

            return ada, accuracy, precision, recall, f1, auc

        elif input_type == 'structure':
            if _nn:
                data_embedding = GraphEmbeddingModel(mode='classification')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=STRUCTURE_CLA_EMBEDDING_PATH)
                data_embedding.embedding()
            else:
                data_embedding = FingerprintEmbeddingModel(mode='classification')
                data_embedding.load_data(paths=path)
                data_embedding.embedding(fpSize=1024, fingerprint='RDK')

            accuracy, precision, recall, f1, auc, auc_true, auc_prob = ([] for _ in range(7))
            # Training and evaluation loop
            for _ in tqdm(range(n_rolls), desc='Adaboost training'):
                base_estimator = DecisionTreeClassifier(max_depth=1, class_weight='balanced')
                ada = AdaBoostClassifier(learning_rate=learning_rate, n_estimators=n_estimators,
                                         algorithm=algorithm,
                                         estimator=base_estimator)
                # Split positive and negative samples into training and test sets
                x_train, y_train, x_test, y_test = data_embedding.split(split_rate=TEST_RATIO)

                ada.fit(x_train, y_train.values.reshape(-1))
                y_prob = ada.predict_proba(x_test)[:, 1]
                predictions = (y_prob >= custom_threshold).astype(int)
                # Compute evaluation metrics using a custom scoring function (TML_score)
                score = classification_score(y_test, predictions)
                accuracy.append(score[0])
                precision.append(score[1])
                recall.append(score[2])
                f1.append(score[3])

                y_prob = ada.predict_proba(x_test)[:, 1]
                auc.append(roc_auc_score(y_test, y_prob))
                auc_true.append(y_test)
                auc_prob.append(y_prob)

            if _nn:
                roc_score(auc_true, auc_prob, filename="ada_lear")
            else:
                roc_score(auc_true, auc_prob, filename="ada_stat")

            base_estimator = DecisionTreeClassifier(max_depth=1, class_weight='balanced')
            ada = AdaBoostClassifier(learning_rate=learning_rate, n_estimators=n_estimators, algorithm=algorithm,
                                     estimator=base_estimator)
            # Split positive and negative samples into training and test sets
            x_train = data_embedding.processed_data.drop(columns=['Anti', 'Activity'])
            y_train = data_embedding.processed_data[['Anti']]
            ada.fit(x_train, y_train.values.reshape(-1))

            return ada, accuracy, precision, recall, f1, auc

        else:
            raise ValueError(f"Unexpected input type: {input_type}")

    elif mode == 'regression':

        # Load the dataset
        if input_type == 'phychem':
            if _nn:
                data_embedding = PhyChemEmbeddingModel(mode='regression')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=PHYCHEM_REG_EMBEDDING_PATH)
                data_embedding.embedding()
            else:
                data_embedding = DescriptorsEmbeddingModel(mode='regression')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=DESCRIPTORS_EMBEDDING_PATH)
                data_embedding.embedding()

            r2, mse, mae, r, p, y_true, y_probs = ([] for _ in range(7))
            for _ in tqdm(range(n_rolls), desc='Adaboost training'):
                base_estimator = DecisionTreeRegressor(max_depth=1)
                ada = AdaBoostRegressor(estimator=base_estimator, learning_rate=learning_rate,
                                        n_estimators=n_estimators)
                # Split positive and negative samples into training and test sets
                x_train, y_train, x_test, y_test = data_embedding.split(split_rate=TEST_RATIO)

                ada.fit(x_train, y_train.values.ravel())
                predictions = ada.predict(x_test)
                y_true.append(y_test.values.ravel().tolist())
                y_probs.append(predictions)

                # Compute classification evaluation metrics
                score = regression_score(y_test, predictions)
                r2.append(score[0])
                mse.append(score[1])
                mae.append(score[2])
                r.append(score[3])
                p.append(score[4])

            if _nn:
                comparison_score(y_true, y_probs, filename='ada_lear_')
            else:
                comparison_score(y_true, y_probs, filename='ada_stat_')

            base_estimator = DecisionTreeRegressor(max_depth=1)
            ada = AdaBoostRegressor(estimator=base_estimator, learning_rate=learning_rate,
                                    n_estimators=n_estimators)
            x_train = data_embedding.processed_data.drop(columns=['Anti', 'Activity'])
            y_train = data_embedding.processed_data[['Activity']]
            ada.fit(x_train, y_train.values.reshape(-1))

            return ada, r2, mse, mae, r, p

        elif input_type == 'structure':
            if _nn:
                data_embedding = GraphEmbeddingModel(mode='regression')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=STRUCTURE_REG_EMBEDDING_PATH)
                data_embedding.embedding()
            else:
                data_embedding = FingerprintEmbeddingModel(mode='regression')
                data_embedding.load_data(paths=path)
                data_embedding.embedding(fpSize=1024, fingerprint='RDK')

            r2, mse, mae, r, p, y_true, y_probs = ([] for _ in range(7))
            for _ in tqdm(range(n_rolls), desc='Adaboost training'):
                base_estimator = DecisionTreeRegressor(max_depth=1)
                ada = AdaBoostRegressor(estimator=base_estimator, learning_rate=learning_rate,
                                        n_estimators=n_estimators)
                # Split positive and negative samples into training and test sets
                x_train, y_train, x_test, y_test = data_embedding.split(split_rate=TEST_RATIO)

                ada.fit(x_train, y_train.values.ravel())
                predictions = ada.predict(x_test)
                y_true.append(y_test.values.ravel().tolist())
                y_probs.append(predictions)

                # Compute classification evaluation metrics
                score = regression_score(y_test, predictions)
                r2.append(score[0])
                mse.append(score[1])
                mae.append(score[2])
                r.append(score[3])
                p.append(score[4])

            if _nn:
                comparison_score(y_true, y_probs, filename='ada_lear_')
            else:
                comparison_score(y_true, y_probs, filename='ada_stat_')

            base_estimator = DecisionTreeRegressor(max_depth=1)
            ada = AdaBoostRegressor(estimator=base_estimator, learning_rate=learning_rate,
                                    n_estimators=n_estimators)
            x_train = data_embedding.processed_data.drop(columns=['Anti', 'Activity'])
            y_train = data_embedding.processed_data[['Activity']]
            ada.fit(x_train, y_train.values.reshape(-1))

            return ada, r2, mse, mae, r, p

        else:
            raise ValueError(f"Unexpected input type: {input_type}")

    else:
        raise ValueError(f'Model type "{mode}" is not supported.')


def MLP_evaluation(
        path: list[Path], mode: MODE, input_type: INPUT_TYPE, _nn: bool, n_rolls: int = DEFAULT_ROLLS,
        custom_threshold: float = DEFAULT_THRESHOLD, epochs: int = 10000, lr: float = 1e-4
) -> tuple:
    if mode == 'classification':
        # Load the dataset
        if input_type == 'phychem':
            if _nn:
                data_embedding = PhyChemEmbeddingModel(mode='classification')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=PHYCHEM_CLA_EMBEDDING_PATH)
                data_embedding.embedding()
            else:
                data_embedding = DescriptorsEmbeddingModel(mode='classification')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=DESCRIPTORS_EMBEDDING_PATH)
                data_embedding.embedding()
        elif input_type == 'structure':
            if _nn:
                data_embedding = GraphEmbeddingModel(mode='classification')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=STRUCTURE_CLA_EMBEDDING_PATH)
                data_embedding.embedding()
            else:
                data_embedding = FingerprintEmbeddingModel(mode='classification')
                data_embedding.load_data(paths=path)
                data_embedding.embedding(fpSize=2048, fingerprint='RDK')
        else:
            raise ValueError(f"Unexpected input type: {input_type}")
        accuracy, precision, recall, f1, auc, auc_true, auc_prob = ([] for _ in range(7))

        for _ in tqdm(range(n_rolls), desc='MLP training'):
            x_train, y_train, x_test, y_test = data_embedding.split(split_rate=TEST_RATIO)
            neural_dim = x_train.shape[1]
            X_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
            Y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
            X_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)

            if input_type == 'phychem':
                MLP = nn.Sequential(
                    nn.Linear(in_features=neural_dim, out_features=neural_dim * 2),
                    nn.ReLU(),
                    nn.BatchNorm1d(neural_dim * 2),
                    nn.Dropout(0.3),
                    nn.Linear(in_features=neural_dim * 2, out_features=neural_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(neural_dim),
                    nn.Dropout(0.3),
                    nn.Linear(in_features=neural_dim, out_features=1)
                )

            elif input_type == 'structure':
                MLP = nn.Sequential(
                    nn.Linear(in_features=neural_dim, out_features=neural_dim * 2),
                    nn.ReLU(),
                    nn.BatchNorm1d(neural_dim * 2),
                    nn.Dropout(0.4),
                    nn.Linear(in_features=neural_dim * 2, out_features=neural_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(neural_dim),
                    nn.Dropout(0.3),
                    nn.Linear(in_features=neural_dim, out_features=neural_dim // 2),
                    nn.ReLU(),
                    nn.BatchNorm1d(neural_dim // 2),
                    nn.Dropout(0.3),
                    nn.Linear(in_features=neural_dim // 2, out_features=1)
                )
            else:
                raise ValueError(f"Unexpected input type: {input_type}")

            # 定义训练参数
            criterion = nn.BCEWithLogitsLoss()  # 如果是 classification
            optimizer = torch.optim.Adam(MLP.parameters(), lr=lr)

            MLP.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = MLP(X_train_tensor)
                loss = criterion(outputs, Y_train_tensor)
                loss.backward()
                optimizer.step()

            MLP.eval()
            with torch.no_grad():
                logits = MLP(X_test_tensor)
                probs = torch.sigmoid(logits).squeeze().numpy()
                predictions = (probs > custom_threshold).astype(int)

            auc.append(roc_auc_score(y_test, probs))
            score = classification_score(y_test, list(predictions))
            accuracy.append(score[0])
            precision.append(score[1])
            recall.append(score[2])
            f1.append(score[3])
            auc_true.append(y_test)
            auc_prob.append(probs)

        if _nn:
            roc_score(auc_true, auc_prob, filename="mlp_lear")
        else:
            roc_score(auc_true, auc_prob, filename="mlp_stat")
        x_train = data_embedding.processed_data.drop(columns=['Anti', 'Activity'])
        y_train = data_embedding.processed_data[['Anti']]
        X_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
        Y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        neural_dim = x_train.shape[1]

        if input_type == 'phychem':
            MLP = nn.Sequential(
                nn.Linear(in_features=neural_dim, out_features=neural_dim * 2),
                nn.ReLU(),
                nn.BatchNorm1d(neural_dim * 2),
                nn.Dropout(0.3),
                nn.Linear(in_features=neural_dim * 2, out_features=neural_dim),
                nn.ReLU(),
                nn.BatchNorm1d(neural_dim),
                nn.Dropout(0.3),
                nn.Linear(in_features=neural_dim, out_features=1)
            )
        elif input_type == 'structure':
            MLP = nn.Sequential(
                nn.Linear(in_features=neural_dim, out_features=neural_dim * 2),
                nn.ReLU(),
                nn.BatchNorm1d(neural_dim * 2),
                nn.Dropout(0.4),
                nn.Linear(in_features=neural_dim * 2, out_features=neural_dim),
                nn.ReLU(),
                nn.BatchNorm1d(neural_dim),
                nn.Dropout(0.3),
                nn.Linear(in_features=neural_dim, out_features=neural_dim // 2),
                nn.ReLU(),
                nn.BatchNorm1d(neural_dim // 2),
                nn.Dropout(0.3),
                nn.Linear(in_features=neural_dim // 2, out_features=1)
            )
        else:
            raise ValueError(f"Unexpected input type: {input_type}")

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(MLP.parameters(), lr=lr)
        MLP.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = MLP(X_train_tensor)
            loss = criterion(outputs, Y_train_tensor)
            loss.backward()
            optimizer.step()

        return MLP, accuracy, precision, recall, f1, auc

    elif mode == 'regression':
        # Load the dataset
        if input_type == 'phychem':
            if _nn:
                data_embedding = PhyChemEmbeddingModel(mode='regression')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=PHYCHEM_REG_EMBEDDING_PATH)
                data_embedding.embedding()
            else:
                data_embedding = DescriptorsEmbeddingModel(mode='regression')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=DESCRIPTORS_EMBEDDING_PATH)
                data_embedding.embedding()

            r2, mse, mae, r, p, y_true, y_probs = ([] for _ in range(7))
            for _ in tqdm(range(n_rolls), desc='MLP training'):
                x_train, y_train, x_test, y_test = data_embedding.split(split_rate=TEST_RATIO)
                neural_dim = x_train.shape[1]
                X_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
                Y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
                X_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)

                if input_type == 'phychem':
                    MLP = nn.Sequential(
                        nn.Linear(in_features=neural_dim, out_features=neural_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(neural_dim),
                        nn.Linear(neural_dim, 1)
                    )
                elif input_type == 'structure':
                    MLP = nn.Sequential(
                        nn.Linear(in_features=neural_dim, out_features=neural_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(neural_dim),
                        nn.Linear(in_features=neural_dim, out_features=neural_dim // 2),
                        nn.ReLU(),
                        nn.BatchNorm1d(neural_dim // 2),
                        nn.Linear(neural_dim // 2, 1)
                    )
                else:
                    raise ValueError(f"Unexpected input type: {input_type}")

                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(MLP.parameters(), lr=lr)

                MLP.train()
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    outputs = MLP(X_train_tensor)
                    loss = criterion(outputs, Y_train_tensor)
                    loss.backward()
                    optimizer.step()
                MLP.eval()
                with torch.no_grad():
                    predictions = MLP(X_test_tensor).squeeze().cpu().numpy()

                y_true.append(y_test.values.ravel().tolist())
                y_probs.append(predictions)

                # Compute classification evaluation metrics
                score = regression_score(y_test, list(predictions))
                r2.append(score[0])
                mse.append(score[1])
                mae.append(score[2])
                r.append(score[3])
                p.append(score[4])

            if _nn:
                comparison_score(y_true, y_probs, filename='mlp_lear_')
            else:
                comparison_score(y_true, y_probs, filename='mlp_stat_')

            x_train = data_embedding.processed_data.drop(columns=['Anti', 'Activity'])
            y_train = data_embedding.processed_data[['Activity']]
            X_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
            Y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
            neural_dim = x_train.shape[1]
            if input_type == 'phychem':
                MLP = nn.Sequential(
                    nn.Linear(in_features=neural_dim, out_features=neural_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(neural_dim),
                    nn.Linear(neural_dim, 1)
                )
            elif input_type == 'structure':
                MLP = nn.Sequential(
                    nn.Linear(in_features=neural_dim, out_features=neural_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(neural_dim),
                    nn.Linear(in_features=neural_dim, out_features=neural_dim // 2),
                    nn.ReLU(),
                    nn.BatchNorm1d(neural_dim // 2),
                    nn.Linear(neural_dim // 2, 1)
                )
            else:
                raise ValueError(f"Unexpected input type: {input_type}")

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(MLP.parameters(), lr=lr)
            MLP.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = MLP(X_train_tensor)
                loss = criterion(outputs, Y_train_tensor)
                loss.backward()
                optimizer.step()

        elif input_type == 'structure':
            if _nn:
                data_embedding = GraphEmbeddingModel(mode='regression')
                data_embedding.load_data(paths=path)
                data_embedding.reload(path=STRUCTURE_REG_EMBEDDING_PATH)
                data_embedding.embedding()
            else:
                data_embedding = FingerprintEmbeddingModel(mode='regression')
                data_embedding.load_data(paths=path)
                data_embedding.embedding(fpSize=256, fingerprint='RDK')

            r2, mse, mae, r, p, y_true, y_probs = ([] for _ in range(7))
            for _ in tqdm(range(n_rolls), desc='MLP training'):
                x_train, y_train, x_test, y_test = data_embedding.split(split_rate=TEST_RATIO)
                neural_dim = x_train.shape[1]
                X_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
                Y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
                X_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)

                if input_type == 'phychem':
                    MLP = nn.Sequential(
                        nn.Linear(in_features=neural_dim, out_features=neural_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(neural_dim),
                        nn.Linear(neural_dim, 1)
                    )
                elif input_type == 'structure':
                    MLP = nn.Sequential(
                        nn.Linear(in_features=neural_dim, out_features=neural_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(neural_dim),
                        nn.Linear(in_features=neural_dim, out_features=neural_dim // 2),
                        nn.ReLU(),
                        nn.BatchNorm1d(neural_dim // 2),
                        nn.Linear(neural_dim // 2, 1)
                    )
                else:
                    raise ValueError(f"Unexpected input type: {input_type}")

                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(MLP.parameters(), lr=lr)

                MLP.train()
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    outputs = MLP(X_train_tensor)
                    loss = criterion(outputs, Y_train_tensor)
                    loss.backward()
                    optimizer.step()
                MLP.eval()
                with torch.no_grad():
                    predictions = MLP(X_test_tensor).squeeze().cpu().numpy()

                y_true.append(y_test.values.ravel().tolist())
                y_probs.append(predictions)

                # Compute classification evaluation metrics
                score = regression_score(y_test, list(predictions))
                r2.append(score[0])
                mse.append(score[1])
                mae.append(score[2])
                r.append(score[3])
                p.append(score[4])

            if _nn:
                comparison_score(y_true, y_probs, filename='mlp_lear_')
            else:
                comparison_score(y_true, y_probs, filename='mlp_stat_')

            x_train = data_embedding.processed_data.drop(columns=['Anti', 'Activity'])
            y_train = data_embedding.processed_data[['Activity']]
            X_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
            Y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
            neural_dim = x_train.shape[1]
            if input_type == 'phychem':
                MLP = nn.Sequential(
                    nn.Linear(in_features=neural_dim, out_features=neural_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(neural_dim),
                    nn.Linear(neural_dim, 1)
                )
            elif input_type == 'structure':
                MLP = nn.Sequential(
                    nn.Linear(in_features=neural_dim, out_features=neural_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(neural_dim),
                    nn.Linear(in_features=neural_dim, out_features=neural_dim // 2),
                    nn.ReLU(),
                    nn.BatchNorm1d(neural_dim // 2),
                    nn.Linear(neural_dim // 2, 1)
                )
            else:
                raise ValueError(f"Unexpected input type: {input_type}")

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(MLP.parameters(), lr=lr)
            MLP.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = MLP(X_train_tensor)
                loss = criterion(outputs, Y_train_tensor)
                loss.backward()
                optimizer.step()

        else:
            raise ValueError(f"Unexpected input type: {input_type}")

        return MLP, r2, mse, mae, r, p
    return None
