"""Evaluation"""
import numpy as np
from pathlib import Path
import joblib
import pandas as pd

from trade.constants import SOURCE_SET_PATH, TARGET_SET_PATH, DEFAULT_ROLLS, MODE, \
    STRUCTURE_LAYER_PATH, PHYCHEM_LAYER_PATH, RANKING_LAYER_PATH
from trade.model_selector.models import MLP_evaluation, RandomForest_evaluate, Xgboost_evaluate, Adaboost_evaluate
import shutil

width = shutil.get_terminal_size().columns
classification_evaluation = []
regression_evaluation = []


def str_to_bool(
        string: str = None
) -> bool:
    """Convert a string to a boolean value.

    :param string: The input string, expected to be 'True' or any other value.
    :return: True if the string is 'True', otherwise False.
    """
    if string == 'True':
        return True
    else:
        return False


def format_metric(value):
    if isinstance(value, (list, np.ndarray)):
        arr = np.array(value, dtype=float)
        if len(arr) > 2:
            arr = np.sort(arr)[1:-1]
        return f"{round(np.mean(arr), 3)} ± {round(np.std(arr, ddof=1), 3)}"
    else:
        return round(value, 3)


def evaluation_collect(model_name, mode: MODE, **kwargs):
    if mode == 'classification':
        AC, PR, RE, F1, AUC = (kwargs[k] for k in ('AC', 'PR', 'RE', 'F1', 'AUC'))
        result = {
            'Model': model_name,
            'Accuracy': format_metric(AC),
            'Precision': format_metric(PR),
            'Recall': format_metric(RE),
            'F1-score': format_metric(F1),
            'AUC': format_metric(AUC)
        }
        classification_evaluation.append(result)

    elif mode == 'regression':
        R2, MSE, MAE, R, p = (kwargs[k] for k in ('R2', 'MSE', 'MAE', 'R', 'p'))
        result = {
            'Model': model_name,
            'R2': format_metric(R2),
            'MSE': format_metric(MSE),
            'MAE': format_metric(MAE),
            'R': format_metric(R),
            'p': format_metric(p)
        }
        regression_evaluation.append(result)
    else:
        raise ValueError("Expected 3 (regression) or 4 (classification) keyword arguments: AC, PR, RE, (optional F1)")


def evaluate_classification(
        target_set_file: Path = TARGET_SET_PATH, save_model: bool = False, verbose: bool = False
) -> None:
    """Evaluate the classification model and display performance metrics.

    :param target_set_file: Path of the target dataset.
    :param save_model: Whether to save the trained model after evaluation.
    """
    print("─" * width)
    print('Evaluating the Pretrained Classification Model:')
    print()
    path_list = [target_set_file]

    print(f"Section 1 -> PhyChem Descriptors with Method-1 learning")
    # rf_1, AC, PR, RE, F1, AUC = RandomForest_evaluate(path=path_list, input_type='phychem', mode='classification',
    #                                                   custom_threshold=0.5, _nn=False, ccp_alpha=0, max_features='sqrt',
    #                                                   max_samples=0.8, min_weight_fraction_leaf=0, min_samples_split=3,
    #                                                   n_estimators=3000)
    # evaluation_collect('rf (PhyChem Method-1)', mode='classification', AC=AC, PR=PR, RE=RE, F1=F1, AUC=AUC)
    #
    # xgb_1, AC, PR, RE, F1, AUC = Xgboost_evaluate(path=path_list, input_type='phychem', mode='classification',
    #                                               _nn=False, custom_threshold=0.5)
    # evaluation_collect('xgboost (PhyChem Method-1)', mode='classification', AC=AC, PR=PR, RE=RE, F1=F1, AUC=AUC)
    #
    # ada_1, AC, PR, RE, F1, AUC = Adaboost_evaluate(path=path_list, input_type='phychem', mode='classification',
    #                                                _nn=False, custom_threshold=0.5, )
    # evaluation_collect('adaboost (PhyChem Method-1)', mode='classification', AC=AC, PR=PR, RE=RE, F1=F1, AUC=AUC)
    #
    # mlp_1, AC, PR, RE, F1, AUC = MLP_evaluation(path=path_list, input_type='phychem', mode='classification',
    #                                             _nn=False, custom_threshold=0.5, )
    # evaluation_collect('mlp (PhyChem Method-1)', mode='classification', AC=AC, PR=PR, RE=RE, F1=F1, AUC=AUC)

    print(f"Section 2 -> PhyChem Descriptors with Method-2 learning")
    # rf_2, AC, PR, RE, F1, AUC = RandomForest_evaluate(path=path_list, input_type='phychem', mode='classification',
    #                                                   custom_threshold=0.5, _nn=True, ccp_alpha=0, max_features='sqrt',
    #                                                   max_samples=0.8, min_weight_fraction_leaf=0, min_samples_split=3,
    #                                                   n_estimators=3000)
    # evaluation_collect('rf (PhyChem Method-2)', mode='classification', AC=AC, PR=PR, RE=RE, F1=F1, AUC=AUC)
    #
    xgb_2, AC, PR, RE, F1, AUC = Xgboost_evaluate(path=path_list, input_type='phychem', mode='classification',
                                                  _nn=True, custom_threshold=0.4)
    evaluation_collect('xgboost (PhyChem Method-2)', mode='classification', AC=AC, PR=PR, RE=RE, F1=F1, AUC=AUC)

    # ada_2, AC, PR, RE, F1, AUC = Adaboost_evaluate(path=path_list, input_type='phychem', mode='classification',
    #                                                _nn=True, custom_threshold=0.5)
    # evaluation_collect('adaboost (PhyChem Method-2)', mode='classification', AC=AC, PR=PR, RE=RE, F1=F1, AUC=AUC)

    # mlp_2, AC, PR, RE, F1, AUC = MLP_evaluation(path=path_list, input_type='phychem', mode='classification',
    #                                             _nn=True, custom_threshold=0.5)
    # evaluation_collect('mlp (PhyChem Method-2)', mode='classification', AC=AC, PR=PR, RE=RE, F1=F1, AUC=AUC)

    print(f"Section 3 -> Structure information with Method-1 learning")
    # rf_3, AC, PR, RE, F1, AUC = RandomForest_evaluate(path=path_list, input_type='structure', mode='classification',
    #                                                   _nn=False, custom_threshold=0.5)
    # evaluation_collect('rf (Structure Method-1)', mode='classification', AC=AC, PR=PR, RE=RE, F1=F1, AUC=AUC)
    #
    # xgb_3, AC, PR, RE, F1, AUC = Xgboost_evaluate(path=path_list, input_type='structure', mode='classification',
    #                                               _nn=False, custom_threshold=0.5)
    # evaluation_collect('xgboost (Structure Method-1)', mode='classification', AC=AC, PR=PR, RE=RE, F1=F1, AUC=AUC)
    #
    # ada_3, AC, PR, RE, F1, AUC = Adaboost_evaluate(path=path_list, input_type='structure', mode='classification',
    #                                                _nn=False, custom_threshold=0.5)
    # evaluation_collect('adaboost (Structure Method-1)', mode='classification', AC=AC, PR=PR, RE=RE, F1=F1, AUC=AUC)
    #
    # mlp_3, AC, PR, RE, F1, AUC = MLP_evaluation(path=path_list, input_type='structure', mode='classification',
    #                                             _nn=False, custom_threshold=0.5)
    # evaluation_collect('mlp (Structure Method-1)', mode='classification', AC=AC, PR=PR, RE=RE, F1=F1, AUC=AUC)

    print(f"Section 4 -> Structure information with Method-2 learning")
    rf_4, AC, PR, RE, F1, AUC = RandomForest_evaluate(path=path_list, input_type='structure', mode='classification',
                                                      _nn=True, custom_threshold=0.5, ccp_alpha=0, max_features='sqrt',
                                                      max_samples=0.9, min_weight_fraction_leaf=0, min_samples_split=3,
                                                      n_estimators=4000)
    evaluation_collect('rf (Structure Method-2)', mode='classification', AC=AC, PR=PR, RE=RE, F1=F1, AUC=AUC)

    # xgb_4, AC, PR, RE, F1, AUC = Xgboost_evaluate(path=path_list, input_type='structure', mode='classification',
    #                                               _nn=True, custom_threshold=0.5)
    # evaluation_collect('xgboost (Structure Method-2)', mode='classification', AC=AC, PR=PR, RE=RE, F1=F1, AUC=AUC)

    # ada_4, AC, PR, RE, F1, AUC = Adaboost_evaluate(path=path_list, input_type='structure', mode='classification',
    #                                                _nn=True, custom_threshold=0.5)
    # evaluation_collect('adaboost (Structure Method-2)', mode='classification', AC=AC, PR=PR, RE=RE, F1=F1, AUC=AUC)

    # mlp_4, AC, PR, RE, F1, AUC = MLP_evaluation(path=path_list, input_type='structure', mode='classification',
    #                                             _nn=True, custom_threshold=0.5)
    # evaluation_collect('mlp (Structure Method-2)', mode='classification', AC=AC, PR=PR, RE=RE, F1=F1, AUC=AUC)

    if verbose:
        result = pd.DataFrame(classification_evaluation)
        print(result)
    if save_model:
        # joblib.dump(xgb_2, PHYCHEM_LAYER_PATH)
        joblib.dump(rf_4, STRUCTURE_LAYER_PATH)


def evaluate_regression(
        source_set_file: Path = SOURCE_SET_PATH, target_set_file: Path = TARGET_SET_PATH,
        n_rolls: int = DEFAULT_ROLLS, save_model: bool = False, verbose: bool = False
) -> None:
    """Evaluate the regression model and display the performance by r^2 score.

    :param source_set_file: Path of the source dataset.
    :param target_set_file: Path of the target dataset.
    :param n_rolls: Number of test runs for evaluation.
    :param save_model: Whether to save the trained model after evaluation.
    """
    print("─" * width)
    print('Evaluating the Regression Model:')
    print()

    path_list = [TARGET_SET_PATH]
    # path_list = [SOURCE_SET_PATH]

    print(f"Section 1 -> PhyChem Descriptors with Method-1 learning")
    # rf_1, R2, MSE, MAE, R, p = RandomForest_evaluate(path=path_list, input_type='phychem', mode='regression',
    #                                                  custom_threshold=0.5, _nn=False, ccp_alpha=0.001,
    #                                                  max_features='log2', max_samples=0.8, min_weight_fraction_leaf=0,
    #                                                  min_samples_split=3, n_estimators=5000)
    # evaluation_collect('rf (PhyChem Method-1)', mode='regression', R2=R2, MSE=MSE, MAE=MAE, R=R, p=p)
    #
    # xgb_1, R2, MSE, MAE, R, p = Xgboost_evaluate(path=path_list, input_type='phychem', mode='regression', _nn=False,
    #                                              n_estimators=4000, learning_rate=0.05, max_depth=7, min_child_weight=1,
    #                                              colsample_bytree=1, gamma= 0.5, reg_alpha=1, reg_lambda=1)
    # evaluation_collect('xgboost (PhyChem Method-1)', mode='regression', R2=R2, MSE=MSE, MAE=MAE, R=R, p=p)
    #
    # ada_1, R2, MSE, MAE, R, p = Adaboost_evaluate(path=path_list, input_type='phychem', mode='regression', _nn=False,
    #                                               n_estimators=4000, learning_rate=0.05)
    # evaluation_collect('adaboost (PhyChem Method-1)', mode='regression', R2=R2, MSE=MSE, MAE=MAE, R=R, p=p)
    #
    # mlp_1, R2, MSE, MAE, R, p = MLP_evaluation(path=path_list, input_type='phychem', mode='regression', _nn=False)
    # evaluation_collect('mlp (PhyChem Method-1)', mode='regression', R2=R2, MSE=MSE, MAE=MAE, R=R, p=p)
    #
    print(f"Section 2 -> PhyChem Descriptors with Method-2 learning")
    # rf_2, R2, MSE, MAE, R, p = RandomForest_evaluate(path=path_list, input_type='phychem', mode='regression',
    #                                                  custom_threshold=0.5, _nn=True, ccp_alpha=0.0005,
    #                                                  max_features='log2', max_samples=0.8,
    #                                                  min_weight_fraction_leaf=0,
    #                                                  min_samples_split=3, n_estimators=5000)
    # evaluation_collect('rf (PhyChem Method-2)', mode='regression', R2=R2, MSE=MSE, MAE=MAE, R=R, p=p)

    # xgb_2, R2, MSE, MAE, R, p = Xgboost_evaluate(path=path_list, input_type='phychem', mode='regression', _nn=True,
    #                                              n_estimators=5000, learning_rate=0.05, max_depth=7, min_child_weight=1,
    #                                              colsample_bytree=1, gamma= 0.5, reg_alpha=1, reg_lambda=1)
    # evaluation_collect('xgboost (PhyChem Method-2)', mode='regression', R2=R2, MSE=MSE, MAE=MAE, R=R, p=p)

    # ada_2, R2, MSE, MAE, R, p = Adaboost_evaluate(path=path_list, input_type='phychem', mode='regression', _nn=True,
    #                                               n_estimators=4000, learning_rate=0.05)
    # evaluation_collect('adaboost (PhyChem Method-2)', mode='regression', R2=R2, MSE=MSE, MAE=MAE, R=R, p=p)

    # mlp_2, R2, MSE, MAE, R, p = MLP_evaluation(path=path_list, input_type='phychem', mode='regression', _nn=True)
    # evaluation_collect('mlp (PhyChem Method-2)', mode='regression', R2=R2, MSE=MSE, MAE=MAE, R=R, p=p)

    print(f"Section 3 -> Structure information with Method-1 learning")
    # rf_3, R2, MSE, MAE, R, p = RandomForest_evaluate(path=path_list, input_type='structure', mode='regression',
    #                                                  custom_threshold=0.5, _nn=False, ccp_alpha=0.0005,
    #                                                  max_features='log2', max_samples=0.8, min_weight_fraction_leaf=0,
    #                                                  min_samples_split=3, n_estimators=500)
    # evaluation_collect('rf (Structure Method-1)', mode='regression', R2=R2, MSE=MSE, MAE=MAE, R=R, p=p)

    # xgb_3, R2, MSE, MAE, R, p = Xgboost_evaluate(path=path_list, input_type='structure', mode='regression', _nn=False,
    #                                              n_estimators=5000, learning_rate=0.05, max_depth=7, min_child_weight=1,
    #                                              colsample_bytree=1, gamma= 0.5, reg_alpha=1, reg_lambda=1)
    # evaluation_collect('xgboost (Structure Method-1)', mode='regression', R2=R2, MSE=MSE, MAE=MAE, R=R, p=p)

    # ada_3, R2, MSE, MAE, R, p = Adaboost_evaluate(path=path_list, input_type='structure', mode='regression', _nn=False,
    #                                               n_estimators=4000, learning_rate=0.05)
    # evaluation_collect('adaboost (Structure Method-1)', mode='regression', R2=R2, MSE=MSE, MAE=MAE, R=R, p=p)

    # mlp_3, R2, MSE, MAE, R, p = MLP_evaluation(path=path_list, input_type='structure', mode='regression', _nn=False)
    # evaluation_collect('mlp (Structure Method-1)', mode='regression', R2=R2, MSE=MSE, MAE=MAE, R=R, p=p)
    #
    print(f"Section 4 -> Structure information with Method-2 learning")

    rf_4, R2, MSE, MAE, R, p = RandomForest_evaluate(path=path_list, input_type='structure', mode='regression',
                                                     custom_threshold=0.5, _nn=True, ccp_alpha=0.0005,
                                                     max_features='log2', max_samples=0.8, min_weight_fraction_leaf=0,
                                                     min_samples_split=3, n_estimators=5000)
    evaluation_collect('rf (Structure Method-2)', mode='regression', R2=R2, MSE=MSE, MAE=MAE, R=R, p=p)

    # xgb_4, R2, MSE, MAE, R, p = Xgboost_evaluate(path=path_list, input_type='structure', mode='regression', _nn=True,
    #                                              n_estimators=5000, learning_rate=0.05, max_depth=7, min_child_weight=1,
    #                                              colsample_bytree=1, gamma= 0.5, reg_alpha=1, reg_lambda=1)
    # evaluation_collect('xgboost (Structure Method-2)', mode='regression', R2=R2, MSE=MSE, MAE=MAE, R=R, p=p)

    # ada_4, R2, MSE, MAE, R, p = Adaboost_evaluate(path=path_list, input_type='structure', mode='regression',_nn=True,
    #                                               n_estimators=4000, learning_rate=0.05)
    # evaluation_collect('adaboost (Structure Method-2)', mode='regression', R2=R2, MSE=MSE, MAE=MAE, R=R, p=p)

    # mlp_4, R2, MSE, MAE, R, p = MLP_evaluation(path=path_list, input_type='structure', mode='regression', _nn=True)
    # evaluation_collect('mlp (Structure Method-2)', mode='regression', R2=R2, MSE=MSE, MAE=MAE, R=R, p=p)

    if verbose:
        result = pd.DataFrame(regression_evaluation)
        print(result)
    if save_model:
        joblib.dump(rf_4, RANKING_LAYER_PATH)


def model_evaluate(
        classification_evaluate: str = "True", regression_evaluate: str = "True", save_model: str = "False"
) -> None:
    """Evaluate classification and regression models, with an option to save the results.

    :param classification_evaluate: Whether to evaluate the classification model.
    :param regression_evaluate: Whether to evaluate the regression model.
    :param save_model: Whether to save the trained models after evaluation.
    """
    classification_evaluate = str_to_bool(classification_evaluate)
    regression_evaluate = str_to_bool(regression_evaluate)
    save_model = str_to_bool(save_model)

    # if any(MODEL_SAVE_PATH.glob('*.pkl')):
    #     print()
    #     print('Deleting existed Models:')
    # for file in MODEL_SAVE_PATH.glob('*.pkl'):
    #     try:
    #         file.unlink()
    #         print(f"\t Deleted: {file.name}")
    #     except Exception as e:
    #         print(f"Error deleting {file.name}: {e}")

    if classification_evaluate:
        evaluate_classification(save_model=save_model)
    if regression_evaluate:
        evaluate_regression(save_model=save_model)


if __name__ == '__main__':
    """Run generate function from command line."""
    from tap import tapify

    tapify(model_evaluate)
