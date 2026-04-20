import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from tap import tapify
from scripts.constants import DATA_DIR, SHAP_DIR, SHAP_FEATURES


def Random_Forest(X_train, Y_train):
    RF = RandomForestClassifier(ccp_alpha=0, max_features=4, max_samples=21, min_impurity_decrease=0,
                                min_samples_split=3, min_weight_fraction_leaf=0, n_estimators=10000, oob_score=True,
                                random_state=5)
    RF.fit(X_train, Y_train.values.reshape(-1))
    return RF


def filter_shap_features(selected_keys):
    return {key: SHAP_FEATURES[key] for key in selected_keys if key in SHAP_FEATURES}


def plot_detailed_SHAP(
        set_path: Path = DATA_DIR,
        shap_types: str = None,
        Save: bool = False,
) -> None:
    shap.initjs()
    # Check if the shap_type
    if shap_types == 'NP_Characters':
        selected_keys = ['NP Characters', 'Physical Property', 'Structure Property',
                         'Topological Property', 'Surface Area', 'E-State', 'Fragments', 'Drug Descriptors']
        filtered_features = {key: SHAP_FEATURES[key] for key in selected_keys if key in SHAP_FEATURES}
        all_features = sum(filtered_features.values(), [])
        TrainingSet = pd.read_csv(SHAP_DIR)
        X = TrainingSet[all_features]
        Y = TrainingSet['SA2']
        scaler = StandardScaler()
        scaler.fit(X)
        X = pd.DataFrame(scaler.transform(X), columns=all_features)
        RF = Random_Forest(X, Y)
        explainer = shap.TreeExplainer(RF)
        shap_values = explainer(X)
        shap_array = shap_values.values
        feature_names = shap_values.feature_names

        merged_shap_list = []
        merged_feature_names = []

        for group_name, feature_list in filtered_features.items():
            indices = [feature_names.index(f) for f in feature_list if f in feature_names]
            group_shap = shap_array[:, indices].sum(axis=1, keepdims=True)
            merged_shap_list.append(group_shap)
            merged_feature_names.append(group_name)
        merged_shap_array = np.hstack(merged_shap_list)
        merged_shap_array_final = merged_shap_array[:, :, 1] - merged_shap_array[:, :, 0]
        merged_shap = shap.Explanation(values=merged_shap_array_final, feature_names=merged_feature_names)

        shap.plots.bar(merged_shap, max_display=8, show=True)
        shap.plots.beeswarm(merged_shap, max_display=8, show=True)

        if Save:
            shap_df = pd.DataFrame(merged_shap_array_final, columns=merged_feature_names)
            shap_df['Sample_Index'] = np.arange(len(shap_df))
            shap_df.to_csv(set_path / f'SHAP_{shap_types}_merged.csv', index=False)

    elif shap_types == 'Nano_particle':
        # Filter features related to 'NP Characters'
        filtered_features = {key: value for key, value in SHAP_FEATURES.items() if key in ['NP Characters']}
        all_features = sum(filtered_features.values(), [])
        # Load the training dataset
        TrainingSet = pd.read_csv(SHAP_DIR)
        X = TrainingSet[all_features]
        Y = TrainingSet['SA2']
        # Scale the feature data
        scaler = StandardScaler()
        scaler.fit(X)
        X = pd.DataFrame(scaler.transform(X), columns=all_features)
        RF = Random_Forest(X, Y)

        # Use SHAP's TreeExplainer to calculate SHAP values for the features
        explainer = shap.TreeExplainer(RF)
        shap_values = explainer(X)
        # Create a new SHAP explanation object with the calculated SHAP values
        new_shap = shap.Explanation(values=shap_values, feature_names=shap_values.feature_names)
        # Create a new summary bar plot with SHAP values, displaying the top features
        shap.plots.bar(new_shap[:, :, 1], max_display=10, show=True)
        shap.plots.beeswarm(new_shap[:, :, 1], max_display=10, color='magma', show=True, alpha=0.6)

        if Save:
            import matplotlib.pyplot as plt
            plt.tight_layout()
            plt.savefig(set_path / f"SHAP_{shap_types}.svg", format='svg', bbox_inches='tight')
            plt.close()

            final_shap_values = shap_values.values[:, :, 1] - shap_values.values[:, :, 0]
            shap_df = pd.DataFrame(final_shap_values, columns=shap_values.feature_names)
            shap_df['Sample_Index'] = np.arange(len(shap_df))
            shap_df.to_csv(set_path / f"SHAP_{shap_types}.csv", index=False)

    elif shap_types == 'Ligand':
        # Filter features related to 'Physical Property', 'Structure Property', etc.
        filtered_features = {key: value for key, value in SHAP_FEATURES.items() if
                             key in ['Physical Property', 'Structure Property', 'Topological Property', 'Surface Area',
                                     'E-State', 'Fragments', 'Drug Descriptors']}
        all_features = sum(filtered_features.values(), [])
        # Load the training dataset
        TrainingSet = pd.read_csv(SHAP_DIR)
        X = TrainingSet[all_features]
        Y = TrainingSet['SA2']
        # Scale the feature data
        scaler = StandardScaler()
        scaler.fit(X)
        X = pd.DataFrame(scaler.transform(X), columns=all_features)
        RF = Random_Forest(X, Y)

        # Use SHAP's TreeExplainer to calculate SHAP values for the features
        explainer = shap.TreeExplainer(RF)
        shap_values = explainer(X)
        # Create a new SHAP explanation object with the calculated SHAP values
        new_shap = shap.Explanation(values=shap_values, feature_names=shap_values.feature_names)
        # Create a new summary bar plot with SHAP values, displaying the top features
        shap.plots.bar(new_shap[:, :, 1], max_display=10, show=True)

        final_shap_values = shap_values.values[:, :, 1] - shap_values.values[:, :, 0]

        mean_abs_shap = np.abs(final_shap_values).mean(axis=0)
        top_10_indices = np.argsort(mean_abs_shap)[-10:][::-1]  # 排序取前10（从大到小）

        top_10_shap_values = final_shap_values[:, top_10_indices]
        top_10_features = [shap_values.feature_names[i] for i in top_10_indices]
        top_10_X = X.iloc[:, top_10_indices]

        top10_explanation = shap.Explanation(
            values=top_10_shap_values,
            feature_names=top_10_features,
            data=top_10_X
        )
        import matplotlib.pyplot as plt
        shap.plots.beeswarm(top10_explanation, max_display=10, color='magma', show=True, alpha=0.6)
        if Save:
            plt.tight_layout()
            plt.savefig(set_path / f"SHAP_{shap_types}_Ligand_top10.svg", format='svg', bbox_inches='tight')
            plt.close()

    elif shap_types == 'Ligand_details':

        # Filter features related to 'Physical Property', 'Structure Property', etc.
        filtered_features = {key: value for key, value in SHAP_FEATURES.items() if
                             key in ['Physical Property', 'Structure Property', 'Topological Property', 'Surface Area',
                                     'E-State', 'Fragments', 'Drug Descriptors']}
        all_features = sum(filtered_features.values(), [])
        # Load the training dataset
        TrainingSet = pd.read_csv(SHAP_DIR)
        X = TrainingSet[all_features]
        Y = TrainingSet['SA2']
        # Scale the feature data
        scaler = StandardScaler()
        scaler.fit(X)
        X = pd.DataFrame(scaler.transform(X), columns=all_features)
        RF = Random_Forest(X, Y)

        # Use SHAP's TreeExplainer to calculate SHAP values for the features
        explainer = shap.TreeExplainer(RF)
        shap_values = explainer(X)
        # Create a new SHAP explanation object with the calculated SHAP values
        new_shap = shap.Explanation(values=shap_values, feature_names=shap_values.feature_names)
        # Create a new summary bar plot with SHAP values, displaying the top features
        shap.plots.bar(new_shap[:, :, 1], max_display=250, show=True)
        import matplotlib.pyplot as plt
        shap.plots.beeswarm(new_shap[:, :, 1], max_display=250, color='magma', show=True, alpha=0.6)
        if Save:

            plt.tight_layout()
            plt.savefig(set_path / f"SHAP_{shap_types}.svg", format='svg', bbox_inches='tight')
            plt.close()

            final_shap_values = shap_values.values[:, :, 1] - shap_values.values[:, :, 0]
            shap_df = pd.DataFrame(final_shap_values, columns=shap_values.feature_names)
            shap_df['Sample_Index'] = np.arange(len(shap_df))
            shap_df.to_csv(set_path / f"SHAP_{shap_types}.csv", index=False)

    else:
        # If shap_types is neither 'Ligand' nor 'Nano_particle', print an error
        print('Error: shap_types must be defined.')


if __name__ == '__main__':
    """Run generate function from command line."""
    tapify(plot_detailed_SHAP)


