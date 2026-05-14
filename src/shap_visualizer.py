import os
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pipeline_model = None
explainer = None


def initialize_shap(model):
    global pipeline_model, explainer

    pipeline_model = model

    if hasattr(model, "named_steps"):
        final_model = list(model.named_steps.values())[-1]
    else:
        final_model = model

    explainer = shap.TreeExplainer(final_model)


def prepare_model_input(patient_dict):
    input_df = pd.DataFrame([patient_dict])

    if hasattr(pipeline_model, "named_steps"):
        steps = list(pipeline_model.named_steps.values())

        if len(steps) > 1:
            preprocessor = steps[0]
            transformed_input = preprocessor.transform(input_df)

            if hasattr(transformed_input, "toarray"):
                transformed_input = transformed_input.toarray()

            try:
                feature_names = list(preprocessor.get_feature_names_out())
            except Exception:
                feature_names = list(input_df.columns)

            return transformed_input, feature_names

    return input_df, list(input_df.columns)


def generate_shap_bar_chart(patient_dict):
    if explainer is None:
        raise ValueError("SHAP explainer is not initialized")

    transformed_input, feature_names = prepare_model_input(patient_dict)

    shap_values = explainer.shap_values(transformed_input)

    if isinstance(shap_values, list):
        values = shap_values[0][0]
    else:
        values = np.array(shap_values)

        if values.ndim == 3:
            values = values[0, :, 0]
        elif values.ndim == 2:
            values = values[0]
        else:
            values = values.flatten()

    values = np.array(values).flatten()

    min_len = min(len(feature_names), len(values))

    feature_importance = pd.DataFrame({
        "feature": feature_names[:min_len],
        "shap_value": values[:min_len]
    })

    feature_importance["abs_value"] = feature_importance["shap_value"].abs()

    feature_importance = feature_importance.sort_values(
        by="abs_value",
        ascending=False
    ).head(10)

    os.makedirs("shap_outputs", exist_ok=True)

    output_path = "shap_outputs/latest_shap_bar.png"

    plt.figure(figsize=(10, 6))
    plt.barh(
        feature_importance["feature"],
        feature_importance["shap_value"]
    )
    plt.xlabel("SHAP Value")
    plt.ylabel("Feature")
    plt.title("Top Patient Risk Contributors")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path, feature_importance.to_dict(orient="records")