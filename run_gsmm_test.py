import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
import mlflow

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, average_precision_score
from sklearn.metrics import ConfusionMatrixDisplay

from ml_gapfill.fba_classifier import FBAClassifier
from ml_gapfill.visualisation import plot_models_dendrogram_v2

import matplotlib.pyplot as plt

from pathlib import Path


def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)


def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f"{parent_name}.{k}", v)
            else:
                mlflow.log_param(f"{parent_name}.{k}", v)

    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f"{parent_name}.{i}", v)

            else:
                mlflow.log_param(f"{parent_name}.{i}", v)


def generate_model_metrics(
    model, test_df, compound_prefix, species_label_column, growth_label_column
):
    y_true = test_df.loc[:, growth_label_column].values

    y_pred = model.predict(test_df, compound_prefix, species_label_column)

    f1_test = f1_score(y_true, y_pred)

    mlflow.log_metric("f1_test", f1_test)

    acc_test = accuracy_score(y_true, y_pred)

    mlflow.log_metric("acc_test", acc_test)

    prec_test = average_precision_score(y_true, y_pred)
    mlflow.log_metric("prec_test", prec_test)

    #
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        cmap=plt.cm.Blues,
    )
    mlflow.log_figure(disp.figure_, "./test_confusion_matrix.png")

    test_df["Y_pred"] = y_pred

    fig = plot_models_dendrogram_v2(test_df, "test_dendrogram.html", confusion=False)
    fig.write_image("./test_dendrogram.pdf")
    mlflow.log_artifact("test_dendrogram.pdf")
    fig.write_html("./test_dendrogram.html")
    mlflow.log_artifact("test_dendrogram.html")

    fig = plot_models_dendrogram_v2(
        test_df, "test_confusion_dendrogram.html", confusion=True
    )
    fig.write_image("./test_confusion_dendrogram.pdf")
    mlflow.log_artifact("test_confusion_dendrogram.pdf")

    fig.write_html("./test_confusion_dendrogram.html")
    mlflow.log_artifact("test_confusion_dendrogram.html")


@hydra.main(config_path="./configs/fba_configs", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")
    mlflow.set_experiment(experiment_name=cfg.experiment_name)
    # mlflow.sklearn.autolog()

    experiment = mlflow.get_experiment_by_name(cfg.experiment_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id):

        print(cfg)
        # Load test data
        test_df = pd.read_csv(cfg.test_data)

        test_df["has_gsmm"] = 0

        # Look for models
        for species_idx, species in enumerate(test_df[cfg.species_label].unique()):
            model_path = f"{cfg.models_dir}/{species}.xml"
            if Path(model_path).exists():
                print(model_path)
                test_df.loc[test_df[cfg.species_label] == species, "has_gsmm"] = 1

            else:
                test_df.loc[test_df[cfg.species_label] == species, "has_gsmm"] = 0

        test_df = test_df.loc[test_df["has_gsmm"] == 1]
        test_df.drop("has_gsmm", inplace=True, axis=1)

        # Initialise FBA classifier
        model = FBAClassifier(models_dir=cfg.models_dir)
        generate_model_metrics(
            model,
            test_df,
            compound_prefix=cfg.compound_prefix,
            species_label_column=cfg.species_label,
            growth_label_column=cfg.growth_label,
        )


if __name__ == "__main__":
    main()
