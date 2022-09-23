import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
import mlflow

from loguru import logger
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    average_precision_score,
    recall_score,
    precision_score,
)
from sklearn.base import is_classifier

from sweet_ml.visualisation import plot_confusion_matrix

from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTE

import torch
from omegaconf import OmegaConf
from ml_gapfill.visualisation import plot_models_dendrogram, plot_models_dendrogram_v2

import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc


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


def make_roc_curve(model, X, y_true):
    y_score = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    fig = px.area(
        x=fpr,
        y=tpr,
        title=f"ROC Curve (AUC={auc(fpr, tpr):.4f})",
        labels=dict(x="False Positive Rate", y="True Positive Rate"),
        width=700,
        height=500,
    )
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain="domain")

    return fig


def make_precision_recall_curve(model, X, y_true):
    y_score = model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    fig = px.area(
        x=recall,
        y=precision,
        title=f"Precision-Recall Curve (AUC={auc(fpr, tpr):.4f})",
        labels=dict(x="Recall", y="Precision"),
        width=700,
        height=500,
    )
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=1, y1=0)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain="domain")

    return fig


def generate_model_metrics(model, data_loader):
    logger.info("Generating validation metrics...")

    X_train, y_train = data_loader.get_train_data_arrays()
    X_val, y_val = data_loader.get_val_data_arrays()
    X_test, y_test = data_loader.get_test_data_arrays()

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    if is_classifier(model):

        if len(X_val) != 0:
            y_pred_val = model.predict(X_val)

            f1_val = f1_score(y_val, y_pred_val)
            acc_val = accuracy_score(y_val, y_pred_val)
            prec_val = precision_score(y_val, y_pred_val)
            recall_val = recall_score(y_val, y_pred_val)

            mlflow.log_metric("acc_validation", acc_val)
            mlflow.log_metric("f1_validation", f1_val)
            mlflow.log_metric("prec_val", prec_val)
            mlflow.log_metric("recall_val", recall_val)

        
        f1_train = f1_score(y_train, y_pred_train)
        f1_test = f1_score(y_test, y_pred_test)

        mlflow.log_metric("f1_train", f1_train)
        mlflow.log_metric("f1_test", f1_test)

        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)

        mlflow.log_metric("acc_train", acc_train)
        mlflow.log_metric("acc_test", acc_test)

        prec_train = precision_score(y_train, y_pred_train)
        prec_test = precision_score(y_test, y_pred_test)

        mlflow.log_metric("prec_train", prec_train)
        mlflow.log_metric("prec_test", prec_test)

        recall_train = recall_score(y_train, y_pred_train)
        recall_test = recall_score(y_test, y_pred_test)

        mlflow.log_metric("recall_train", recall_train)
        mlflow.log_metric("recall_test", recall_test)


def model_evaluation(model, data_loader):
    logger.info("Generating test metrics...")

    # Split dataframes into train, test arrays
    X_test, y_test = data_loader.get_test_data_arrays()

    y_pred = model.predict(X_test)

    f1_test = f1_score(y_test, y_pred)
    mlflow.log_metric("f1_test", f1_test)

    plot_confusion_matrix(y_test, X_test, model, "./test_confusion_matrix.png")

    y_pred = model.predict(X_test)

    test_df_labelled = data_loader.test_df.copy()
    test_df_labelled["Y_pred"] = y_pred

    fig = plot_models_dendrogram_v2(
        test_df_labelled, "test_dendrogram.html", confusion=False
    )
    fig.write_image("./test_dendrogram.pdf")
    mlflow.log_artifact("test_dendrogram.pdf")

    fig = plot_models_dendrogram_v2(test_df_labelled, "test_confusion_dendrogram.html")
    fig.write_html("./test_confusion_dendrogram.html")
    mlflow.log_artifact("test_confusion_dendrogram.html")

    fig.write_image("./test_confusion_dendrogram.pdf")
    mlflow.log_artifact("test_confusion_dendrogram.pdf")

    fig = make_roc_curve(model, X_test, y_test)
    fig.write_html("./test_roc_classification.html")
    mlflow.log_artifact("test_roc_classification.html")

    fig.write_image("./test_roc_classification.pdf")
    mlflow.log_artifact("test_roc_classification.pdf")

    fig = make_precision_recall_curve(model, X_test, y_test)
    fig.write_image("./test_pr_classification.pdf")
    mlflow.log_artifact("test_pr_classification.pdf")


def model_visualisation(model, data_loader):
    logger.info("Generating performance visualisations...")

    if is_classifier(model):
        X_val, y_val = data_loader.get_val_data_arrays()
        if len(X_val) > 0:
            plot_confusion_matrix(y_val, X_val, model, "./val_confusion_matrix.png")

            y_pred = model.predict(X_val)

            val_df_labelled = data_loader.val_df.copy()
            val_df_labelled["Y_pred"] = y_pred

            fig = plot_models_dendrogram_v2(
                val_df_labelled, "val_dendrogram.html", confusion=False
            )
            fig.write_image("./val_dendrogram.pdf")
            mlflow.log_artifact("val_dendrogram.pdf")

            fig = plot_models_dendrogram_v2(
                val_df_labelled, "val_confusion_dendrogram.html"
            )
            fig.write_html("./val_confusion_dendrogram.html")
            mlflow.log_artifact("val_confusion_dendrogram.html")

            fig.write_image("./val_confusion_dendrogram.pdf")
            mlflow.log_artifact("val_confusion_dendrogram.pdf")

            fig = make_roc_curve(model, X_val, y_val)
            fig.write_html("./val_roc_classification.html")
            mlflow.log_artifact("val_roc_classification.html")

            fig.write_image("./val_roc_classification.pdf")
            mlflow.log_artifact("val_roc_classification.pdf")

            fig = make_precision_recall_curve(model, X_val, y_val)
            fig.write_image("./val_pr_classification.pdf")
            mlflow.log_artifact("val_pr_classification.pdf")

            # fig = plot_models_dendrogram(val_df_labelled, "val_confusion_dendrogram.html", confusion=True)
            # fig.write_image("./fig_val_dendrogram_confusion.pdf")
            # mlflow.log_artifact("fig_val_dendrogram_confusion.pdf")

            # fig = plot_models_dendrogram(val_df_labelled, "val_dendrogram.html", confusion=False)
            # fig.write_image("./fig_val_dendrogram.pdf")
            # mlflow.log_artifact("fig_val_dendrogram.pdf")


def train_sklearn_model(cfg: DictConfig):
    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")
    mlflow.set_experiment(experiment_name=cfg.experiment_name)
    # mlflow.sklearn.autolog()

    experiment = mlflow.get_experiment_by_name(cfg.experiment_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        logger.info(f"Starting run...")

        log_params_from_omegaconf_dict(cfg)
        mlflow.log_metric("run_idx", cfg.run_idx)
        OmegaConf.save(cfg, "./config.yaml")
        mlflow.log_artifact("./config.yaml")

        # Initialise data loader
        data_loader = instantiate(cfg.data_loader, _recursive_=True, _convert_=all)
        data_loader.fit_preprocessing_pipeline()

        torch.save(data_loader.preprocessing_pipeline, "preprocessing.pth")
        mlflow.log_artifact("./preprocessing.pth")

        model = hydra.utils.instantiate(cfg.model, _recursive_=False)

        # Split dataframes Features and labels
        X_train, y_train = data_loader.get_train_data_arrays()
        X_val, y_val = data_loader.get_val_data_arrays()

        model.fit(X_train, y_train)

        generate_model_metrics(model, data_loader)

        model_visualisation(model, data_loader)

        model_evaluation(model, data_loader)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )


@hydra.main(config_path="./configs/train", config_name="config")
def main(cfg: DictConfig):
    if cfg.run_mode == "train_sklearn_model":
        train_sklearn_model(cfg)

    # elif cfg.run_mode == "train_pytorch_model":
    #     train_pytorch_model(cfg)


if __name__ == "__main__":
    main()
