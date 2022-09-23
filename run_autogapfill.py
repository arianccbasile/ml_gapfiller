import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
import mlflow
import cobra
import pandas as pd
import pickle
import torch
from glob import glob
import subprocess
import shutil
from pathlib import Path

"""
INPUTS:
    GSMM directory
    Species name
    Media features (M1 - M10)
    Classifier path

Steps:
    Load GSMM
    Extract Features
    Assemble species-media rows

    Get classification for each row
    Perform multigapfill according to rows predicted to grow

    Save model

"""


def combine_media_model_data(media_features_df, model_features_df):

    species_labels = model_features_df.Y_species_label.unique()
    media_labels = media_features_df.media_label.unique()
    media_features_df.rename({"media_label": "Y_media_label"}, inplace=True, axis=1)

    media_features_df.rename(
        columns={
            i: f"X_MEDIA_{i}"
            for i in media_features_df.loc[
                :, ~media_features_df.columns.str.startswith("Y_")
            ].columns
        },
        inplace=True,
    )
    features = []

    # Make a dataframe of models for each media
    for s in species_labels:
        for m in media_labels:

            species_row = model_features_df.loc[
                model_features_df["Y_species_label"] == s
            ].reset_index(drop=True)
            media_row = media_features_df.loc[
                media_features_df["Y_media_label"] == m
            ].reset_index(drop=True)

            features.append(pd.concat([species_row, media_row], axis=1))

    features_df = pd.concat(features, axis=0).reset_index(drop=True)

    # Rename columns for features and labels
    # features_df.rename({"species_label": "Y_species_label"}, inplace=True, axis=1)

    # All other columns are features

    return features_df


def gapfill_from_proteome(gapfill_medias, media_db_path, proteome_path, output_path):
    if len(gapfill_medias) > 0:
        gapfill_command = (
            f"--gapfill {','.join(map(str, gapfill_medias))} --mediadb {media_db_path}"
        )

    else:
        gapfill_command = ""

    command = f"carve {gapfill_command} --fbc2 -o {output_path} {proteome_path}"
    output = subprocess.run(command, shell=True)


def gapfill_existing_model(gapfill_medias, media_db_path, model_path, output_path):
    #  Copy media file to new directory
    shutil.copyfile(f"{model_path}", f"{output_path}")
    if len(gapfill_medias) > 0:
        for m in gapfill_medias:
            command = f"gapfill -v {output_path} -m {m}  --mediadb {media_db_path} --fbc2 -o {output_path}"
            output = subprocess.run(command, shell=True)


@hydra.main(config_path="./configs/autogapfill/", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")
    mlflow.set_experiment(experiment_name=cfg.experiment_name)
    # mlflow.sklearn.autolog()

    experiment = mlflow.get_experiment_by_name(cfg.experiment_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id):

        # Load features
        model_features = pd.read_csv(cfg.model_features_path)
        media_features = pd.read_csv(cfg.media_features_path)

        # Load sklearn_model
        classifier_model = mlflow.pyfunc.load_model(f"{cfg.classifier_path}")

        # Load preprocessing pipeline
        local_path = mlflow.artifacts.download_artifacts(run_id=cfg.classifier_run_id)

        with open(f"{local_path}/preprocessing.pth", "rb") as handle:
            preprocessing_pipeline = torch.load(handle)

        # Split labels and features
        data_df = combine_media_model_data(media_features, model_features)

        for f in preprocessing_pipeline:

            data_df = f.transform(data_df)

        feature_columns = data_df.columns[data_df.columns.str.startswith("X_")]

        for model in glob(f"{cfg.models_dir}/*.xml"):
            species_name = model.split("/")[-1]
            species_name = species_name.split(".")[0]

            # Subset model for just our species
            model_features = data_df.loc[data_df["Y_species_label"] == species_name]
            media_labels = data_df["Y_media_label"]

            output = classifier_model.predict(model_features[feature_columns].values)

            gapfill_medias = []
            if (output.sum()) > 0:
                for idx, x in enumerate(output):
                    if x:
                        gapfill_medias.append(media_labels[idx])

            output_dir = "./gapfilled_models/"
            Path(f"{output_dir}").mkdir(parents=True, exist_ok=True)
            print(gapfill_medias)

            # gapfill_from_proteome(
            #     gapfill_medias=gapfill_medias,
            #     proteome_path=f"{cfg.proteomes_dir}/{species_name}/{species_name}.faa",
            #     media_db_path=f"{cfg.media_db}",
            #     output_path=f"{output_dir}/{species_name}.xml",
            # )

            gapfill_existing_model(
                gapfill_medias=gapfill_medias,
                model_path=f"{cfg.models_dir}/{species_name}.xml",
                media_db_path=f"{cfg.media_db}",
                output_path=f"{output_dir}/{species_name}.xml",
            )

        mlflow.log_artifact("./gapfilled_models/")


if __name__ == "__main__":
    main()
