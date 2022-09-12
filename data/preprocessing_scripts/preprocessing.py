import numpy as np
import pandas as pd
import re

from ml_gapfill.gsmm_analysis import GSMMAnalysis


def generate_media_feature_matrix(media_path, output_dir):
    media_df = pd.read_csv(media_path, sep="\t")
    unique_medias = media_df.medium.unique()
    unique_compounds = media_df.compound.unique()

    features_dict = {"media_label": []}

    # Initialise lists
    for c in unique_compounds:
        features_dict[c] = []

    for m in unique_medias:
        sub_df = media_df.loc[media_df["medium"] == m]
        media_compounds = sub_df.compound.values

        features_dict["media_label"].append(m)

        for c in unique_compounds:
            if c in media_compounds:
                features_dict[c].append(1)

            else:
                features_dict[c].append(0)

    features_df = pd.DataFrame.from_dict(features_dict)

    features_df.to_csv(f"{output_dir}/media_features.csv", index=False)

    return features_df


def format_species_column(df, species_col="species"):
    df[species_col] = df[species_col].apply(lambda x: x.replace(" ", "_"))
    df[species_col] = df[species_col].apply(lambda x: x.replace(".", "_"))
    df[species_col] = df[species_col].apply(lambda x: x.replace("-", "_"))
    df[species_col] = df[species_col].apply(lambda x: x.replace("/", "_"))
    df[species_col] = df[species_col].apply(lambda x: x.replace("__", "_"))

    return df


def format_annotated_media_column(ann_df):
    unique_ann_medias = ann_df.media.unique()

    for m in unique_ann_medias:
        formatted_m = m

        if type(formatted_m) == str:
            if formatted_m == "15 A":
                formatted_m = "15A"

            elif formatted_m == "15 B":
                formatted_m = "15B"

        else:
            formatted_m = str(formatted_m)
            formatted_m = "M" + formatted_m

        # Remove white space
        formatted_m = re.sub(" +", " ", formatted_m)
        formatted_m = formatted_m.strip()

        ann_df.replace({"media": {m: formatted_m}}, inplace=True)

    return ann_df


def ann_data_formatting(ann_df):
    # Remove rows with measurement errors
    ann_df = ann_df.loc[ann_df["plate ID"].str.contains("Big_growth_curves")]

    ann_df = format_species_column(ann_df, species_col="species")
    ann_df = format_annotated_media_column(ann_df)
    return ann_df


def individiual_growth_df_formatting(individual_growth_df):
    individual_growth_df = format_species_column(
        individual_growth_df, species_col="species"
    )
    individual_growth_df["time"] = 48.0
    return individual_growth_df


def make_max_rep_col(df):
    df["max_rep"] = df.groupby("species")["rep"].transform(max)
    return df


def write_target_data(individual_growth_df, output_dir):
    # Make distance data for each media
    distances_df = pd.DataFrame()

    for species in individual_growth_df.species.unique():
        for medium in individual_growth_df.medium.unique():
            max_fold_change = -1

            sub_df = individual_growth_df.loc[individual_growth_df["medium"] == medium]
            sub_df = sub_df.loc[sub_df["species"] == species]

            for rep in sub_df.rep:
                rep_sub_df = sub_df.loc[sub_df["rep"] == rep]

                column_name = f"{species}_{medium}_{rep}_ODfc"
                max_rep_col_name = f"{species}_{medium}_ODfc"
                distances_df[column_name] = rep_sub_df.fold_change.values

                if rep_sub_df.fold_change.values[-1] > max_fold_change:
                    max_fold_change = rep_sub_df.fold_change.values[-1]
                    distances_df[max_rep_col_name] = rep_sub_df.fold_change.values

    distances_df.to_csv(f"{output_dir}/mel_indiv_target_data.csv")


def construct_feature_matrix(growth_df, model_feat_df, media_feat_df, output_dir):
    media_label_column = "media_label"
    species_label_column = "species_label"

    media_feat_df.rename(
        {media_label_column: f"Y_{media_label_column}"}, inplace=True, axis=1
    )
    model_feat_df.rename(
        {species_label_column: f"Y_{species_label_column}"}, inplace=True, axis=1
    )
    model_feat_df.rename(
        {"growth_class": "Y_fba_predicted_growth_class"}, inplace=True, axis=1
    )

    media_label_column = f"Y_{media_label_column}"
    species_label_column = "Y_species_label"

    # Rename with prefix
    media_feat_df.rename(
        columns={
            i: f"X_MEDIA_{i}"
            for i in media_feat_df.loc[
                :, ~media_feat_df.columns.str.startswith("Y_")
            ].columns
        },
        inplace=True,
    )

    # Rename with prefix
    model_feat_df.rename(
        columns={
            i: f"X_MODEL_{i}"
            for i in model_feat_df.loc[
                :, ~model_feat_df.columns.str.startswith("Y_")
            ].columns
        },
        inplace=True,
    )

    label_column = []
    output_dfs = []

    for idx, row in growth_df.iterrows():
        medium = row.medium

        # Skip mucin medias
        if medium == "M8" or medium == "M9":
            continue

        species = row.species

        label = row.OD
        if np.isnan(label):
            continue

        print(species)
        # Extract features
        print(model_feat_df.shape)
        sub_media_df = media_feat_df.loc[
            media_feat_df[media_label_column] == medium
        ].reset_index(drop=True)
        sub_model_df = model_feat_df.loc[
            model_feat_df[species_label_column] == species
        ].reset_index(drop=True)

        if sub_model_df.shape[0] == 0 or sub_media_df.shape[0] == 0:
            continue

        else:
            label_column.append(label)
            output_dfs.append(pd.concat([sub_media_df, sub_model_df], axis=1))

    df = pd.concat(output_dfs, axis=0).reset_index(drop=True)
    # df.drop([species_label_column, media_label_column], inplace=True, axis=1)

    df["Y_OD"] = label_column

    #  Make binary label column
    f = lambda row: int(1) if row["Y_OD"] > 0.1 else int(0)
    df["Y_growth_class"] = df.apply(f, axis=1)

    df.to_csv(f"{output_dir}/labelled_monoculture_df.csv", index=False)


def split_train_test_data(labelled_data_path, models_dir, output_dir, test_size=0.2):
    df = pd.read_csv(labelled_data_path)

    g = GSMMAnalysis(models_dir, debug=False)
    g.remove_non_unique_features()
    fig = g.create_dendrogram(orientation="bottom")

    dendro_leaves = fig["layout"]["xaxis"]["ticktext"]
    total_leaves = len(dendro_leaves)

    step = np.ceil(total_leaves / (test_size * total_leaves))
    test_leaf_indexes = np.arange(0, total_leaves, step=step)

    test_species = []
    for idx in test_leaf_indexes:
        test_species.append(dendro_leaves[int(idx)])

    test_df = df.loc[df["Y_species_label"].isin(test_species)].reset_index()
    train_df = df.loc[~df["Y_species_label"].isin(test_species)].reset_index()

    train_df.to_csv(f"{output_dir}/train_df.csv", index=False)
    test_df.to_csv(f"{output_dir}/test_df.csv", index=False)


def main():
    # Set working directory
    wd = "/Users/bezk/Documents/CAM/research_code/ml_informed_gapfill/"
    data_dir = f"{wd}/data/melanie_data//"
    raw_data_dir = f"{data_dir}/raw/"
    intermediate_data_dir = f"{data_dir}/intermediate/"
    processed_data_dir = f"{data_dir}/processed/"

    # Define paths to input files
    individual_growth_path = f"{raw_data_dir}/individual_growth.tsv"
    annotated_growth_path = f"{raw_data_dir}/supp_material.xlsx"

    models_dir = f"{data_dir}/melanie_screen_GEMs/models/no_gapfill_renamed"
    models_dir = "/Users/bezk/Documents/CAM/research_code/ml_informed_gapfill/data/melanie_data/models"
    media_path = f"{raw_data_dir}/melanie_data_media.tsv"

    ann_data = pd.read_excel(
        annotated_growth_path, sheet_name="S3. Annotated data", skiprows=2
    )
    individual_growth_df = pd.read_csv(individual_growth_path, sep="\t")

    # Apply formatting functions
    ann_df = ann_data_formatting(ann_data)
    individual_growth_df = individiual_growth_df_formatting(individual_growth_df)

    # Keep only the defined medias used in the anndata
    individual_growth_df = individual_growth_df.loc[
        individual_growth_df.medium.isin(ann_df.media.unique())
    ]
    unique_species = individual_growth_df.species.unique()

    individual_growth_df = individual_growth_df.loc[individual_growth_df.medium != "M9"]
    individual_growth_df = individual_growth_df.loc[individual_growth_df.medium != "M8"]
    individual_growth_df = individual_growth_df.loc[individual_growth_df.medium != "M4"]
    individual_growth_df = individual_growth_df.loc[
        individual_growth_df.medium != "M15A"
    ]
    individual_growth_df = individual_growth_df.loc[
        individual_growth_df.medium != "M15B"
    ]

    individual_growth_df.reset_index(inplace=True)

    individual_growth_df = (
        individual_growth_df.groupby(["species", "medium"])["OD"]
        .agg("max")
        .reset_index()
    )

    # Save intermediate data
    individual_growth_df.to_csv(
        f"{intermediate_data_dir}/formatted_individual_growth.csv", sep=","
    )

    # # # # Create model features data
    # g = GSMMAnalysis(models_dir, debug=False)
    # g.remove_non_unique_features()
    # model_feat_df = g.create_feature_dataframe(intermediate_data_dir)
    # media_feat_df = generate_media_feature_matrix(media_path, intermediate_data_dir)
    # model_feat_df = pd.read_csv(f"{intermediate_data_dir}/model_features.csv")

    # construct_feature_matrix(
    #     individual_growth_df, model_feat_df, media_feat_df, processed_data_dir
    # )

    split_train_test_data(
        processed_data_dir + "labelled_monoculture_df.csv",
        models_dir,
        output_dir=processed_data_dir,
        test_size=0.2,
    )


if __name__ == "__main__":
    main()
