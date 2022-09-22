import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path
import subprocess
from ml_gapfill.gsmm_analysis import GSMMAnalysis
import ast

def format_ph_column(ph_data):
    fmt_ph_min = []
    fmt_ph_max = []

    for row in ph_data:
        ph_min = np.nan
        ph_max = np.nan

        if type(row) == float:
            ph_min = row
            ph_max = row

        else:
            # Remove reference to compound used for buffer
            row = row.split(",")[0]
            row = row.split("-")
            if len(row) == 2:
                ph_min = float(row[0])
                ph_max = float(row[1])

            elif len(row) == 1:
                row = row[0]
                if "?" in row:
                    row = row.split("?")
                    ph_min = float(row[0])
                    ph_max = float(row[1])

                else:
                    ph_min = float(row)
                    ph_max = float(row)

            else:
                print(row)

        fmt_ph_min.append(ph_min)
        fmt_ph_max.append(ph_max)

    return fmt_ph_min, fmt_ph_max


def format_bool_column(column_data):
    fmt_bools = []

    for d in column_data:
        if d == "true":
            fmt_bools.append(True)

        elif d == "false":
            fmt_bools.append(False)

        else:
            print(d)
            fmt_bools.append(np.nan)

    return fmt_bools


def format_media_list(media_data):
    medias = []
    for m in media_data:

        if isinstance(m, str):
            m = m.split("|")

            # Remove trailing and leading whitespace
            m = [i.rstrip().lstrip() for i in m]

            # Replace spaces with underscores
            m = [i.replace(" ", "_") for i in m]

        elif np.isnan(m):
            m = m

        else:
            print(m)

        medias.append(m)

    return medias


def get_organisms_summary(media_summary_df):
    df = pd.read_html(
        "https://komodo.modelseed.org/servlet/KomodoTomcatServerSideUtilitiesModelSeed?OrganismMedia"
    )[1]
    df.rename(columns=df.iloc[0], inplace=True)
    df.drop(df.index[0], inplace=True)

    df["Media list"] = format_media_list(df["Media list"].values)

    # Drop organisms with nan media
    df.dropna(subset=["Media list"], how="all", inplace=True)

    # Drop organisms without a taxon
    df.dropna(subset=["Taxon ID"], how="all", inplace=True)

    # Filter out media definitions that are not in
    # the media summary
    keep_media_list = []
    for idx, org_row in df.iterrows():
        org_medias = org_row["Media list"]
        new_org_medias = []

        for m in org_medias:
            if m in media_summary_df.Name.values:
                sub_media_summary = media_summary_df.loc[media_summary_df['Name'] == m]
                new_org_medias.append(sub_media_summary['ID'].values[0])

        keep_media_list.append(new_org_medias)

    df["Media list"] = keep_media_list

    # Format organism names
    df["Organism Name"] = df["Organism Name"].apply(lambda x: x.replace(" ", "_"))
    df["Organism Name"] = df["Organism Name"].apply(lambda x: x.replace(".", ""))

    # drop rows with no media
    df = df.loc[df["Media list"].str.len() > 0].reset_index(drop=True)

    return df


def get_media_summary_dataframe(min_ph=5.0, max_ph=12.0, aerobic=True):
    """Generates a dataframe summarising the medias used in the KOMODO
    database. Performs so selection based on ph, and aerobic conditions

    Args:
        min_ph: _description_. Defaults to 5.0.
        max_ph: _description_. Defaults to 12.0.
        aerobic: _description_. Defaults to True.

    Returns:
        _description_
    """
    # Read in media list
    df = pd.read_html(
        "https://komodo.modelseed.org/servlet/KomodoTomcatServerSideUtilitiesModelSeed?MediaList"
    )[1]
    df.rename(columns=df.iloc[0], inplace=True)
    df.drop(df.index[0], inplace=True)

    df["Is Aerobic"] = format_bool_column(df["Is Aerobic"].values)
    df["Is Complex"] = format_bool_column(df["Is Complex"].values)
    df["Is SubMedium"] = format_bool_column(df["Is SubMedium"].values)

    print(df.shape)

    # Remove complex medias
    df = df.loc[df["Is Complex"] == False]
    print(df.shape)

    # Parse ph column and set min max ph buffers
    ph_min, ph_max = format_ph_column(df["PH Info"].values)
    df["ph_min"] = ph_min
    df["ph_max"] = ph_max

    # Drop columns with nan ph
    df.dropna(subset=["ph_min"], how="all", inplace=True)
    df.dropna(subset=["ph_max"], how="all", inplace=True)

    # Clean names
    df["Name"] = df["Name"].apply(lambda x: x.lstrip().rstrip())
    df["Name"] = df["Name"].apply(lambda x: x.replace(" ", "_"))

    # Apply selection criterias
    df = df.loc[df["ph_min"] >= min_ph]
    df = df.loc[df["ph_max"] <= max_ph]
    df = df.loc[df["Is Aerobic"] == aerobic]
    df = df.loc[df["Is SubMedium"] == False]

    return df


def get_media_definitions(media_name, seed_to_vmh_df):
    media_name = media_name.replace(" ", "%20")

    media_url = f"https://komodo.modelseed.org/servlet/KomodoTomcatServerSideUtilitiesModelSeed?MediaInfo={media_name}"

    # Last table is the SEED metabolites
    df = pd.read_html(media_url)[-1]
    df.rename(columns=df.iloc[0], inplace=True)
    df.drop(df.index[0], inplace=True)

    bigg_ids = []
    for seed_id in df["Seed ID"].values:
        row = seed_to_vmh_df.loc[seed_to_vmh_df["seed_id"] == seed_id]

        if len(row) == 0:
            bigg_ids.append(np.nan)

        else:
            bigg_ids.append(row["vmh_id"].values[0])

    df["bigg_ids"] = bigg_ids

    # Set null molar concentrations to 1000
    df["Mol Concentration"] = df["Mol Concentration"].astype(float)
    df["Mol Concentration"] = df["Mol Concentration"].apply(
        lambda x: 1000 if x == 0.0 else x
    )

    # Drop rows with unidentifiable bigg
    df.dropna(subset=["bigg_ids"], how="all", inplace=True)

    return df


def load_seed_to_vmh(path):
    seed_to_bigg_df = pd.read_csv(path, header=None, names=["seed_id", "vmh_id"])

    # Drop entries with a nan
    seed_to_bigg_df.dropna(subset=["seed_id"], how="all", inplace=True)
    seed_to_bigg_df.dropna(subset=["vmh_id"], how="all", inplace=True)

    # Clean up format
    seed_to_bigg_df["seed_id"] = seed_to_bigg_df["seed_id"].apply(
        lambda x: x.replace("(e)", "")
    )
    seed_to_bigg_df["seed_id"] = seed_to_bigg_df["seed_id"].apply(
        lambda x: x.replace("EX_", "")
    )
    seed_to_bigg_df["vmh_id"] = seed_to_bigg_df["vmh_id"].apply(
        lambda x: x.replace("EX_", "")
    )
    seed_to_bigg_df["vmh_id"] = seed_to_bigg_df["vmh_id"].apply(
        lambda x: x.replace("(e)", "")
    )
    seed_to_bigg_df["vmh_id"] = seed_to_bigg_df["vmh_id"].apply(
        lambda x: x.replace("_", "__")
    )

    return seed_to_bigg_df


def write_media_definitions(media_summary_df, output_dir):
    print(media_summary_df.columns)
    seed_to_bigg_df = load_seed_to_vmh("./SEED2VMH_translation.csv")

    for media_id in media_summary_df["ID"]:
        media_def = get_media_definitions(media_id, seed_to_bigg_df)
        media_def["media_label"] = media_id
        media_def.to_csv(f"{output_dir}/media_{media_id}.csv")
        print(f"{output_dir}/media_{media_id}.csv")


def build_media_features(media_summary_df, media_files_dir, output_dir):
    seed_to_bigg_df = load_seed_to_vmh("./SEED2VMH_translation.csv")

    media_file_paths = glob(f"{media_files_dir}/*.csv")

    # Get unique compounds
    unique_compounds = []
    for id in media_summary_df["ID"].values:
        df = pd.read_csv(f"{media_files_dir}/media_{id}.csv")
        unique_compounds.extend(df.bigg_ids.values)

    unique_compounds = list(set(unique_compounds))
    print(unique_compounds)

    # Build empty feature dict
    feature_dict = {"media_label": []}
    for c in unique_compounds:
        feature_dict[c] = []

    for id in media_summary_df["ID"].values:
        feature_dict["media_label"].append(f"KOMODO_{id}")
        df = pd.read_csv(f"{media_files_dir}/media_{id}.csv")

        for c in unique_compounds:
            media_compounds = df.bigg_ids.values
            if c in media_compounds:
                feature_dict[c].append(1)

            else:
                feature_dict[c].append(0)

    feature_dict = pd.DataFrame.from_dict(feature_dict)

    feature_dict.to_csv(f"{output_dir}/media_features.csv", index=False)

    return feature_dict


def get_organism_genomes(organism_summary, output_dir):
    print(organism_summary)

    unique_taxon = organism_summary["Taxon ID"].unique()

    has_genome = []

    for idx, row in organism_summary.iterrows():
        taxon_id = int(row["Taxon ID"])

        try:
            print(row["Organism Name"], taxon_id)
            Path(f'{output_dir}/{row["Organism Name"]}').mkdir(
                parents=True, exist_ok=True
            )

            command = f'ncbi-genome-download --taxids {taxon_id} bacteria --formats fasta -o {output_dir}/{row["Organism Name"]} --parallel 4'
            output = subprocess.run(command, shell=True)

            if output.returncode == 1:
                has_genome.append(False)

            else:
                has_genome.append(True)

        except KeyError:
            has_genome.append(False)

            print(taxon_id)
            print(type(taxon_id))
            print(row)

    organism_summary["has_genome"] = has_genome

    return organism_summary


def make_proteomes(organism_summary, proteomes_dir, genomes_dir):
    sub_df = organism_summary.loc[organism_summary["has_genome"] == 1]

    for species in sub_df["Organism Name"].values:
        print(species)
        species_genome_dir = glob(f"{genomes_dir}/{species}/refseq/bacteria/**/")[0]

        # Unzip fasta
        species_tar = glob(f"{species_genome_dir}/*.gz")
        if len(species_tar) > 0:
            command = f"gunzip {species_tar[0]}"
            output = subprocess.run(command, shell=True)
        
        species_fna = glob(f"{species_genome_dir}/*.fna")[0]
            
        # Run prodigal
        Path(f"{proteomes_dir}/{species}/").mkdir(parents=True, exist_ok=True)
        command = (
            f"prodigal -i {species_fna} -a {proteomes_dir}/{species}/{species}.faa"
        )
        output = subprocess.run(command, shell=True)


def make_gsmms(organism_summary, proteomes_dir, models_dir):
    sub_df = organism_summary.loc[organism_summary["has_genome"] == 1]

    for species in sub_df["Organism Name"].values:
        command = f"carve -v --fbc2 -o {models_dir}/{species}.xml {proteomes_dir}/{species}/{species}.faa"
        output = subprocess.run(command, shell=True)


def make_model_features(models_dir, output_dir):
    g = GSMMAnalysis(models_dir, debug=False)
    model_feat_df = g.create_feature_dataframe(output_dir)


def make_combined_features(
    media_features_path,
    model_features_path,
    organism_summary,
    media_summary,
    output_dir,
):
    media_features = pd.read_csv(media_features_path)
    model_features = pd.read_csv(model_features_path)

    # Rename columns for features and labels
    media_features.rename({'media_label': "Y_media_label"}, inplace=True, axis=1)
    model_features.rename({'species_label': "Y_species_label"}, inplace=True, axis=1)

    # All other columns are features
    media_features.rename(
        columns={
            i: f"X_MEDIA_{i}" for i in media_features.loc[:, ~media_features.columns.str.startswith("Y_")].columns
        },
        inplace=True,
    )
    # All other columns are features
    model_features.rename(
        columns={
            i: f"X_MODEL_{i}" for i in model_features.loc[:, ~model_features.columns.str.startswith("Y_")].columns
        },
        inplace=True,
    )

    model_features.to_csv(f'{output_dir}/model_features.csv')
    media_features.to_csv(f'{output_dir}/media_features.csv')

    # Make list of organism-media pairs
    organism = []
    media = []

    for idx, row in organism_summary.iterrows():
        if row['has_genome'] == False:
            continue

        organism_name = row['Organism Name']
        growth_medias = ast.literal_eval(row['Media list'])
        
        for m in growth_medias:
            sub_media_summary = media_summary.loc[media_summary['ID'] == m]

            media.append(f"KOMODO_{m}")            
            organism.append(organism_name)            

    # Combine rows of media and organism features
    features_list = []
    for m, o in zip(media, organism):
        media_row = media_features.loc[media_features['Y_media_label'] == m].reset_index(drop=True)
        org_row = model_features.loc[model_features['Y_species_label'] == o].reset_index(drop=True)
        
        features_list.append(pd.concat([media_row, org_row], axis=1))

    features_df = pd.concat(features_list, axis=0).reset_index(drop=True)

    # Add growth class column KOMODO database, all pairs are growing so == 1
    features_df['Y_growth_class'] = 1
    features_df.dropna(axis=0, inplace=True)
    features_df.reset_index(drop=True, inplace=True)

    features_df.to_csv(f"{output_dir}/labelled_komodo_df.csv", index=False)

def build_media_db_file(media_files_dir, output_dir):
    """Make Carveme compatible media database

    Args:
        media_files_dir: _description_
        output_dir: _description_
    """
    media_dfs = []
    media_file_paths = glob(media_files_dir + '/*.csv')

    for p in media_file_paths:
        print(p)
        df = pd.read_csv(p, index_col=0)
        media_dfs.append(df)

    media_db = pd.concat(media_dfs)
    
    media_db.rename({'media_label': "medium"}, inplace=True, axis=1)
    media_db.rename({'bigg_ids': "compound"}, inplace=True, axis=1)
    media_db.rename({'Name': "name"}, inplace=True, axis=1)
    media_db['description'] = media_db['medium']

    media_db = media_db[['medium', 'description', 'compound', 'name']]

    media_db.to_csv(f'{output_dir}/media_db.tsv', sep='\t', index=False)

def main():
    wd = "/rds/user/bk445/hpc-work/ml_gapfiller/data/komodo_scraper/"

    intermediate_dir = f"{wd}/intermediate_data/"
    processed_dir = f"{wd}/processed_data/"
    media_files_dir = f"{intermediate_dir}/media_files"
    genomes_dir = f"{intermediate_dir}/genomes/"
    proteomes_dir = f"{intermediate_dir}/proteomes/"
    models_dir = f"{intermediate_dir}/models/"

    Path(intermediate_dir).mkdir(parents=True, exist_ok=True)
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    Path(media_files_dir).mkdir(parents=True, exist_ok=True)
    Path(genomes_dir).mkdir(parents=True, exist_ok=True)
    Path(proteomes_dir).mkdir(parents=True, exist_ok=True)
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    media_summary_df = get_media_summary_dataframe(
        min_ph=6.5, max_ph=7.5, aerobic=False
    )

    media_summary_df.to_csv(f"{intermediate_dir}/media_summary.csv")
    media_summary_df = pd.read_csv(f"{intermediate_dir}/media_summary.csv")

    organism_df = get_organisms_summary(media_summary_df)
    organism_df.to_csv(f"{intermediate_dir}/organism_summary.csv")

    write_media_definitions(
        media_summary_df, output_dir=media_files_dir
    )

    build_media_db_file(media_files_dir, processed_dir)

    build_media_features(
        media_summary_df,
        media_files_dir=media_files_dir,
        output_dir=intermediate_dir
    )

    organism_df = pd.read_csv(f"{intermediate_dir}/organism_summary.csv")

    organism_df = get_organism_genomes(organism_df, genomes_dir)
    organism_df.to_csv(f"{intermediate_dir}/organism_summary.csv")
    organism_df = pd.read_csv(f"{intermediate_dir}/organism_summary.csv")

    make_proteomes(organism_df, proteomes_dir, genomes_dir)
    make_gsmms(organism_df, proteomes_dir, models_dir)

    organism_df = pd.read_csv(f"{intermediate_dir}/organism_summary.csv")

    make_model_features(models_dir, intermediate_dir)

    media_features_path = f"{intermediate_dir}/media_features.csv"
    model_features_path = f"{intermediate_dir}/model_features.csv"
    media_summary_df = pd.read_csv(f"{intermediate_dir}/media_summary.csv")

    make_combined_features(
        media_features_path,
        model_features_path,
        organism_df,
        media_summary_df,
        output_dir=processed_dir,
    )


if __name__ == "__main__":
    main()
