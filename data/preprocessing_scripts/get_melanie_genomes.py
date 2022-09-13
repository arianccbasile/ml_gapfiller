import pandas as pd
import subprocess
from pathlib import Path
from glob import glob
import numpy as np
import shutil


def format_species_column(df, species_col="species"):
    df[species_col] = df[species_col].apply(lambda x: x.replace(" ", "_"))
    df[species_col] = df[species_col].apply(lambda x: x.replace(".", "_"))
    df[species_col] = df[species_col].apply(lambda x: x.replace("-", "_"))
    df[species_col] = df[species_col].apply(lambda x: x.replace("/", "_"))
    df[species_col] = df[species_col].apply(lambda x: x.replace("__", "_"))

    return df


def get_genomes(organism_summary, output_dir):
    # Set working directory
    Path(f"{output_dir}").mkdir(parents=True, exist_ok=True)

    has_genome = []

    for idx, row in organism_summary.iterrows():
        gca = row.assembly

        if isinstance(gca, str):
            if gca.startswith("NT"):
                has_genome.append(False)
                continue

            gca = gca.split("_")
            gca = "_".join([gca[0], gca[1]])

            output_name = row["designation in screen"]

            Path(f"{output_dir}/{output_name}/").mkdir(parents=True, exist_ok=True)
            command = f"ncbi-genome-download -A {gca} bacteria --formats fasta --section genbank -o {output_dir}/{output_name} --verbose --parallel 4"
            output = subprocess.run(command, shell=True)

        elif np.isnan(gca):
            has_genome.append(False)
            continue

        if output.returncode == 1:
            has_genome.append(False)

        else:
            has_genome.append(True)

        print(output_name, gca, has_genome)

    organism_summary["has_genome"] = has_genome
    return organism_summary


def make_proteomes(organism_summary, proteomes_dir, genomes_dir):
    sub_df = organism_summary.loc[organism_summary["has_genome"] == 1]

    for species in sub_df["designation in screen"].values:
        species_genome_dir = glob(f"{genomes_dir}/{species}/genbank/bacteria/**/")[0]

        # Unzip fasta
        species_tar = glob(f"{species_genome_dir}/*.gz")[0]
        command = f"gunzip {species_tar} --keep -f"
        output = subprocess.run(command, shell=True)

        species_fna = glob(f"{species_genome_dir}/*.fna")[0]

        # Run prodigal
        Path(f"{proteomes_dir}/{species}/").mkdir(parents=True, exist_ok=True)
        command = (
            f"prodigal -i {species_fna} -a {proteomes_dir}/{species}/{species}.faa"
        )
        output = subprocess.run(command, shell=True)


def move_uniprot_proteomes(organism_summary, proteomes_dir, uniprotproteomes_dir):
    sub_df = organism_summary.loc[organism_summary["has_genome"] == 0]
    uniprot_df = pd.read_csv(f"{uniprotproteomes_dir}/proteomics_species.csv")

    for idx, row in sub_df.iterrows():
        assembly = row.assembly
        species = row["designation in screen"]
        uniprot_row = uniprot_df.loc[uniprot_df["NT_code"] == assembly]

        if len(uniprot_row) == 0:
            print(species, assembly)
            continue
        
        elif isinstance(uniprot_row.Uniprot.values[0], str):
            pass

        elif np.isnan(uniprot_row.Uniprot.values[0]):
            continue

        Path(f"{proteomes_dir}/{species}/").mkdir(parents=True, exist_ok=True)
        uniprot_file = f"uniprot-proteome {uniprot_row.Uniprot.values[0]}.fasta"

        shutil.copy(
            f"{uniprotproteomes_dir}/{uniprot_file}", f"{proteomes_dir}/{species}/"
        )
        # species_genome_dir = glob(shutil.copy(item[0], "/Users/username/Desktop/testPhotos")f"{genomes_dir}/{species}/genbank/bacteria/**/")[0]


def make_gsmms(organism_summary, proteomes_dir, models_dir):
    has_proteome = []
    has_gsmm = []
    for species in organism_summary["designation in screen"].values:
        num_proteomes = glob(f"{proteomes_dir}/{species}/*.faa")
        if len(num_proteomes) == 0:
            has_proteome.append(False)
            

        else:
            has_proteome.append(True)

            command = f"carve -v --fbc2 -o {models_dir}/{species}.xml {proteomes_dir}/{species}/{species}.faa"

            output = subprocess.run(command, shell=True)
            if output.returncode == 1:
                has_gsmm.append(False)

            else:
                has_gsmm.append(True)
    
    organism_summary["has_gsmm"] = has_gsmm
    organism_summary["has_proteome"] = has_proteome
    return organism_summary


def main():
    wd = "/rds/user/bk445/hpc-work/ml_gapfiller"
    data_dir = f"{wd}/data/melanie_data"
    genomes_dir = f"{data_dir}/genomes/"
    proteomes_dir = f"{data_dir}/proteomes/"
    uniprotproteomes_dir = f"{data_dir}/uniprot_proteomes/"
    models_dir = f"{data_dir}/models/"

    raw_data_dir = f"{data_dir}/raw/"
    intermediate_data_dir = f"{data_dir}/intermediate/"

    # # # Load supp
    # organism_summary = pd.read_excel(
    #     f"{raw_data_dir}/supp_material.xlsx", sheet_name="S1. Selected gut bacteria"
    # )
    # format_species_column(organism_summary, species_col="designation in screen")

    # organism_summary = get_genomes(organism_summary, output_dir=genomes_dir)
    # organism_summary.to_csv(f"{data_dir}/organism_summary.csv")

    organism_summary = pd.read_csv(f"{data_dir}/organism_summary.csv", index_col=0)

    # make_proteomes(organism_summary, proteomes_dir, genomes_dir)
    move_uniprot_proteomes(organism_summary, proteomes_dir, uniprotproteomes_dir)
    organism_summary = make_gsmms(organism_summary, proteomes_dir, models_dir)
    organism_summary.to_csv(f"{data_dir}/organism_summary.csv")


if __name__ == "__main__":
    main()
