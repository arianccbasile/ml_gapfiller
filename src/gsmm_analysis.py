import numpy as np
import pandas as pd
from glob import glob
import cobra
import operator
from pathlib import Path
from functools import reduce
import plotly.figure_factory as ff


class GSMMAnalysis:
    def __init__(self, models_dir, debug=False):
        self.models_dir = models_dir
        self.debug = debug

        self.generate_model_reactions_dict()
        self.generate_unique_reactions()
        self.generate_models_feature_matrix()

    def generate_model_reactions_dict(self):
        model_paths = glob(f"{self.models_dir}/*.xml")

        if self.debug:
            model_paths = model_paths[:3]

        model_reactions = {}
        for p in model_paths:
            model = self.load_model(p)
            model_name = Path(p).stem

            model_reactions[model_name] = self.get_model_reactions(model)
            print(len(model_reactions[model_name]))

        self.model_reactions = model_reactions

        return self.model_reactions

    def load_model(self, model_path):
        """
        Loads models from model paths
        """

        model = cobra.io.read_sbml_model(model_path)
        model.solver = "gurobi"
        return model

    def get_model_reactions(self, model):
        df = cobra.util.array.create_stoichiometric_matrix(
            model, array_type="DataFrame"
        )
        return list(df.columns)

    def generate_unique_reactions(self):
        all_reactions = []
        for x in self.model_reactions.keys():
            all_reactions.append(self.model_reactions[x])

        self.unique_reactions = sorted(
            list(reduce(lambda i, j: set(i) | set(j), all_reactions))
        )

    def generate_models_feature_matrix(self):
        feature_matrix = np.zeros(
            shape=[len(self.model_reactions), len(self.unique_reactions)]
        )

        for model_idx, model in enumerate(self.model_reactions.keys()):
            for reaction_idx, reaction in enumerate(self.unique_reactions):
                if reaction in self.model_reactions[model]:
                    feature_matrix[model_idx][reaction_idx] = 1.0

        self.models_feature_matrix = feature_matrix

    def remove_non_unique_features(self):
        n_rows, n_cols = self.models_feature_matrix.shape

        non_unique_columns = []
        unique_columns = []
        for c_idx in range(n_cols):
            if self.models_feature_matrix[:, c_idx].sum() == n_rows:
                non_unique_columns.append(c_idx)

            else:
                unique_columns.append(c_idx)

        self.unique_reactions = [self.unique_reactions[x] for x in unique_columns]
        self.models_feature_matrix = np.delete(
            self.models_feature_matrix, non_unique_columns, 1
        )

    def create_dendrogram(self, orientation="left"):

        fig = ff.create_dendrogram(
            self.models_feature_matrix,
            orientation=orientation,
            labels=list(self.model_reactions.keys()),
        )
        fig.update_layout(width=800, height=800)

        return fig

    def create_feature_dataframe(self, output_dir):
        df = pd.DataFrame(self.models_feature_matrix, columns=[self.unique_reactions])
        df["species_label"] = self.model_reactions.keys()

        df.to_csv(f"{output_dir}/model_features.csv", index=False)

        return df

    def create_model_media_viability_df(self, media_df, output_dir):
        model_paths = glob(f"{self.models_dir}/*.xml")

        output_dict = {"species": [], "media": [], "growth_class": []}

        for model_path in model_paths:
            model = self.load_model(model_path)
            model_name = Path(model_path).stem

            for media in media_df.medium.unique():
                sub_media_df = media_df.loc[media_df["medium"] == media]
                defined_mets = [
                    "EX_" + x + "_e" for x in sub_media_df["compound"].values
                ]

                for met in list(model.medium):
                    if met in defined_mets:
                        # print("Defined in media file and model media", met)
                        key = met.replace("EX_", "").replace("_e", "")

                        medium = model.medium
                        medium[met] = sub_media_df.loc[sub_media_df["compound"] == key][
                            "mmol_per_L"
                        ].values[0]

                        model.medium = medium

                    else:
                        medium = model.medium
                        medium[met] = 0.0
                        model.medium = medium

                growth_rate = model.slim_optimize()
                if growth_rate > 0.0:
                    growth_class = 1

                else:
                    growth_class = 0

                print(model_name, media, growth_class)
                output_dict["species"].append(model_name)
                output_dict["media"].append(media)
                output_dict["growth_class"].append(int(growth_class))

        model_growth_df = pd.DataFrame.from_dict(output_dict)
        model_growth_df.to_csv(
            f"{output_dir}/model_media_growth_class.csv", index=False
        )

        return model_growth_df


def main():
    wd = "/Users/bezk/Documents/CAM/research_code/community_study_mel_krp"
    models_dir = f"{wd}/analysis_daniel_machado/models"

    g = GSMMAnalysis(models_dir, debug=False)

    media_path = f"/Users/bezk/Documents/CAM/research_code/yeast_LAB_coculture/media/melanie_data_media.tsv"
    media_df = pd.read_csv(media_path, sep="\t")
    g.create_model_media_viability_df(media_df)

    g.remove_non_unique_features()
    g.create_dendrogram()
    g.create_feature_dataframe()

    # print(g.models_feature_matrix)
    # model_paths = glob(f"{models_dir}/*.xml")

    # # models = [load_model(p) for p in model_paths[:2]]

    # model_reactions = {}
    # for p in model_paths[:2]:
    #     model = load_model(p)
    #     model_name = Path(p).stem

    #     model_reactions[model_name] = get_model_reactions(model)
    #     print(len(model_reactions[model_name]))

    # unique_reactions = get_unique_reactions(model_reactions)
    # print(len(unique_reactions))


if __name__ == "__main__":
    main()
