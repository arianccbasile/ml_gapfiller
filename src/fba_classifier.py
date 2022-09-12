import cobra
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger


class FBAClassifier:
    def __init__(self, models_dir):
        self.models_dir = models_dir

    def set_media(self, model, media):
        medium = model.medium

        # Set all to zero
        for met in list(model.medium):
            if met in media.index:
                met = self.remove_prefix(met, prefix="X_MEDIA")
                media_val = media[met] * 1000
                medium[met] = media_val

            else:
                medium[met] = 0.0

        model.medium = medium

    def remove_prefix(self, str, prefix):
        if str.startswith(prefix):
            return f"EX_{str[len(prefix) :]}_e"
        else:
            return str

    def predict(self, df, compound_prefix_column, species_label_column):
        # outputs = np.array()

        media_compound_columns = df.columns[
            df.columns.str.startswith(compound_prefix_column)
        ]

        formatted_compound_columns = media_compound_columns[:]
        formatted_compound_columns = [
            self.remove_prefix(x, compound_prefix_column)
            for x in formatted_compound_columns
        ]

        # df = df.filter(items=media_compound_columns)

        df["index1"] = df.index

        species_dfs = []

        for species_idx, species in enumerate(df[species_label_column].unique()):
            species_df = df.loc[df[species_label_column] == species]
            model_path = f"{self.models_dir}/{species}.xml"
            p = Path(model_path)
            model = cobra.io.read_sbml_model(model_path, verbose=False)

            species_predict = []

            for idx, row in species_df.iterrows():
                row.index = [
                    self.remove_prefix(x, compound_prefix_column) for x in row.index
                ]

                self.set_media(model, media=row[formatted_compound_columns])

                growth_rate = model.optimize().objective_value
                if growth_rate > 0:
                    growth_rate = 1

                elif np.isnan(growth_rate):
                    growth_rate = 0

                elif growth_rate <= 0:
                    growth_rate = 0

                species_predict.append(growth_rate)

            logger.info(f"Model: {model}, Predict: {growth_rate}")

            species_df["Y_predict"] = species_predict
            species_dfs.append(species_df)

        df = pd.concat(species_dfs, axis=0)
        df.sort_values("index1", inplace=True)

        return df.Y_predict.values
