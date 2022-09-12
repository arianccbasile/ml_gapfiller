import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms

import pandas as pd

import torch

import numpy as np

from loguru import logger


class ClassificationDataSet(Dataset):
    def __init__(self, df, label_columns, feature_columns):
        super().__init__(),
        self.data_df = df
        self.label_columns = label_columns
        self.feature_columns = feature_columns

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx):
        features = torch.from_numpy(
            self.data_df.iloc[idx][self.feature_columns].to_numpy(dtype=np.float32)
        )
        labels = torch.from_numpy(
            self.data_df.iloc[idx][self.label_columns].to_numpy(dtype=np.float32)
        )

        data = {"X": features, "y": labels}

        return data


class ClassificationDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        train_val_data_path: str,
        test_data_path: str,
        train_val_splitter: callable,
        feature_prefix: str,
        label_columns: list,
        autoencoder_mode: bool = False,
        preprocessing_pipeline: list = None,
        batch_size=8,
        num_workers=0,
    ):
        """
        Args:
            data_csv_path (string): path to csv file
            train_test_splitter (callable): function that splits data into train and test
            test_size (float, optional): Fraction of data to be used for test set.
            transform (callable, optional): Optional transform to be applied
        """
        super().__init__()

        self.train_val_data_path = train_val_data_path
        self.test_data_path = test_data_path
        self._BATCH_SIZE = batch_size
        self._NUM_WORKERS = num_workers

        self.transform = transforms.Compose(transforms)
        self.train_val_splitter = train_val_splitter

        self.label_columns = label_columns

        self.preprocessing_pipeline = preprocessing_pipeline

        self.train_val_df = pd.read_csv(self.train_val_data_path)
        self.test_df = pd.read_csv(self.test_data_path)

        # Split data into train and test
        self.train_df, self.val_df = self.train_val_splitter.split(self.train_val_df)

        if autoencoder_mode:
            self.label_columns = self.feature_columns

    def fit_preprocessing_pipeline(self):
        logger.info("Fitting preprocessing pipeline... ")

        train_data = self.train_df.copy()
        # Fit transform preprocessing pipeline
        for f in self.preprocessing_pipeline:
            train_data = f.fit_transform(train_data)

        self.feature_columns = train_data.columns[
            train_data.columns.str.startswith("X_")
        ]

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def get_input_size(self):
        return len(self.feature_columns)

    def get_output_size(self):
        return len(self.label_columns)

    def get_train_data_arrays(self):
        train_df = self.train_df.copy()

        # Transform validation and test datasets
        for f in self.preprocessing_pipeline:
            train_df = f.transform(train_df)

        X = train_df.loc[:, self.feature_columns].values
        y = train_df.loc[:, self.label_columns].values

        return X, y

    def get_val_data_arrays(self):

        val_df = self.val_df.copy()

        # Transform validation and test datasets
        for f in self.preprocessing_pipeline:
            val_df = f.transform(val_df)

        X = val_df.loc[:, self.feature_columns].values
        y = val_df.loc[:, self.label_columns].values

        return X, y

    def get_test_data_arrays(self):
        test_df = self.test_df.copy()

        # Transform validation and test datasets
        for f in self.preprocessing_pipeline:
            test_df = f.transform(test_df)

        X = test_df.loc[:, self.feature_columns].values
        y = test_df.loc[:, self.label_columns].values
        return X, y

    def train_dataloader(self):
        dataset = ClassificationDataSet(
            self.train_df, self.label_columns, self.feature_columns
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            shuffle=True,
        )

    def val_dataloader(self):
        dataset = ClassificationDataSet(
            self.val_df, self.label_columns, self.feature_columns
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=len(dataset),
            num_workers=self._NUM_WORKERS,
            shuffle=False,
        )

    def test_dataloader(self):
        dataset = ClassificationDataSet(
            self.test_df, self.label_columns, self.feature_columns
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=len(dataset),
            num_workers=self._NUM_WORKERS,
            shuffle=False,
        )

    def predict_dataloader(self):
        pass
