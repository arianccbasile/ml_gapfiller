from sklearn.model_selection import train_test_split
import numpy as np


class RandomSplitData:
    """
    Randomly split data into train and test sets.

    Args:
        df (pandas.DataFrame): DataFrame to split.
        test_size (float): Fraction of data to be used for test set.

    Returns:
        tuple: Tuple of train and test DataFrames.
    """

    def __init__(self, test_size):
        self.test_size = test_size

    def split(self, df):
        train, test = train_test_split(df, test_size=self.test_size, shuffle=True)

        return train, test


class SplitOnColumn:
    """
    Splits data according to a given column. Elements in this column
    will appear exclusively in training or test sets.

    Args:
        test_size (float): Fraction of data to be used for test set.
        column: column whose elements should be split between test and training data

    Returns:
        tuple: Tuple of train and test DataFrames.
    """

    def __init__(self, test_size, column):
        self.test_size = test_size
        self.column = column

    def split(self, df):
        unique_elements = df[self.column].unique()
        n_unique = len(unique_elements)

        n_test_elements = int(len(unique_elements) * self.test_size)

        test_compounds = np.random.choice(unique_elements, n_test_elements)

        test_df = df.loc[df[self.column].isin(test_compounds)]
        train_df = df.loc[~df[self.column].isin(test_compounds)]

        return train_df, test_df
