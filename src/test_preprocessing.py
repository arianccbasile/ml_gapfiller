import unittest
from ml_gapfill import preprocessing
import pandas as pd
import pandas.testing as pd_testing


class TestColumnSelection(unittest.TestCase):
    def test(self):
        data = {"X_a": [1, 2, 3], "X_b": [100, 200, 300], "Y_a": [5, 6, 7]}
        data_2 = {
            "X_a": [1, 2, 3],
            "X_b": [100, 200, 300],
            "Y_a": [5, 6, 7],
            "Y_b": [5, 6, 7],
        }

        df_1 = pd.DataFrame(data)
        df_2 = pd.DataFrame(data_2)

        func = preprocessing.ColumnSelection(
            target_Y_columns=["Y_a"],
            feature_column_prefix="X_",
            drop_constant_columns=True,
        )

        df_1 = func.fit_transform(df_1)
        df_2 = func.fit_transform(df_2)

        try:
            pd_testing.assert_frame_equal(df_1, df_2)
        except AssertionError as e:
            raise self.failureException("ColumnSelection failed") from e
