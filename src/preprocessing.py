class ColumnSelection:
    def __init__(
        self, target_Y_columns, feature_column_prefix="X_", drop_constant_columns=True
    ):
        """
        Slices dataset to keep only columns to be used in training/testing
        Args:
            target_Y (List): List of target columns to keep
            feature_column_prefix (str): prefix for feature columns
        """

        self.target_Y = target_Y_columns
        self.feature_column_prefix = feature_column_prefix
        self.drop_constant_columns = drop_constant_columns

    def fit(self, df):
        if self.drop_constant_columns:
            df = df.loc[:, (df != df.iloc[0]).any()]

        select_columns = list(
            df.columns.values[df.columns.str.startswith(self.feature_column_prefix)]
        )

        select_columns.extend(self.target_Y)

        self.select_columns = select_columns

    def transform(self, df):
        for c in self.select_columns:
            if c not in df.columns:
                df[c] = 0

        df = df[self.select_columns]

        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
