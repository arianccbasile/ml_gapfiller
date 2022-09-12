import pandas as pd
import numpy as np

def drop_constant_column(dataframe):
    """
    Drops constant value columns of pandas dataframe.
    """
    return dataframe.loc[:, (dataframe != dataframe.iloc[0]).any()]


def main():
    runs_data = './runs-4.csv'
    df = pd.read_csv(runs_data)
    df = drop_constant_column(df)

    print(df.columns)
    parameters = ['model.max_depth', 'model.n_estimators']

    stdev = lambda x: np.std(x)
    grp = df.groupby(parameters, as_index=False).agg({'acc_validation':['mean', 'std', 'min', 'max'],  'f1_validation' : ['mean', 'std', 'min', 'max', 'count']})
    # grp = grp.sort_values()
    grp = grp.sort_values([('f1_validation', 'mean')], ascending=False)

    print(grp)

    # new_columns = parameters + ['f1_val_mean', 'f1_val_std']
    # result.reindex(columns=sorted(result.columns)))


if __name__ == "__main__":
    main()