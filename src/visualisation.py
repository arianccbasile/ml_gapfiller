import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from ml_gapfill.gsmm_analysis import GSMMAnalysis

import plotly
import plotly.express as px
import plotly.graph_objects as go

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

import pickle
import plotly.figure_factory as ff


colours = [
    "#003f5c",
    "#58508d",
    "#bc5090",
    "#ff6361",
    "#ffa600",
    "#ffa800",
]


def plot_train_pedict_scatter(y_train, y_pred, output_path):
    fig, ax = plt.subplots()
    ax.scatter(y_train, y_pred)
    plt.xlabel("label")
    plt.ylabel("prediction")
    mlflow.log_figure(fig, output_path)
    plt.close()


def plot_prediction_label_scatter(model, data_df, label_column="Y_OD"):
    feature_names = data_df.columns.values[data_df.columns.str.startswith("X_")]

    X_data = data_df.loc[:, data_df.columns.str.startswith("X_")].values
    y_label = data_df.loc[:, label_column].values
    y_pred = model.predict(X_data)

    layout_min = np.min([np.min(y_label), np.min(y_pred)]) - 0.1
    layout_max = np.max([np.max(y_label), np.max(y_pred)]) + 0.1

    fig = go.Figure(
        data=go.Scatter(
            x=y_label,
            y=y_pred,
            marker={"color": colours[2]},
            mode="markers",
            opacity=0.75,
        ),
        layout_yaxis_range=[layout_min, layout_max],
        layout_xaxis_range=[layout_min, layout_max],
    )

    fig.update_layout(xaxis_title="Actual", yaxis_title="Predicted")
    fig.update_layout(template="simple_white", width=1000, height=1000)

    return fig


def plot_confusion_matrix(y_test, X_test, model, output_path):
    # Plot test confusion matrix
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        cmap=plt.cm.Blues,
    )

    mlflow.log_figure(disp.figure_, output_path)


def plotly_fig2array(fig):
    # convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="pdf")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


def plot_models_dendrogram(
    df,
    output_path,
    models_dir="/Users/bezk/Documents/CAM/research_code/community_study_mel_krp/analysis_daniel_machado/models/",
    dendro_figure_path="/Users/bezk/Documents/CAM/research_code/community_study_mel_krp/ml_project/scripts/preprocessing_scripts/dendrogram_figure.pkl",
    confusion=True,
):

    # Load figure template
    with open(dendro_figure_path, "rb") as f:
        fig = pickle.load(f)

    dendro_leaves = fig["layout"]["xaxis"]["ticktext"]

    # Make column to plot
    # nan=0, True negative=1, False positive=2, False negative=3, True positive=4
    def confusion_label_gen(row):
        if pd.isna(row["Y_pred"]):
            return 0

        # True negative
        elif row["Y_pred"] == 0 and row["Y_growth_class"] == 0:
            return 1

        # False positive
        elif row["Y_pred"] == 1 and row["Y_growth_class"] == 0:
            return 2

        # False negative
        elif row["Y_pred"] == 0 and row["Y_growth_class"] == 1:
            return 3

        # True positive
        elif row["Y_pred"] == 1 and row["Y_growth_class"] == 1:
            return 4

    def accuracy_label_gen(row):
        if pd.isna(row["Y_pred"]):
            print("isnan")
            return 0

        elif row["Y_pred"] == 0 and row["Y_growth_class"] == 0:
            return 2

        elif row["Y_pred"] == 1 and row["Y_growth_class"] == 1:
            return 2

        else:
            return 1

    if confusion:
        df["confusion_label"] = df.apply(confusion_label_gen, axis=1)

        df = df.pivot(
            index="Y_media_label", columns="Y_species_label", values="confusion_label"
        )
        colorscale = [
            [0.0, "rgb(0,0,0)"],
            [0.25, "rgb(9, 118, 148)"],
            [0.5, "rgb(209, 122, 23)"],
            [0.75, "rgb(148, 9, 39)"],
            [1.0, "rgb(9, 148, 44)"],
        ]

    else:
        df["accuracy_label"] = df.apply(accuracy_label_gen, axis=1)

        df = df.pivot(
            index="Y_media_label", columns="Y_species_label", values="accuracy_label"
        )
        colorscale = [
            [0.0, "rgb(0,0,0)"],
            [0.5, "rgb(148, 9, 39)"],
            [1.0, "rgb(9, 148, 44)"],
        ]

    # Make new column with media only digits
    df["media_only_digits"] = df.index.str.extract(r"(\d+)").astype(int).values

    # Sort by digit column
    df.sort_values(by=["media_only_digits"], inplace=True)

    # Drop column
    df.drop(columns=["media_only_digits"], inplace=True)

    # dendro_leaves = [d for d in dendro_leaves if d in df.columns]

    all_model_names = dendro_leaves
    for name in all_model_names:
        if name not in df.columns:
            df[name] = np.nan

    n_rows, n_cols = df.shape

    heatmap_vals = df[dendro_leaves].values

    heatmap_vals[pd.isna(heatmap_vals)] = 0
    # Set anything above zero to one
    # heatmap_vals[heatmap_vals > 0] = 1

    heatmap = [
        go.Heatmap(
            x=list(range(len(dendro_leaves))),
            y=list(range(n_rows)),
            z=heatmap_vals,
            colorscale=colorscale,
        )
    ]

    heatmap[0]["x"] = dendro_leaves
    # Add Heatmap Data to Figure
    for data in heatmap:
        fig.add_trace(data)

    # Edit xaxis
    fig.update_layout(
        xaxis={
            "domain": [0.15, 1],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "ticks": "",
        }
    )
    # Edit xaxis2
    fig.update_layout(
        xaxis2={
            "domain": [0, 0.15],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "showticklabels": False,
            "ticks": "",
        }
    )

    # Edit yaxis
    fig.update_layout(
        yaxis={
            "domain": [0, 0.85],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "showticklabels": True,
            "tickvals": [x - 0.25 for x in list(range(n_rows))],
            "ticktext": list(df.index),
        }
    )
    # Edit yaxis2
    fig.update_layout(
        yaxis2={
            "domain": [0.825, 0.975],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "showticklabels": False,
            "ticks": "",
        }
    )

    fig.update_layout(width=1000, height=1000, showlegend=False)
    mlflow.log_figure(fig, output_path)

    # fig.show()
    # data_dist = pdist(data_array)
    # heat_data = squareform(data_dist)
    # heat_data = heat_data[dendro_leaves,:]
    # heat_data = heat_data[:,dendro_leaves]

    return fig


def plot_models_dendrogram_v2(df, output_path, confusion=True):
    # Extract model features by subsetting for species and keeping constant columns
    unique_species = df.Y_species_label.unique()
    unique_medias = df.Y_media_label.unique()
    model_features = []

    for species in unique_species:
        sub_df = df.loc[df["Y_species_label"] == species]
        features_df = sub_df.loc[:, (sub_df == sub_df.iloc[0]).any()]

        select_columns = list(
            features_df.columns.values[features_df.columns.str.startswith("X_")]
        )

        features_df = sub_df[select_columns]
        features = features_df.values[0]
        model_features.append(features)

    model_features = np.array(model_features)

    fig = ff.create_dendrogram(
        model_features, orientation="bottom", labels=unique_species
    )

    fig.update_layout(width=800, height=800, showlegend=False)

    for i in range(len(fig["data"])):
        fig["data"][i]["yaxis"] = "y2"

    dendro_leaves = fig["layout"]["xaxis"]["ticktext"]

    # Make column to plot
    # nan=0, True negative=1, False positive=2, False negative=3, True positive=4
    def confusion_label_gen(row):
        if pd.isna(row["Y_pred"]):
            return 0

        # True negative
        elif row["Y_pred"] == 0 and row["Y_growth_class"] == 0:
            return 1

        # False positive
        elif row["Y_pred"] == 1 and row["Y_growth_class"] == 0:
            return 2

        # False negative
        elif row["Y_pred"] == 0 and row["Y_growth_class"] == 1:
            return 3

        # True positive
        elif row["Y_pred"] == 1 and row["Y_growth_class"] == 1:
            return 4

    def accuracy_label_gen(row):
        if pd.isna(row["Y_pred"]):
            print("isnan")
            return 0

        elif row["Y_pred"] == 0 and row["Y_growth_class"] == 0:
            return 2

        elif row["Y_pred"] == 1 and row["Y_growth_class"] == 1:
            return 2

        else:
            return 1

    df["accuracy_label"] = df.apply(accuracy_label_gen, axis=1)

    if confusion:
        df["confusion_label"] = df.apply(confusion_label_gen, axis=1)

        df = df.pivot(
            index="Y_media_label", columns="Y_species_label", values="confusion_label"
        )
        colorscale = [
            # [0.0, "rgb(0,0,0)"],
            [0.0, "rgb(9, 148, 44)"],
            [0.333, "rgb(255, 255, 102)"],
            [0.666, "rgb(255, 128, 39)"],
            [1.0, "rgb(9, 148, 44)"],
        ]

    else:
        df = df.pivot(
            index="Y_media_label", columns="Y_species_label", values="accuracy_label"
        )
        colorscale = [
            [0.0, "rgb(148, 9, 39)"],
            [1.0, "rgb(9, 148, 44)"],
        ]

    # Make new column with media only digits
    df["media_only_digits"] = df.index.str.extract(r"(\d+)").astype(int).values

    # Sort by digit column
    df.sort_values(by=["media_only_digits"], inplace=True)

    # Drop column
    df.drop(columns=["media_only_digits"], inplace=True)
    all_model_names = dendro_leaves
    for name in all_model_names:
        if name not in df.columns:
            df[name] = np.nan

    n_rows, n_cols = df.shape

    heatmap_vals = df[dendro_leaves].values

    heatmap_vals[pd.isna(heatmap_vals)] = 0
    # Set anything above zero to one
    # heatmap_vals[heatmap_vals > 0] = 1

    heatmap = [
        go.Heatmap(
            x=list(range(len(dendro_leaves))),
            y=list(range(n_rows)),
            z=heatmap_vals,
            colorscale=colorscale,
        )
    ]

    heatmap[0]["x"] = fig["layout"]["xaxis"]["tickvals"]
    # fig.update_layout(width=1500, height=1000, showlegend=False)

    # Add Heatmap Data to Figure
    for data in heatmap:
        fig.add_trace(data)

    # Edit xaxis
    fig.update_layout(
        xaxis={
            "domain": [0.15, 1],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "tickangle": -45,
            "ticks": "",
        }
    )
    # Edit xaxis2
    fig.update_layout(
        xaxis2={
            "domain": [0, 0.15],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "showticklabels": False,
            "tickangle": -45,
            "ticks": "",
        }
    )

    fig.update_xaxes(tickfont=dict(size=12))

    # Edit yaxis
    fig.update_layout(
        yaxis={
            "domain": [0, 0.85],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "showticklabels": True,
            "tickvals": [x - 0.25 for x in list(range(n_rows))],
            "ticktext": list(df.index),
        }
    )
    # Edit yaxis2
    fig.update_layout(
        yaxis2={
            "domain": [0.85, 0.975],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "showticklabels": False,
            "ticks": "",
        }
    )

    return fig


if __name__ == "__main__":
    wd = "/Users/bezk/Documents/CAM/research_code/community_study_mel_krp"
    models_dir = f"{wd}/analysis_daniel_machado/models/"
    df = pd.read_csv(
        "/Users/bezk/Documents/CAM/research_code/community_study_mel_krp/ml_project/mlruns/1/7fb276c3b78246c9871985d5a916483f/artifacts/test_df.csv"
    )

    plot_models_dendrogram(df, output_path="./test.html")
