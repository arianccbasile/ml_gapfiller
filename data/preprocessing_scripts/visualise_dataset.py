import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import webcolors
import datapane as dp
import re
from ml_gapfill.gsmm_analysis import GSMMAnalysis
import os
import pickle


def plot_models_dendrogram(growth_df_path, models_dir, discrete=False):
    df = pd.read_csv(growth_df_path)

    df = df.loc[df["medium"] != "M15A"]
    df = df.loc[df["medium"] != "M8"]
    df = df.loc[df["medium"] != "M9"]
    df = df.loc[df["medium"] != "M4"]

    df = df.groupby(["species", "medium"])["OD"].agg("max").reset_index()

    df = df.pivot(index="medium", columns="species", values="OD")

    # Make new column with media only digits
    df["media_only_digits"] = df.index.str.extract(r"(\d+)").astype(int).values

    # Sort by digit column
    df.sort_values(by=["media_only_digits"], inplace=True)

    # Drop column
    df.drop(columns=["media_only_digits"], inplace=True)
    df.reset_index()

    # Create dendrogram
    g = GSMMAnalysis(models_dir, debug=False)
    g.remove_non_unique_features()
    fig = g.create_dendrogram()

    # Initialize figure by creating upper dendrogram
    fig = g.create_dendrogram(orientation="bottom")
    for i in range(len(fig["data"])):
        fig["data"][i]["yaxis"] = "y2"

    # # Create Side Dendrogram
    # dendro_side = g.create_dendrogram()
    # for i in range(len(dendro_side['data'])):
    #     dendro_side['data'][i]['xaxis'] = 'x2'

    # Add Side Dendrogram Data to Figure
    # for data in dendro_side['data']:
    #     fig.add_trace(data)

    dendro_leaves = fig["layout"]["xaxis"]["ticktext"]

    n_rows, n_cols = df.shape

    heatmap_vals = df[dendro_leaves].values
    colorscale = "inferno"

    if discrete:
        # Set anything above zero to one
        heatmap_vals[heatmap_vals > 0] = 1
        colorscale = [[0.0, "rgb(0, 0, 0)"], [1.0, "rgb(9, 148, 44)"]]

    heatmap = [
        go.Heatmap(
            x=list(range(len(dendro_leaves))),
            y=list(range(n_rows)),
            z=heatmap_vals,
            colorscale=colorscale,
        )
    ]

    heatmap[0]["x"] = fig["layout"]["xaxis"]["tickvals"]
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

    fig.update_xaxes(tickfont=dict(size=9))

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

    fig.update_layout(width=1500, height=1000, showlegend=False)

    fig.show()
    # data_dist = pdist(data_array)
    # heat_data = squareform(data_dist)
    # heat_data = heat_data[dendro_leaves,:]
    # heat_data = heat_data[:,dendro_leaves]

    return fig, g


def plot_media_compositions(media_df_path):
    media_df = pd.read_csv(media_df_path, delimiter="\t")
    media = media_df.medium.unique()

    # Sort media by digits
    media = sorted(media, key=lambda x: int(re.sub("\D", "", x)), reverse=True)

    # Remove ignored medias
    media = [x for x in media if x not in ["M8", "M9", "M15A", "M15B", "M4"]]

    compounds = media_df.compound.unique()

    media_contents_arr = np.zeros([len(media), len(compounds)])

    for media_idx, m in enumerate(media):
        sub_df = media_df.loc[media_df["medium"] == m]

        media_compounds = sub_df.compound.unique()

        for cmpd_idx, cmpd in enumerate(compounds):
            if cmpd in media_compounds:
                media_contents_arr[media_idx][cmpd_idx] = 1

    fig = px.imshow(
        media_contents_arr,
        x=compounds,
        y=media,
        color_continuous_scale=[[0.0, "rgb(0,0,0)"], [1.0, "rgb(88, 80, 141)"]],
    )

    fig.update_layout(
        xaxis=dict(
            title="Compounds", tickmode="linear", tickfont=dict(size=7), tickangle=-45
        ),
        yaxis=dict(title="Media", tickmode="linear", tickfont=dict(size=7)),
        showlegend=False,
    )
    fig.update_traces(dict(showscale=False))
    fig.layout.coloraxis.showscale = False
    fig["layout"].update(margin=dict(l=10, r=10, b=10, t=10))

    return fig


def main():
    # Set working directory
    wd = "/Users/bezk/Documents/CAM/research_code/ml_informed_gapfill/"
    data_dir = f"{wd}/data/melanie_data/"
    raw_data_dir = f"{data_dir}/raw/"
    intermediate_data_dir = f"{data_dir}/intermediate/"
    processed_data_dir = f"{data_dir}/processed/"

    media_df_path = f"{raw_data_dir}/melanie_data_media.tsv"

    figure_output_dir = f"{data_dir}/figures/"

    # Make folder if not existing
    if not os.path.exists(figure_output_dir):
        os.makedirs(figure_output_dir)

    models_dir = f"{data_dir}/models/"

    # # Make dendrogram with final ODs
    fig_dendrogram, gsmm_obj = plot_models_dendrogram(
        f"{intermediate_data_dir}/formatted_individual_growth.csv",
        models_dir,
        discrete=False,
    )
    fig_dendrogram.write_image(f"{intermediate_data_dir}/fig_dendrogram_cont.pdf")

    # # Make discretised dendrogram for groweres/ non-growers
    fig_dendrogram, gsmm_obj = plot_models_dendrogram(
        f"{intermediate_data_dir}/formatted_individual_growth.csv",
        models_dir,
        discrete=True,
    )
    fig_dendrogram.write_image(f"{figure_output_dir}/fig_dendrogram_discrete.pdf")

    # # Pickle gsmm_obj
    with open(f"{figure_output_dir}/dendrogram_figure.pkl", "wb") as f:
        pickle.dump(fig_dendrogram, f)

    fig_media_composition = plot_media_compositions(media_df_path)
    fig_media_composition.write_image(f"{figure_output_dir}/fig_media_composition.pdf")

    # fig_mix_contents = d.plot_species_mixes()
    # fig_dom_species = d.plot_dominant_species_matrix()


if __name__ == "__main__":
    main()
