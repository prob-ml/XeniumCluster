# %%
import pandas as pd
import json
import os

import warnings 
warnings.filterwarnings("ignore")
from importlib import reload

# this ensures that I can update the class without losing my variables in my notebook
import xenium_cluster
reload(xenium_cluster)
from xenium_cluster import XeniumCluster
from utils.metrics import *

# %%
# Path to your .gz file
file_path = 'data/hBreast/transcripts.csv.gz'

# Read the gzipped CSV file into a DataFrame
df_transcripts = pd.read_csv(file_path, compression='gzip')

# drop cells without ids
df_transcripts = df_transcripts[df_transcripts["cell_id"] != -1]

# drop blanks and controls
df_transcripts = df_transcripts[~df_transcripts["feature_name"].str.startswith('BLANK_') & ~df_transcripts["feature_name"].str.startswith('NegControl')]

# %%
def run_experiment(data, dataset_name: str, current_spot_size: int, third_dim: bool, num_pcs: int, n_clusters=15):
    
    clustering = XeniumCluster(data=data, dataset_name=dataset_name)
    clustering.set_spot_size(current_spot_size)
    clustering.create_spot_data(third_dim=third_dim, save_data=True)

    print(f"The size of the spot data is {clustering.xenium_spot_data.shape}")

    clustering.normalize_counts(clustering.xenium_spot_data)
    clustering.generate_neighborhood_graph(clustering.xenium_spot_data, plot_pcas=False)

    BayesSpace_cluster = clustering.BayesSpace(clustering.xenium_spot_data, num_pcs=num_pcs, K=n_clusters)

    return clustering, BayesSpace_cluster

# %%
import numpy as np
import torch
from scipy.spatial.distance import cdist

def record_results(original_data, cluster_dict, results_dir, model_name, filename, spot_size, third_dim, num_pcs, K=None, resolution=None, uses_spatial=True):

    dirpath = f"{results_dir}/{model_name}/{num_pcs}/{(str(resolution) if resolution is not None else str(K))}/clusters/{spot_size}"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    wss = {}
    if resolution is not None:
        current_clustering = np.array(cluster_dict[model_name][spot_size][third_dim][num_pcs].get(
            resolution, 
            cluster_dict[model_name][spot_size][third_dim][num_pcs]
        ))
    else:
        current_clustering = np.array(cluster_dict[model_name][spot_size][third_dim][num_pcs][uses_spatial].get(
            K, 
            cluster_dict[model_name][spot_size][third_dim][num_pcs][uses_spatial]
        ))
    cluster_labels = np.unique(current_clustering)

    original_data.xenium_spot_data.obs[f"{model_name} cluster"] = np.array(current_clustering)
    # Extracting row, col, and cluster values from the dataframe
    rows = torch.tensor(original_data.xenium_spot_data.obs["row"].astype(int))
    cols = torch.tensor(original_data.xenium_spot_data.obs["col"].astype(int))
    clusters = torch.tensor(original_data.xenium_spot_data.obs[f"{model_name} cluster"].astype(int))
    cluster_labels = np.unique(clusters)
    num_clusters = len(cluster_labels)

    num_rows = int(max(rows) - min(rows) + 1)
    num_cols = int(max(cols) - min(cols) + 1)

    cluster_grid = torch.zeros((num_rows, num_cols), dtype=torch.int)

    cluster_grid[rows, cols] = torch.tensor(clusters, dtype=torch.int)

    for label in cluster_labels:
        current_cluster_locations = torch.stack(torch.where((cluster_grid == label)), axis=1).to(float)
        wss[f"Cluster {label}"] = (spot_size ** 2) * torch.mean(torch.cdist(current_cluster_locations, current_cluster_locations)).item()
        print(f"POSSIBLE {len(cluster_labels)}", label, wss[f"Cluster {label}"])

    wss_dirpath = f"{results_dir}/{model_name}/{num_pcs}/{(str(resolution) if resolution is not None else str(K))}/wss/{spot_size}/"
    if not os.path.exists(wss_dirpath):
        os.makedirs(wss_dirpath)

    wss_filepath = f"{wss_dirpath}/{filename}_wss.json"
    with open(wss_filepath, "w") as f:
        json.dump(wss, f, indent=4)

# %%
cluster_dict = {"BayesSpace": {}}
wss = {"BayesSpace": {}}
results_dir = "results/hBreast"

# %%
PC_list = [3, 5, 10, 15, 25]

# %%
import matplotlib
matplotlib.use('Agg')

for spot_size in [50, 75, 100]:
    for third_dim in [False]:
        for K in [17]:
            for num_pcs in PC_list:
                cluster_results_filename = f"clusters_K={K}"
                original_data, BayesSpace_cluster = run_experiment(df_transcripts, "hBreast", spot_size, third_dim, num_pcs, n_clusters=K)

                # BayesSpace
                if "BayesSpace" not in cluster_dict:
                    cluster_dict["BayesSpace"] = {}
                if spot_size not in cluster_dict["BayesSpace"]:
                    cluster_dict["BayesSpace"][spot_size] = {}
                if third_dim not in cluster_dict["BayesSpace"][spot_size]:
                    cluster_dict["BayesSpace"][spot_size][third_dim] = {}
                cluster_dict["BayesSpace"][spot_size][third_dim][num_pcs] = {True: {K: BayesSpace_cluster.tolist()}}
                record_results(original_data, cluster_dict, results_dir, "BayesSpace", cluster_results_filename, spot_size, third_dim, num_pcs, K, uses_spatial=True)

# %%
spot_sizes = [50, 75, 100]
in_billions = 1_000_000_000
method="BayesSpace"
for spot_size in spot_sizes:
    for K in [17]:
        filename = f"results/hBreast/{method}/{K}/wss/{spot_size}/{cluster_results_filename}_wss.json"
        if os.path.exists(filename):
            with open(filename, "r") as wss_dict:
                current_wss = json.load(wss_dict)
            print("Method:", method, "Spot Size", spot_size, "Num Clusters:", len(current_wss), "Total WSS", sum(current_wss.values()) / in_billions)

# %%



