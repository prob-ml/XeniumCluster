# %%
import pandas as pd
import json
import os
import torch

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
def run_experiment(data, dataset_name: str, current_spot_size: int, third_dim: bool, init_method: str = "mclust", num_pcs: int = 15, n_clusters=15):
    
    clustering = XeniumCluster(data=data, dataset_name=dataset_name)
    clustering.set_spot_size(current_spot_size)
    clustering.create_spot_data(third_dim=third_dim, save_data=True)

    print(f"The size of the spot data is {clustering.xenium_spot_data.shape}")

    clustering.normalize_counts(clustering.xenium_spot_data)
    clustering.generate_neighborhood_graph(clustering.xenium_spot_data, plot_pcas=False)

    BayesSpace_cluster = clustering.BayesSpace(clustering.xenium_spot_data, init_method=init_method, num_pcs=num_pcs, K=n_clusters)

    return clustering, BayesSpace_cluster

# %%
import numpy as np
import torch
from scipy.spatial.distance import cdist

def record_results(original_data, cluster_dict, results_dir, model_name, filename, spot_size, third_dim, num_pcs, init_method, K=None, resolution=None, uses_spatial=True):

    dirpath = f"{results_dir}/{model_name}/{num_pcs}/{(str(resolution) if resolution is not None else str(K))}/clusters/{init_method}/{spot_size}"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    for gamma in np.linspace(1, 3, 9):
        gamma_str = f"{gamma:.2f}"
        try:
            current_clustering = pd.read_csv(f"{dirpath}/{gamma_str}/{filename}.csv", index_col=0)["BayesSpace cluster"].values

            original_data.xenium_spot_data.obs[f"{model_name} cluster"] = current_clustering
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
            
            wss = {}
            for label in cluster_labels:
                current_cluster_locations = torch.stack(torch.where((cluster_grid == label)), axis=1).to(float)
                wss[f"Cluster {label}"] = (spot_size ** 2) * torch.mean(torch.cdist(current_cluster_locations, current_cluster_locations)).item()
                print(f"POSSIBLE {len(cluster_labels)}", label, wss[f"Cluster {label}"])

            wss_dirpath = f"{results_dir}/{model_name}/{num_pcs}/{(str(resolution) if resolution is not None else str(K))}/wss/{init_method}/{spot_size}"
            if not os.path.exists(wss_dirpath):
                os.makedirs(wss_dirpath)

            wss_filepath = f"{wss_dirpath}/{gamma_str}/{filename}_wss.json"
            with open(wss_filepath, "w") as f:
                json.dump(wss, f, indent=4)

        except:
            continue

# %%
cluster_dict = {"BayesSpace": {}}
wss = {"BayesSpace": {}}
results_dir = "results/hBreast"

# %%
PC_list = [3, 5, 10, 15, 25]
init_methods = ["mclust", "kmeans"]

# %%
import matplotlib
matplotlib.use('Agg')

for spot_size in [50, 75, 100]:
    for third_dim in [False]:
        for K in [17]:
            for num_pcs in PC_list:
                for init_method in init_methods:
                    cluster_results_filename = f"clusters_K={K}"
                    original_data, BayesSpace_cluster = run_experiment(df_transcripts, "hBreast", spot_size, third_dim, init_method, num_pcs, n_clusters=K)

                    # BayesSpace
                    if "BayesSpace" not in cluster_dict:
                        cluster_dict["BayesSpace"] = {}
                    if spot_size not in cluster_dict["BayesSpace"]:
                        cluster_dict["BayesSpace"][spot_size] = {}
                    if third_dim not in cluster_dict["BayesSpace"][spot_size]:
                        cluster_dict["BayesSpace"][spot_size][third_dim] = {}
                    if init_method not in cluster_dict["BayesSpace"][spot_size][third_dim]:
                        cluster_dict["BayesSpace"][spot_size][third_dim][init_method] = {}
                    cluster_dict["BayesSpace"][spot_size][third_dim][init_method][num_pcs] = {True: {K: BayesSpace_cluster.tolist()}}
                    record_results(original_data, cluster_dict, results_dir, "BayesSpace", cluster_results_filename, spot_size, third_dim, num_pcs, init_method, K, uses_spatial=True)

# %%
spot_sizes = [50, 75, 100]
in_billions = 1_000_000_000
method="BayesSpace"
for spot_size in spot_sizes:
    for K in [17]:
        for init_method in init_methods:
            for num_pcs in PC_list:
                for gamma in np.linspace(1, 3, 9):
                    gamma_str = f"{gamma:.2f}"
                    cluster_results_filename = f"clusters_K={K}"
                    filename = f"results/hBreast/{method}/{num_pcs}/{K}/wss/{init_method}/{spot_size}/{gamma_str}/{cluster_results_filename}_wss.json"
                    if os.path.exists(filename):
                        with open(filename, "r") as wss_dict:
                            current_wss = json.load(wss_dict)
                        print("Method:", method, "Spot Size", spot_size, "Num Clusters:", len(current_wss), "Num PCs", num_pcs, "\u03B3", f": {gamma_str}", "Initial Method:", init_method, "Total WSS:", sum(current_wss.values()) / in_billions)

# %%



