# %%
import json
import jsonlines
import os
import torch
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm

from importlib import reload
import xenium_cluster
reload(xenium_cluster)
from xenium_cluster import XeniumCluster

from scipy.sparse import csr_matrix

# %% [markdown]
# # Variables Setup

# %%
dataset_name = "hBreast"
models = ["BayXenSmooth", "Leiden", "Louvain", "K-Means", "K-Means_No_Spatial", "Hierarchical", "Hierarchical_No_Spatial", "BayesSpace"]
resolutions = [0.75]
spot_sizes = [100, 75, 50]
K_values = [17]

# %%
# BayXenSmooth Hyperparameters
BayXenSmooth_PCs = [3, 5, 10, 25]
BayesSpace_PCs = [3, 5, 10, 15, 25]
neighborhood_sizes = [1,2,3,4,5]
sample_for_assignment = False
concentration_amp = 1.0
spatial_norms = [0.05, 0.1]
aggs = ["sum", "mean", "weighted"]

# %% [markdown]
# # Load Data

# %%
# Path to your .gz file
file_path = f'data/{dataset_name}/transcripts.csv.gz'

# Read the gzipped CSV file into a DataFrame
df_transcripts = pd.read_csv(file_path, compression='gzip')

# drop cells without ids
df_transcripts = df_transcripts[df_transcripts["cell_id"] != -1]

# drop blanks and controls
df_transcripts = df_transcripts[~df_transcripts["feature_name"].str.startswith('BLANK_') & ~df_transcripts["feature_name"].str.startswith('NegControl')]

# %% [markdown]
# # Other Metric Implementations
# 
# - Variation Index (TODO) If we want to compare competing methods clustering with our clustering.
# 

# %%
def morans_i_cluster_similarity(clustering, locations, clusters):
    print("Starting Moran's I Calculation.")
    moran_clusters = ad.AnnData(locations)
    sc.pp.neighbors(moran_clusters, n_pcs=0, n_neighbors=100)
    print("Neighbors calculated.")

    cluster_labels = clusters.values
    # Calculate Moran's I for the binary presence of each cluster
    unique_clusters = np.unique(cluster_labels)
    morans_i_results = {}
    for cluster in unique_clusters:
        cluster_indicator = (cluster_labels == cluster).astype(int)
        morans_i = sc.metrics.morans_i(moran_clusters, vals=cluster_indicator)
        morans_i_results[cluster] = morans_i

    print("Done!")
    return np.mean(list(morans_i_results.values()))

# %%
def gearys_c_cluster_similarity(clustering, locations, clusters):
    print("Starting Gearys's C Calculation.")
    gearys_clusters = ad.AnnData(locations)
    sc.pp.neighbors(gearys_clusters, n_pcs=0, n_neighbors=100)
    print("Neighbors calculated.")

    cluster_labels = clusters.values
    # Calculate Gearys C for the binary presence of each cluster
    unique_clusters = np.unique(cluster_labels)
    gearys_c_results = {}
    for cluster in unique_clusters:
        cluster_indicator = (cluster_labels == cluster).astype(int)
        gearys_c = sc.metrics.gearys_c(gearys_clusters, vals=cluster_indicator)
        gearys_c_results[cluster] = gearys_c

    print("Done!")
    return np.mean(list(gearys_c_results.values()))

# %%
def save_results(results, directory, metric_name, specification=None):
    subdirectory = f"{specification}" if specification else ""
    full_path = f"{directory}/{subdirectory}"
    
    # Create the directory if it doesn't exist
    os.makedirs(full_path, exist_ok=True)
    
    with jsonlines.open(f"{full_path}/{metric_name}.jsonl", mode='w') as writer:
        try:
            for key, value in results.items():
                writer.write({key: value})
        except AttributeError: # b/c it's not a dictionary so .items() fails
            writer.write(results)

# %% [markdown]
# # Calculate the Silhouette Score (and other metrics of note.)

# %%
# for spot_size in spot_sizes:
#     clustering = XeniumCluster(data=df_transcripts, dataset_name=dataset_name)
#     clustering.set_spot_size(spot_size)
#     clustering.create_spot_data(third_dim=False, save_data=True)
#     locations = clustering.xenium_spot_data.obs[["row", "col"]]
#     for model in models:
#         for K in K_values:
#             if model in ["Leiden", "Louvain"]:
#                 for resolution in resolutions:
#                     clusters = pd.read_csv(f"results/{dataset_name}/{model}/{resolution}/clusters/{spot_size}/clusters_RES={resolution}.csv")[f"{model} cluster"]
#                     save_results(silhouette_score(locations, clusters), dataset_name, model, "silhouette_score", spot_size, resolution=resolution)
#                     save_results(morans_i_cluster_similarity(clustering, locations, clusters), dataset_name, model, "morans_i", spot_size, resolution=resolution)
#                     save_results(gearys_c_cluster_similarity(clustering, locations, clusters), dataset_name, model, "gearys_c", spot_size, resolution=resolution)
#             elif model == "BayXenSmooth":
#                 min_expressions_per_spot = 10
#                 clustering.xenium_spot_data = clustering.xenium_spot_data[clustering.xenium_spot_data.X.sum(axis=1) > min_expressions_per_spot]
#                 for neighborhood_size in neighborhood_sizes:
#                     clusters = pd.read_csv(f"results/{dataset_name}/{model}/clusters/PCA/{BayXenSmooth_PCs}/KMEANSINIT=True/NEIGHBORSIZE={neighborhood_size}/NUMCLUSTERS={K}/SPATIALINIT=True/SAMPLEFORASSIGNMENT={sample_for_assignment}/SPATIALNORM={spatial_norm}/SPATIALPRIORMULT={concentration_amp}/SPOTSIZE={spot_size}/AGG={agg}/clusters_K={K}.csv")[f"{model} cluster"]
#                     save_results(silhouette_score(locations, clusters), dataset_name, model, "silhouette_score", spot_size, K=K)
#                     save_results(morans_i_cluster_similarity(clustering, locations, clusters), dataset_name, model, "morans_i", spot_size, K=K, sample_for_assignment=sample_for_assignment)
#                     save_results(gearys_c_cluster_similarity(clustering, locations, clusters), dataset_name, model, "gearys_c", spot_size, K=K, sample_for_assignment=sample_for_assignment)
#             else:
#                 clusters = pd.read_csv(f"results/{dataset_name}/{model}/{K}/clusters/{spot_size}/clusters_K={K}.csv")[f"{model} cluster"]
#                 save_results(silhouette_score(locations, clusters), dataset_name, model, "silhouette_score", spot_size, K=K)
#                 save_results(morans_i_cluster_similarity(clustering, locations, clusters), dataset_name, model, "morans_i", spot_size, K=K)
#                 save_results(gearys_c_cluster_similarity(clustering, locations, clusters), dataset_name, model, "gearys_c", spot_size, K=K)

# %% [markdown]
# # Marker Gene Autocorrelation

# %%
MARKER_GENES = ["BANK1", "CEACAM6", "FASN", "FGL2", "IL7R", "KRT6B", "POSTN", "TCIM"]

# %%
def gene_morans_i(clustering, moran_clusters, clusters, num_neighbors=100, kernel='umap', p=1, marker_genes=MARKER_GENES, print_output=False):

    # Create a binary adjacency matrix indicating if points are in the same cluster
    cluster_labels = clusters.values
    same_cluster = (cluster_labels[:, None] == cluster_labels).astype(int)

    if kernel == 'umap':
        sc.pp.neighbors(moran_clusters, n_neighbors=num_neighbors, use_rep='X', n_pcs=0, method=kernel)
        moran_clusters.obsp["adjacency"] = moran_clusters.obsp["connectivities"].multiply(csr_matrix(same_cluster))
    elif kernel == 'gauss':
        sc.pp.neighbors(moran_clusters, n_neighbors=num_neighbors, use_rep='X', n_pcs=0, method=kernel)
        moran_clusters.obsp["adjacency"] = moran_clusters.obsp["connectivities"].multiply(csr_matrix(same_cluster))
    elif kernel == 'naive_distance':
        def naive_distance(x, p=1):
            return 1 / ((1 + x)**(1/p))

        nbrs = NearestNeighbors(n_neighbors=num_neighbors).fit(moran_clusters.X)
        distances, indices = nbrs.kneighbors(moran_clusters.X)
        connectivities = csr_matrix((moran_clusters.shape[0], moran_clusters.shape[0]))
        connectivities[np.arange(len(indices))[:, None], indices] = naive_distance(distances, p)
        moran_clusters.obsp["connectivities"] = connectivities
        moran_clusters.obsp["adjacency"] = moran_clusters.obsp["connectivities"].multiply(csr_matrix(same_cluster))
    elif kernel == 'basic':
        moran_clusters.obsp["adjacency"] = csr_matrix(same_cluster)
    else:
        warnings.warn(f"Kernel '{kernel}' not implemented. Using 'basic' kernel instead. We recommend 'umap'.", UserWarning)
        kernel = 'basic'
        moran_clusters.obsp["adjacency"] = csr_matrix(same_cluster)

    # Calculate Moran's I for the genes
    morans_i = sc.metrics.morans_i(moran_clusters.obsp["adjacency"], vals=clustering.xenium_spot_data.X.T)
    morans_i_dict = dict(zip(clustering.xenium_spot_data.var.index, [{"Moran's I": v} for v in morans_i]))

    # p-value Calculation
    N = moran_clusters.X.shape[0]
    S_0 = moran_clusters.obsp["adjacency"].sum()
    S_1 = 2 * (moran_clusters.obsp["adjacency"] ** 2).sum()
    S_2 = (4 * np.square(moran_clusters.obsp["adjacency"].sum(axis=0))).sum()

    E_I = -1/(N-1)
    VAR_I = ( (N**2)*(N-1)*S_1 - N*(N-1)*S_2 - 2*(S_0**2) ) / ( (N+1)*((N-1)**2)*(S_0**2) )

    for gene in morans_i_dict:
        morans_i_dict[gene]["p-val"] = norm.sf( (morans_i_dict[gene]["Moran's I"] - E_I)/ (VAR_I**0.5) )

    # Print the number of non-zero adjacencies
    if print_output:
        num_nonzero = moran_clusters.obsp["adjacency"].getnnz()
        print(f"Number of non-zero adjacencies: {num_nonzero}")
        for gene in marker_genes:
            print(num_neighbors, gene, morans_i_dict[gene])

    return morans_i_dict

# %%
def gene_gearys_c(clustering, gearys_clusters, clusters, num_neighbors=100):

    # Create a binary adjacency matrix indicating if points are in the same cluster
    cluster_labels = clusters.values
    same_cluster = (cluster_labels[:, None] == cluster_labels).astype(int)
    gearys_clusters.obsp["adjacency"] = gearys_clusters.obsp["connectivities"].multiply(csr_matrix(same_cluster))
    print("Connectivities formed.")

    # Calculate Geary's C for the genes
    gearys_c= sc.metrics.gearys_c(gearys_clusters.obsp["adjacency"], vals=clustering.xenium_spot_data.X.T)

    gearys_c_dict = dict(zip(clustering.xenium_spot_data.var.index, gearys_c))

    return gearys_c_dict

# %%
models = ["BayXenSmooth"]
num_neighboring_spots = [50, 100, 200, 400]
spot_sizes = [50]
kernels = ['umap']
K = 17

# %%
for spot_size in spot_sizes:
    clustering = XeniumCluster(data=df_transcripts, dataset_name=dataset_name)
    clustering.set_spot_size(spot_size)
    clustering.create_spot_data(third_dim=False, save_data=True)
    locations = clustering.xenium_spot_data.obs[["row", "col"]]
    moran_clusters = ad.AnnData(locations)
    gearys_clusters = ad.AnnData(locations)
    for spot_size in spot_sizes:
        for model in models:
 
            moran_clusters = ad.AnnData(locations)
            gearys_clusters = ad.AnnData(locations)

            # Define the directory where the results are stored
            results_dir = f"results/hBreast/{model}"

            # Loop through all subdirectories in the results directory
            for root, dirs, files in os.walk(results_dir):
                for file in files:
                    # Check if the file is named morans_i_by_gene.json
                    if file == f"clusters_K={K}.csv" and str(spot_size) in root:

                        for neighboring_spots in num_neighboring_spots:
                            for kernel in kernels:

                                clusters = pd.read_csv(os.path.join(root, file))[f"{model} cluster"]
                                save_results(gene_morans_i(clustering, moran_clusters, clusters, num_neighbors=neighboring_spots, kernel=kernel, print_output=True), root, "morans_i_by_gene", specification=f"{kernel}/{neighboring_spots}")