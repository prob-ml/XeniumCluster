import argparse
import os
import torch
import pyro
import json
import math
import time
from tqdm import tqdm
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.optim import PyroOptim
from pyro.optim import Adam
import pyro.distributions as dist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import get_cmap
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree
import seaborn as sns

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import lightning as L
import torch.nn.functional as F
from torch.utils.data import DataLoader

import subprocess
import warnings
warnings.filterwarnings("ignore")
from importlib import reload

# this ensures that I can update the class without losing my variables in my notebook
import xenium_cluster
reload(xenium_cluster)
from xenium_cluster import XeniumCluster
from utils.metrics import *

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import GPUtil

if torch.cuda.is_available():
    print("YAY! GPU available :3")
    
    # Get all available GPUs sorted by memory usage (lowest first)
    available_gpus = GPUtil.getAvailable(order='memory', limit=1)
    
    if available_gpus:
        selected_gpu = available_gpus[0]
        
        # Set the GPU with the lowest memory usage
        torch.cuda.set_device(selected_gpu)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        
        print(f"Using GPU: {selected_gpu} with the lowest memory usage.")
    else:
        print("No GPUs available with low memory usage.")
else:
    print("No GPU available :(")

def prepare_DLPFC_data(
    section_id=151670,
    num_pcs=5,
    log_normalize=True,
):
    section = ad.read_h5ad(f"data/DLPFC/{section_id}.h5ad")
    section.var["feature_name"] = section.var.index

    spatial_locations = section.obs[["array_row", "array_col"]]
    spatial_locations.columns = ["row", "col"]

    clustering = XeniumCluster(data=section.X, dataset_name="DLPFC")
    clustering.xenium_spot_data = section
    if log_normalize:
        clustering.xenium_spot_data.X = np.log1p(clustering.xenium_spot_data.X)

    sc.tl.pca(clustering.xenium_spot_data, svd_solver='arpack', n_comps=num_pcs)
    data = clustering.xenium_spot_data.obsm["X_pca"]

    return data, spatial_locations, clustering

def prepare_Xenium_data(
        dataset="hBreast", 
        spots=True, 
        spot_size=100, 
        third_dim=False, 
        log_normalize=True,
        likelihood_mode="PCA",
        num_pcs=5,
        hvg_var_prop=0.5,
        min_expressions_per_spot=10
    ):

    data_filepath = f"data/spot_data/{dataset}/hBreast_SPOTSIZE={spot_size}um_z={third_dim}.h5ad"
    
    if spots:

        if os.path.exists(data_filepath):

            clustering = XeniumCluster(data=None, dataset_name="hBreast")
            clustering.set_spot_size(spot_size)
            print("Loading data.")
            clustering.xenium_spot_data = ad.read_h5ad(data_filepath)

        else:

            # Path to your .gz file
            file_path = f'data/{dataset}/transcripts.csv.gz'

            # Read the gzipped CSV file into a DataFrame
            df_transcripts = pd.read_csv(file_path, compression='gzip')
            df_transcripts["error_prob"] = 10 ** (-df_transcripts["qv"]/10)
            df_transcripts.head(), df_transcripts.shape

            # drop cells without ids
            df_transcripts = df_transcripts[df_transcripts["cell_id"] != -1]

            # drop blanks and controls
            df_transcripts = df_transcripts[~df_transcripts["feature_name"].str.startswith('BLANK_') & ~df_transcripts["feature_name"].str.startswith('NegControl')]

            clustering = XeniumCluster(data=df_transcripts, dataset_name="hBreast")
            clustering.set_spot_size(spot_size)

            if not os.path.exists(data_filepath):
                print("Generating and saving data")
                clustering.create_spot_data(third_dim=third_dim, save_data=True)
                clustering.xenium_spot_data.write_h5ad(data_filepath)

        print("Number of spots: ", clustering.xenium_spot_data.shape[0])
        clustering.xenium_spot_data = clustering.xenium_spot_data[clustering.xenium_spot_data.X.sum(axis=1) > min_expressions_per_spot]
        print("Number of spots after filtering: ", clustering.xenium_spot_data.shape[0])

        if log_normalize:
            clustering.normalize_counts(clustering.xenium_spot_data)

        if likelihood_mode == "PCA":
            sc.tl.pca(clustering.xenium_spot_data, svd_solver='arpack', n_comps=num_pcs)
            data = clustering.xenium_spot_data.obsm["X_pca"]
        elif likelihood_mode == "HVG":
            min_dispersion = torch.distributions.normal.Normal(0.0, 1.0).icdf(hvg_var_prop)
            clustering.filter_only_high_variable_genes(clustering.xenium_spot_data, flavor="seurat", min_mean=0.0125, max_mean=1000, min_disp=min_dispersion)
            clustering.xenium_spot_data = clustering.xenium_spot_data[:,clustering.xenium_spot_data.var.highly_variable==True]
            data = clustering.xenium_spot_data.X
        elif likelihood_mode == "ALL":
            data = clustering.xenium_spot_data.X

        spatial_locations = clustering.xenium_spot_data.obs[["row", "col"]]
    
    # prepare cells data
    else:

        cells = df_transcripts.groupby(['cell_id', 'feature_name']).size().reset_index(name='count')
        cells_pivot = cells.pivot_table(index='cell_id', 
                                        columns='feature_name', 
                                        values='count', 
                                        fill_value=0)
        
        location_means = df_transcripts.groupby('cell_id').agg({
            'x_location': 'mean',
            'y_location': 'mean',
            'z_location': 'mean'
        }).reset_index()

        cells_pivot = location_means.join(cells_pivot, on='cell_id')

        if log_normalize:
            # log normalization
            cells_pivot.iloc[:, 4:] = np.log1p(cells_pivot.iloc[:, 4:])

        if likelihood_mode == "PCA":
            pca = PCA(n_components=num_pcs)
            data = pca.fit_transform(cells_pivot.iloc[:, 4:])

        elif likelihood_mode == "HVG":
            genes = cells_pivot.iloc[:, 4:]
            gene_variances = genes.var(axis=0)
            gene_variances = gene_variances.sort_values(ascending=False)
            gene_var_proportions = (gene_variances / sum(gene_variances))
            relevant_genes = list(gene_var_proportions[(gene_var_proportions.cumsum() < hvg_var_prop)].index)
            cells_pivot.iloc[:, 4:] = cells_pivot.iloc[:, 4:][[relevant_genes]]
            data = cells_pivot.iloc[:, 4:]

        elif likelihood_mode == "ALL":
            data = cells_pivot.iloc[:, 4:]

        spatial_locations = cells_pivot[["x_location", "y_location"]]

    # the last one is to regain var/obs access from original data
    return data, spatial_locations, clustering 

def Xenium_SVI(
        gene_data,
        spatial_locations,
        original_adata,
        spot_size = 100,
        data_mode="PCA",
        num_pcs=5,
        hvg_var_prop=0.5,
        dataset_name="hBreast",
        custom_init=False,
        num_clusters=6, 
        batch_size=512,
        neighborhood_size=2,
        neighborhood_agg="sum",
        uncertainty_values = [0.25, 0.5, 0.75, 0.9, 0.99],
        evaluate_markers=True, 
        num_posterior_samples=100,
        mu_prior_scale=1.0,
        sigma_prior_scale=1.0,
        logits_prior_scale=1.0,
        learn_global_variances=False,
        weighted_p=5,
    ):

    if torch.cuda.is_available():
        print("YAY! GPU available :3")
        
        # Get all available GPUs sorted by memory usage (lowest first)
        available_gpus = GPUtil.getAvailable(order='memory', limit=1)
        
        if available_gpus:
            selected_gpu = available_gpus[0]
            
            # Set the GPU with the lowest memory usage
            torch.cuda.set_device(selected_gpu)
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            
            print(f"Using GPU: {selected_gpu} with the lowest memory usage.")
        else:
            print("No GPUs available with low memory usage.")
    else:
        print("No GPU available :(")

        print(f"Batch Size is {batch_size}.")

    def custom_cluster_initialization(original_adata, method, K=17):

        original_adata.generate_neighborhood_graph(original_adata.xenium_spot_data, plot_pcas=False)

        # This function initializes clusters based on the specified method
        if method == "K-Means":
            initial_clusters = original_adata.KMeans(original_adata.xenium_spot_data, save_plot=False, K=K, include_spatial=False)
        elif method == "Hierarchical":
            initial_clusters = original_adata.Hierarchical(original_adata.xenium_spot_data, save_plot=True, num_clusters=K)
        elif method == "Leiden":
            initial_clusters = original_adata.Leiden(original_adata.xenium_spot_data, resolutions=[0.75], save_plot=False, K=K)[0.75]
        elif method == "Louvain":
            initial_clusters = original_adata.Louvain(original_adata.xenium_spot_data, resolutions=[1.0], save_plot=False, K=K)[1.0]
        elif method == "mclust":
            original_adata.pca(original_adata.xenium_spot_data, num_pcs)
            initial_clusters = original_adata.mclust(original_adata.xenium_spot_data, G=K, model_name = "EEE")
        else:
            raise ValueError(f"Unknown method: {method}")

        return initial_clusters
    
    def save_filepath(model, component, sample_for_assignment=None):

        total_file_path = (
            f"results/{dataset_name}/{model}/{component}/{data_file_path}/"
            f"INIT={custom_init}/NEIGHBORSIZE={neighborhood_size}/NUMCLUSTERS={num_clusters}/"
            f"/SAMPLEFORASSIGNMENT={sample_for_assignment}/"
            f"/SPATIALPRIORMULT=DIRECT/SPOTSIZE={spot_size}/AGG={neighborhood_agg}/"
            f"MU_PRIOR={mu_prior_scale}/SIGMA_PRIOR={sigma_prior_scale}/LOGITS_PRIOR={logits_prior_scale}/"
            f"LEARN_GLOBAL_VARS={learn_global_variances}"
        )

        return total_file_path

    pyro.clear_param_store()

    # Clamping
    MIN_CONCENTRATION = 0.001

    spatial_init_data = StandardScaler().fit_transform(gene_data)
    gene_data = StandardScaler().fit_transform(gene_data)
    empirical_prior_means = torch.ones(num_clusters, spatial_init_data.shape[1])
    empirical_prior_scales = torch.ones(num_clusters, spatial_init_data.shape[1])

    rows = spatial_locations["row"].astype(int)
    columns = spatial_locations["col"].astype(int)

    num_rows = max(rows) + 1
    num_cols = max(columns) + 1

    if custom_init:

        initial_clusters = custom_cluster_initialization(original_adata, custom_init, K=num_clusters)

        match data_mode:
            case "PCA":
                data_file_path = f"{data_mode}/{num_pcs}"
            case "HVG": 
                data_file_path = f"{data_mode}/{hvg_var_prop}"
            case "ALL":
                data_file_path = f"{data_mode}"
            case _:
                raise ValueError("The data mode specified is not supported.")
            

        if not os.path.exists(kmeans_clusters_filepath := save_filepath("KMeans", "clusters")):
            os.makedirs(kmeans_clusters_filepath)
        _ = plt.savefig(
            f"{kmeans_clusters_filepath}/result.png"
        )

        cluster_grid = torch.zeros((num_rows, num_cols), dtype=torch.int)
        
        cluster_grid[rows, columns] = torch.tensor(initial_clusters, dtype=torch.int) + 1

        colors = plt.cm.get_cmap('viridis', num_clusters + 1)
        colormap = ListedColormap(colors(np.linspace(0, 1, num_clusters + 1)))

        plt.figure(figsize=(6, 6))
        plt.imshow(cluster_grid.cpu(), cmap=colormap, interpolation='nearest', origin='lower')
        plt.colorbar(ticks=range(num_clusters + 1), label='Cluster Values')
        plt.title('Cluster Assignment with KMeans')

        if dataset_name == "DLPFC":
            # Create a DataFrame for easier handling
            data = pd.DataFrame({
                'ClusterAssignments': initial_clusters,
                'Region': original_adata.xenium_spot_data.obs["Region"]
            })

            # Drop rows where 'Region' is NaN
            filtered_data = data.dropna(subset=['Region'])

            # Calculate ARI and NMI only for the non-NaN entries
            ari = ARI(filtered_data['ClusterAssignments'], filtered_data['Region'])
            nmi = NMI(filtered_data['ClusterAssignments'], filtered_data['Region'])
            cluster_metrics = {
                "ARI": ari,
                "NMI": nmi
            }

            data_file_path = f"{data_mode}/{num_pcs}"

            if not os.path.exists(kmeans_cluster_metrics_filepath := save_filepath("KMeans", "cluster_metrics")):
                os.makedirs(kmeans_cluster_metrics_filepath)
            with open(f"{kmeans_cluster_metrics_filepath}/mpd.json", 'w') as fp:
                json.dump(cluster_metrics, fp)

        for i in range(num_clusters):
            cluster_data = gene_data[initial_clusters == i]
            if cluster_data.size > 0:  # Check if there are any elements in the cluster_data
                empirical_prior_means[i] = torch.tensor(cluster_data.mean(axis=0))
                empirical_prior_scales[i] = torch.tensor(cluster_data.std(axis=0))
        cluster_probs_prior = torch.zeros((initial_clusters.shape[0], num_clusters))
        cluster_probs_prior[torch.arange(initial_clusters.shape[0]), initial_clusters - 1] = 1.

    else:

        cluster_probs_prior = torch.ones((len(gene_data), num_clusters), dtype=float)

    locations_tensor = torch.as_tensor(spatial_locations.values, dtype=torch.float16, device='cuda')

    # Compute the number of elements in each dimension
    num_spots = cluster_probs_prior.shape[0]

    # Initialize an empty tensor for spatial cluster probabilities
    spatial_cluster_probs_prior = torch.zeros_like(cluster_probs_prior, dtype=torch.float64)

    spot_locations = KDTree(locations_tensor.cpu())  # Ensure this tensor is in host memory
    neighboring_spot_indexes = spot_locations.query_ball_point(locations_tensor.cpu(), r=neighborhood_size, p=1, workers=8)

    # Iterate over each spot
    for i in tqdm(range(num_spots)):

        # Select priors in the neighborhood
        priors_in_neighborhood = cluster_probs_prior[neighboring_spot_indexes[i]]
        # print(f"Spot {i} has {len(neighboring_spot_indexes[i])} neighbors")
        # print(priors_in_neighborhood)

        # Compute the sum or mean, or apply a custom weighting function
        if neighborhood_agg == "sum":
            neighborhood_priors = priors_in_neighborhood.sum(dim=0)
        elif neighborhood_agg == "mean":
            neighborhood_priors = priors_in_neighborhood.mean(dim=0)
        else:
            locations = original_adata.xenium_spot_data.obs[["x_location", "y_location", "z_location"]].values
            neighboring_locations = locations[neighboring_spot_indexes[i]].astype(float)
            distances = torch.tensor(np.linalg.norm(neighboring_locations - locations[i], axis=1))
            def distance_weighting(x, p=weighted_p):
                weight = 1/(1 + x/spot_size) ** (1/weighted_p)
                # print(weight)
                return weight / weight.sum()
            neighborhood_priors = (priors_in_neighborhood * distance_weighting(distances).reshape(-1, 1)).sum(dim=0)
        # Update the cluster probabilities
        # print(neighborhood_expression)
        spatial_cluster_probs_prior[i] += neighborhood_priors
    
    spatial_cluster_probs_prior = spatial_cluster_probs_prior.clamp(MIN_CONCENTRATION)
    sample_for_assignment_options = [False, True]

    num_prior_samples = num_posterior_samples
    for sample_for_assignment in sample_for_assignment_options:

        if sample_for_assignment:
            cluster_assignments_prior_TRUE = pyro.sample("cluster_assignments", dist.Categorical(spatial_cluster_probs_prior).expand_by([num_prior_samples])).mode(dim=0).values
            cluster_assignments_prior = cluster_assignments_prior_TRUE
        else:
            cluster_assignments_prior_FALSE = spatial_cluster_probs_prior.argmax(dim=1)
            cluster_assignments_prior = cluster_assignments_prior_FALSE
        # Load the data
        data = torch.tensor(gene_data).float()

        cluster_grid = torch.zeros((num_rows, num_cols), dtype=torch.long)

        cluster_grid[rows, columns] = cluster_assignments_prior + 1

        colors = plt.cm.get_cmap('viridis', num_clusters + 1)
        
        colormap_colors = np.vstack(([[1, 1, 1, 1]], colors(np.linspace(0, 1, num_clusters))))
        colormap = ListedColormap(colormap_colors)

        plt.figure(figsize=(6, 6))
        plt.imshow(cluster_grid.cpu(), cmap=colormap, interpolation='nearest', origin='lower')
        plt.colorbar(ticks=range(num_clusters + 1), label='Cluster Values')
        plt.title('Prior Cluster Assignment with BayXenSmooth')

        if not os.path.exists(bayxensmooth_clusters_filepath := save_filepath("BayXenSmooth", "clusters", sample_for_assignment)):
            os.makedirs(bayxensmooth_clusters_filepath)
        _ = plt.savefig(
            f"{bayxensmooth_clusters_filepath}/prior_result.png"
        )

    NUM_PARTICLES = 25
    expected_total_param_dim = 2 # K x D

    def model(data):

        with pyro.plate("clusters", num_clusters):

            # Define the means and variances of the Gaussian components
            cluster_means = pyro.sample("cluster_means", dist.Normal(empirical_prior_means, mu_prior_scale).to_event(1))
            cluster_scales = pyro.sample("cluster_scales", dist.LogNormal(empirical_prior_scales, sigma_prior_scale).to_event(1))

        # Define priors for the cluster assignment probabilities and Gaussian parameters
        with pyro.plate("data", len(data), subsample_size=batch_size) as ind:
            batch_data = data[ind]
            mu = torch.log(spatial_cluster_probs_prior[ind])
            cov_matrix = torch.eye(mu.shape[1], dtype=mu.dtype, device=mu.device) * logits_prior_scale
            cluster_probs_logits = pyro.sample("cluster_logits", dist.MultivariateNormal(mu, cov_matrix))
            cluster_probs = torch.softmax(cluster_probs_logits, dim=-1)
            # likelihood for batch
            if cluster_means.dim() == expected_total_param_dim:
                pyro.sample(f"obs", dist.MixtureOfDiagNormals(
                        cluster_means.unsqueeze(0).expand(batch_size, -1, -1), 
                        cluster_scales.unsqueeze(0).expand(batch_size, -1, -1), +
                        cluster_probs
                    ), 
                    obs=batch_data
                )
            # likelihood for batch WITH vectorization of particles
            else:
                pyro.sample(f"obs", dist.MixtureOfDiagNormals(
                        cluster_means.unsqueeze(1).expand(-1, batch_size, -1, -1), 
                        cluster_scales.unsqueeze(1).expand(-1, batch_size, -1, -1), 
                        cluster_probs
                    ), 
                    obs=batch_data
                )

    def guide(data):
        # Initialize cluster assignment probabilities for the entire dataset
        cluster_probs_logits_q_mean = pyro.param("cluster_logits_q_mean", torch.log(spatial_cluster_probs_prior) + torch.randn_like(spatial_cluster_probs_prior) * 0.1)
        cluster_probs_logits_q_scale = pyro.param("cluster_logits_q_scale", torch.ones_like(spatial_cluster_probs_prior, dtype=spatial_cluster_probs_prior.dtype, device=spatial_cluster_probs_prior.device) * logits_prior_scale, dist.constraints.positive)

        with pyro.plate("clusters", num_clusters):
            # Global variational parameters for means and scales
            cluster_means_q_mean = pyro.param("cluster_means_q_mean", empirical_prior_means + torch.randn_like(empirical_prior_means) * 0.05)
            cluster_scales_q_mean = pyro.param("cluster_scales_q_mean", empirical_prior_scales + torch.randn_like(empirical_prior_scales) * 0.01, constraint=dist.constraints.positive)
            if learn_global_variances:
                cluster_means_q_scale = pyro.param("cluster_means_q_scale", torch.ones_like(empirical_prior_means) * mu_prior_scale, constraint=dist.constraints.positive)
                cluster_scales_q_scale = pyro.param("cluster_scales_q_scale", torch.ones_like(empirical_prior_scales) * sigma_prior_scale, constraint=dist.constraints.positive)
                cluster_means = pyro.sample("cluster_means", dist.Normal(cluster_means_q_mean, cluster_means_q_scale).to_event(1))
                cluster_scales = pyro.sample("cluster_scales", dist.LogNormal(cluster_scales_q_mean, cluster_scales_q_scale).to_event(1))
            else:
                cluster_means = pyro.sample("cluster_means", dist.Delta(cluster_means_q_mean).to_event(1))
                cluster_scales = pyro.sample("cluster_scales", dist.Delta(cluster_scales_q_mean).to_event(1))

        with pyro.plate("data", len(data), subsample_size=batch_size) as ind:

            batch_probs_logits_q_mean = cluster_probs_logits_q_mean[ind]
            batch_probs_logits_q_scale = cluster_probs_logits_q_scale[ind]
            logits = pyro.sample("cluster_logits", dist.Normal(batch_probs_logits_q_mean, batch_probs_logits_q_scale).to_event(1))
            cluster_probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities

    NUM_EPOCHS = 500
    NUM_BATCHES = int(math.ceil(data.shape[0] / batch_size))
    # Setup the optimizer
    def per_param_callable(param_name):
        if param_name == 'cluster_means_q_mean':
            return {"lr": 0.001, "betas": (0.9, 0.999)}
        elif param_name == 'cluster_scales_q_mean':
            return {"lr": 0.001, "betas": (0.9, 0.999)}
        else:
            return {"lr": 0.005, "betas": (0.9, 0.999)}

    scheduler = Adam(per_param_callable)

    # Setup the inference algorithm
    svi = SVI(model, guide, scheduler, loss=TraceMeanField_ELBO(num_particles=NUM_PARTICLES, vectorize_particles=True))

    # Create a DataLoader for the data
    # Convert data to CUDA tensors before creating the DataLoader
    data = data.to('cuda')

    # Clear the param store in case we're in a REPL
    pyro.clear_param_store()

    epoch_pbar = tqdm(range(NUM_EPOCHS))
    current_min_loss = float('inf')
    PATIENCE = 10
    patience_counter = 0
    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch}")
        running_loss = 0.0
        for step in range(NUM_BATCHES):
            loss = svi.step(data)
            running_loss += loss / batch_size
            # running_loss += (loss + SPATIAL_PENALTY_WEIGHT * spatial_penalty()) / batch_size
        # svi.optim.step()
        if epoch % 1 == 0:
            # print(f"Epoch {epoch} : loss = {round(running_loss, 4)}")
            # print(current_cluster_means[0])
            if running_loss > current_min_loss:
                patience_counter += 1
            else:
                current_min_loss = running_loss
                patience_counter = 0
            if patience_counter >= PATIENCE:
                break 
            # clusters = pd.DataFrame(cluster_assignments_q.cpu(), columns=["BayXenSmooth cluster"])
            # morans_i_gene_dict = gene_morans_i(original_adata, spatial_locations, clusters["BayXenSmooth cluster"])
            # # gearys_c_gene_dict = gene_gearys_c(original_adata, spatial_locations, clusters["BayXenSmooth cluster"])
            # marker_genes = ["BANK1", "CEACAM6", "FASN", "FGL2", "IL7R", "KRT6B", "POSTN", "TCIM"]
            # morans_i_markers = {k: v for k, v in morans_i_gene_dict.items() if k in marker_genes}
            # # gearys_c_markers = {k: v for k, v in gearys_c_gene_dict.items() if k in marker_genes}
            # print(morans_i_markers)

            if dataset_name == "DLPFC":
                # Create a DataFrame for easier handling
                cluster_data = pd.DataFrame({
                    'ClusterAssignments': cluster_assignments_q,
                    'Region': original_adata.xenium_spot_data.obs["Region"]
                })

                # Drop rows where 'Region' is NaN
                filtered_data = cluster_data.dropna(subset=['Region'])

                # Calculate ARI and NMI only for the non-NaN entries
                ari = ARI(filtered_data['ClusterAssignments'], filtered_data['Region'])
                nmi = NMI(filtered_data['ClusterAssignments'], filtered_data['Region'])
                print(f"Step {step} : ARI = {ari} NMI = {nmi}")

    torch.set_default_tensor_type(torch.FloatTensor)

    # Grab the learned variational parameters
    sample_for_assignment_options = [True, False]

    for sample_for_assignment in sample_for_assignment_options:
        cluster_logits_q_mean = pyro.param("cluster_logits_q_mean")
        cluster_logits_q_scale = pyro.param("cluster_logits_q_scale")
        if sample_for_assignment:
            cluster_probs_q = torch.softmax(pyro.sample("cluster_probs", dist.Normal(cluster_logits_q_mean, cluster_logits_q_scale).expand_by([num_posterior_samples]).to_event(1)).mean(dim=0), dim=-1)
            cluster_assignments_q = pyro.sample("cluster_assignments", dist.Categorical(cluster_probs_q).expand_by([num_posterior_samples])).mode(dim=0).values
            cluster_assignments_prior = cluster_assignments_prior_TRUE
        else:
            cluster_probs_q = torch.softmax(cluster_logits_q_mean, dim=-1)
            cluster_assignments_q = cluster_probs_q.argmax(dim=1)
            cluster_assignments_prior = cluster_assignments_prior_FALSE
        
        cluster_means_q_mean = pyro.param("cluster_means_q_mean").cpu().detach()
        cluster_scales_q_mean = pyro.param("cluster_scales_q_mean").cpu().detach()
        cluster_probs_q = cluster_probs_q.cpu().detach()
        cluster_assignments_q = cluster_assignments_q.cpu().detach()
        cluster_assignments_prior = cluster_assignments_prior.cpu().detach()

        # Plotting
        if spot_size:

            rows = spatial_locations["row"].astype(int)
            columns = spatial_locations["col"].astype(int)

            num_rows = max(rows) + 1
            num_cols = max(columns) + 1

            cluster_grid = torch.zeros((num_rows, num_cols), dtype=torch.long)

            cluster_grid[rows, columns] = cluster_assignments_q + 1

            colors = plt.cm.get_cmap('viridis', num_clusters + 1)

            colormap_colors = np.vstack(([[1, 1, 1, 1]], colors(np.linspace(0, 1, num_clusters))))
            colormap = ListedColormap(colormap_colors)

            plt.figure(figsize=(6, 6))
            plt.imshow(cluster_grid.cpu(), cmap=colormap, interpolation='nearest', origin='lower')
            plt.colorbar(ticks=range(num_clusters + 1), label='Cluster Values')
            plt.title('Posterior Cluster Assignment with BayXenSmooth')

            match data_mode:
                case "PCA":
                    data_file_path = f"{data_mode}/{num_pcs}"
                case "HVG": 
                    data_file_path = f"{data_mode}/{hvg_var_prop}"
                case "ALL":
                    data_file_path = f"{data_mode}"
                case _:
                    raise ValueError("The data mode specified is not supported.")

            if not os.path.exists(bayxensmooth_clusters_filepath := save_filepath("BayXenSmooth", "clusters", sample_for_assignment)):
                os.makedirs(bayxensmooth_clusters_filepath)
            _ = plt.savefig(
                f"{bayxensmooth_clusters_filepath}/result.png"
            )

            clusters = pd.DataFrame(cluster_assignments_q.cpu(), columns=["BayXenSmooth cluster"]).to_csv(f"{bayxensmooth_clusters_filepath}/clusters_K={num_clusters}.csv")
            soft_clusters = pd.DataFrame(cluster_probs_q, columns=[f'P(z_i = {i})'  for i in range(1, num_clusters + 1)]).to_csv(f"{bayxensmooth_clusters_filepath}/soft_clusters_K={num_clusters}.csv")

            if not os.path.exists(bayxensmooth_similar_filepath := save_filepath("BayXenSmooth", "prior_v_posterior", sample_for_assignment)):
                os.makedirs(bayxensmooth_similar_filepath)
            with open(f"{bayxensmooth_similar_filepath}/similarity.txt", 'w') as fp:
                fp.write(str(torch.mean((cluster_assignments_prior == cluster_assignments_q).float()).item()))

            # grab the mpd distance of cluster labels
            cluster_labels = np.unique(clusters)
            mpd = {}
            for label in cluster_labels:
                current_cluster_locations = torch.stack(torch.where((cluster_grid == label)), axis=1).to(float)
                mpd[f"Cluster {label}"] = spot_size * torch.mean(torch.cdist(current_cluster_locations, current_cluster_locations, p = 2)).item()

            if not os.path.exists(bayxensmooth_mpd_filepath := save_filepath("BayXenSmooth", "mpd", sample_for_assignment)):
                os.makedirs(bayxensmooth_mpd_filepath)
            with open(f"{bayxensmooth_mpd_filepath}/clusters_K={num_clusters}_mpd.json", 'w') as fp:
                json.dump(mpd, fp)

            cmap = get_cmap('rainbow')

            if evaluate_markers:

                if isinstance(original_adata.xenium_spot_data.X, csr_matrix):
                    labels = np.unique(cluster_assignments_q)  # Define the number of clusters
                    gene_columns = original_adata.xenium_spot_data.var.index  # Column names from another source
                    mean_expression_by_cluster = pd.DataFrame(columns=gene_columns)

                    # Loop through each cluster label
                    for label in labels:
                        # Find indexes of current cluster
                        current_cluster_indexes = torch.where(cluster_assignments_q == label)[0].numpy()
                        
                        # Efficiently extract the rows for the current cluster using fancy indexing
                        expressions = original_adata.xenium_spot_data.X[current_cluster_indexes, :]
                        
                        # Compute mean expressions; the result is still a csr_matrix
                        mean_expressions = expressions.mean(axis=0)
                        
                        # Convert mean_expressions to a dense format and then to a DataFrame
                        mean_expressions_df = pd.DataFrame(mean_expressions.A, columns=gene_columns)
                        
                        # Append the result to the mean_expression_by_cluster DataFrame
                        mean_expression_by_cluster = pd.concat([mean_expression_by_cluster, mean_expressions_df], ignore_index=True)
                else:
                    # identify marker genes within each cluster
                    mean_expression_by_cluster = pd.DataFrame(columns=original_adata.xenium_spot_data.var.index)

                    for label in range(num_clusters):
                        current_cluster_indexes = list(torch.where(cluster_assignments_q == label)[0].cpu().numpy())
                        expressions = pd.DataFrame(original_adata.xenium_spot_data.X, columns=original_adata.xenium_spot_data.var.index).iloc[current_cluster_indexes, :]
                        mean_expressions = expressions.mean(axis=0).to_frame().T
                        mean_expression_by_cluster = pd.concat([mean_expression_by_cluster, mean_expressions], ignore_index=True)

                for i, gene in enumerate(mean_expression_by_cluster.columns):
                    # using subplots() to draw vertical lines 
                    fig, ax = plt.subplots(figsize=(6, 6)) 
                    ax.vlines(mean_expression_by_cluster[gene].index, ymin=0, ymax=mean_expression_by_cluster[gene], color=cmap(i / (len(mean_expression_by_cluster.columns) - 1))) 
                    
                    # drawing the markers
                    ax.plot(mean_expression_by_cluster[gene].index, mean_expression_by_cluster[gene], "^", c=cmap(i / (len(mean_expression_by_cluster.columns) - 1))) 
                    ax.set_ylim(0) 
                    
                    # formatting and details 
                    ax.set_xlabel('Cluster Label') 
                    ax.set_ylabel('Mean Expression') 
                    ax.set_title(gene) 
                    ax.set_xticks(mean_expression_by_cluster[gene].index) 
                    if not os.path.exists(bayxensmooth_expression_filepath := save_filepath("BayXenSmooth", "expressions", sample_for_assignment)):
                        os.makedirs(f"{bayxensmooth_expression_filepath}")
                    _ = plt.savefig(
                        f"{bayxensmooth_expression_filepath}/GENE={gene}.png"
                    )
            
            # confidence mapping
            cluster_confidences = torch.zeros((num_rows, num_cols), dtype=torch.double)

            cluster_confidences[rows, columns] = cluster_probs_q.max(dim=1).values

            heatmap_bins = 21
            colors = plt.cm.get_cmap('YlOrRd', heatmap_bins)
            colormap_colors = np.vstack(([[1, 1, 1, 1]], colors(np.linspace(0, 1, heatmap_bins - 1))))
            colormap = ListedColormap(colormap_colors)

            plt.figure(figsize=(6, 6))
            plt.imshow(cluster_confidences, cmap=colormap, interpolation='nearest', origin='lower')
            # plt.xticks([])  # Remove x-axis tick marks
            # plt.yticks([])  # Remove y-axis tick marks
            plt.gca().spines['top'].set_visible(False)  # Remove top border
            plt.gca().spines['right'].set_visible(False)  # Remove right border
            # plt.gca().spines['bottom'].set_visible(False)  # Remove bottom border
            # plt.gca().spines['left'].set_visible(False)  # Remove left border
            cbar = plt.colorbar(fraction=0.046, pad=0.04)  # Make colorbar the same height as the figure
            plt.title(r'$P(z_i = k)$')


            colors = plt.cm.get_cmap('Greys', num_clusters + 1)
            colormap = ListedColormap(colors(np.linspace(0, 1, num_clusters + 1)))

            confidence_proportions = {}
            for uncertainty_value in uncertainty_values:
                confidence_matrix = (cluster_confidences > uncertainty_value).float()
                confidence_proportions[uncertainty_value] = torch.mean(confidence_matrix).item()
                plt.figure(figsize=(6, 6))
                plt.imshow(cluster_confidences > uncertainty_value, cmap=colormap, interpolation='nearest', origin='lower')
                # plt.xticks([])  # Remove x-axis tick marks
                # plt.yticks([])  # Remove y-axis tick marks
                plt.gca().spines['top'].set_visible(False)  # Remove top border
                plt.gca().spines['right'].set_visible(False)  # Remove right border
                # plt.gca().spines['bottom'].set_visible(False)  # Remove bottom border
                # plt.gca().spines['left'].set_visible(False)  # Remove left border
                cbar = plt.colorbar(fraction=0.046, pad=0.04)  # Make colorbar the same height as the figure
                # PLOT ALL UNCERTAINTY VALUESs
                plt.title(r'$\max_k \, P(z_i = k) > $' + f'{uncertainty_value}')
                if not os.path.exists(bayxensmooth_uncertainty_filepath := save_filepath("BayXenSmooth", "uncertainty", sample_for_assignment)):
                    os.makedirs(bayxensmooth_uncertainty_filepath)
                _ = plt.savefig(
                    f"{bayxensmooth_uncertainty_filepath}/CONFIDENCE={uncertainty_value}.png"
                )

        else:

            plt.scatter(spatial_locations["x_location"], spatial_locations["y_location"], s=1, c=cluster_assignments_q)
            if not os.path.exists(bayxensmooth_clusters_filepath := save_filepath("BayXenSmooth", "clusters", sample_for_assignment)):
                os.makedirs(bayxensmooth_clusters_filepath)
            _ = plt.savefig(
                f"{bayxensmooth_clusters_filepath}/result.png"
            )

    return cluster_probs_q, cluster_means_q_mean, cluster_scales_q_mean


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="Run Xenium SVI_Approx with different arguments")
    parser.add_argument("--custom_init", type=str, required=False)
    parser.add_argument("--neighborhood_size", type=int, required=True)
    parser.add_argument("--num_clusters", type=int, required=True)
    parser.add_argument("--spot_size", type=int, required=True)
    parser.add_argument("--data_mode", type=str, required=True)
    parser.add_argument("--num_pcs", type=int, required=True)
    parser.add_argument("--hvg_var_prop", type=float, required=True)
    parser.add_argument("--neighborhood_agg", type=str, required=True)
    parser.add_argument("--mu_prior_scale", type=float, required=True)
    parser.add_argument("--sigma_prior_scale", type=float, required=True)
    parser.add_argument("--logits_prior_scale", type=float, required=True)
    parser.add_argument("--learn_global_variances", type=str2bool, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    DATA_TYPE = "XENIUM"

    if DATA_TYPE == "XENIUM":
        # Call prepare_Xenium_data with the appropriate arguments
        gene_data, spatial_locations, original_adata = prepare_Xenium_data(
            dataset="hBreast", 
            spots=True, 
            spot_size=args.spot_size, 
            third_dim=False, 
            log_normalize=True, 
            likelihood_mode=args.data_mode, 
            num_pcs=args.num_pcs,
            hvg_var_prop=args.hvg_var_prop,
            min_expressions_per_spot=0
        )
    elif DATA_TYPE == "DLPFC":
        gene_data, spatial_locations, original_adata = prepare_DLPFC_data(
            section_id=151673,
            num_pcs=args.num_pcs,
        )

    print("Data Completed")
    
    # Call Xenium_SVI with the appropriate arguments
    cluster_probs_q, cluster_means_q_mean, cluster_scales_q_mean = Xenium_SVI(
        gene_data, 
        spatial_locations,
        original_adata,
        data_mode=args.data_mode,
        num_pcs=args.num_pcs,
        hvg_var_prop=args.hvg_var_prop, 
        dataset_name="hBreast" if DATA_TYPE == "XENIUM" else "DLPFC", 
        spot_size=args.spot_size, 
        num_clusters=args.num_clusters, 
        batch_size= 256 * int(2 ** ((100 / args.spot_size) - 1)), 
        custom_init=args.custom_init,
        neighborhood_size=args.neighborhood_size,
        neighborhood_agg=args.neighborhood_agg,
        mu_prior_scale=args.mu_prior_scale,
        sigma_prior_scale=args.sigma_prior_scale,
        logits_prior_scale=args.logits_prior_scale,
        learn_global_variances=args.learn_global_variances
    )

    sample_for_assignment_options = [False, True]

    # FIX THIS LATER
    for sample_for_assignment in sample_for_assignment_options:
        cluster_probs_q = torch.softmax(pyro.param("cluster_logits_q_mean"), dim=1)
        if sample_for_assignment:
            cluster_assignments_q = pyro.sample("cluster_probs", dist.Categorical(cluster_probs_q)) 
        else:
            cluster_assignments_q = cluster_probs_q.argmax(dim=1)

        if DATA_TYPE == "DLPFC":
            # Create a DataFrame for easier handling
            data = pd.DataFrame({
                'ClusterAssignments': cluster_assignments_q,
                'Region': original_adata.xenium_spot_data.obs["Region"]
            })

            # Drop rows where 'Region' is NaN
            filtered_data = data.dropna(subset=['Region'])

            # Calculate ARI and NMI only for the non-NaN entries
            ari = ARI(filtered_data['ClusterAssignments'], filtered_data['Region'])
            nmi = NMI(filtered_data['ClusterAssignments'], filtered_data['Region'])
            cluster_metrics = {
                "ARI": ari,
                "NMI": nmi
            }
            dataset_name="DLPFC"
            data_file_path = f"{args.data_mode}/{args.num_pcs}"

            total_file_path = (
                f"results/{dataset_name}/{args.model}/{args.component}/{data_file_path}/"
                f"NEIGHBORSIZE={args.neighborhood_size}/NUMCLUSTERS={args.num_clusters}"
                f"/SAMPLEFORASSIGNMENT={sample_for_assignment}"
                f"/SPATIALPRIORMULT=DIRECT/SPOTSIZE={args.spot_size}/AGG={args.neighborhood_agg}"
            )

            if not os.path.exists(total_file_path):
                os.makedirs(total_file_path)
            with open(f"{total_file_path}/cluster_metrics.json", 'w') as fp:
                json.dump(cluster_metrics, fp)

def main_test():
    DATA_TYPE = "XENIUM"

    if DATA_TYPE == "XENIUM":
        # Call prepare_Xenium_data with the appropriate arguments
        gene_data, spatial_locations, original_adata = prepare_Xenium_data(
            dataset="hBreast", 
            spots=True, 
            spot_size=50, 
            third_dim=False, 
            log_normalize=True, 
            likelihood_mode="PCA", 
            num_pcs=10,
            hvg_var_prop=0.9,
            min_expressions_per_spot=0
        )
    elif DATA_TYPE == "DLPFC":
        gene_data, spatial_locations, original_adata = prepare_DLPFC_data(
            section_id=151673,
            num_pcs=3,
        )

    print("Data Completed")
    
    # Call Xenium_SVI with the appropriate arguments
    for num_pcs in [3, 5, 10, 15, 25]:
        for init_method in ["K-Means", "mclust"]:
            pyro.clear_param_store()
            start_time = time.time()
            cluster_concentration_params_q, cluster_means_q_mean, cluster_scales_q_mean = Xenium_SVI(
                gene_data, 
                spatial_locations,
                original_adata,
                data_mode="PCA",
                num_pcs=num_pcs,
                hvg_var_prop=0.9, 
                dataset_name="hBreast" if DATA_TYPE == "XENIUM" else "DLPFC", 
                spot_size=50, 
                num_clusters=7, 
                batch_size= 256 * int(2 ** ((100 / 25) - 1)), 
                custom_init=init_method,
                neighborhood_size=1,
                neighborhood_agg="mean",
                mu_prior_scale=1,
                sigma_prior_scale=1,
                logits_prior_scale=1,
                learn_global_variances=True
            )
            end_time = time.time()
            print(f"Time taken for Xenium_SVI with num_pcs={num_pcs} and init_method={init_method}: {end_time - start_time} seconds")

            torch.cuda.empty_cache()

if __name__ == "__main__":
    main_test()