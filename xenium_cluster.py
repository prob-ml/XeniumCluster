import scanpy as sc
import scipy.cluster.hierarchy as sch
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from utils.embeddings import *
from typing import List

class XeniumCluster:

    # TO DO
    # add option to only include high variable genes

    SPOT_SIZE = 100
    THIRD_DIM = False

    def __init__(self, data: pd.DataFrame, dataset_name: str, spot_size: int = 100) -> None:
        
        self.raw_xenium_data = data
        self.xenium_spot_data = None
        self.dataset_name = dataset_name
        self.SPOT_SIZE = spot_size
        self.spot_data_location = f"data/spot_data/{dataset_name}"

    def set_spot_size(self, new_spot_size):

        if not isinstance(new_spot_size, (int, float)): 
            raise TypeError("The spot size must be numeric.")
        if new_spot_size <= 0:
            raise ValueError("Spot size must be positive.")
        self.SPOT_SIZE = new_spot_size

    # update this to be a re-init procedure
    def set_data(self, data):

        self.raw_xenium_data = data

    def convert_pd_to_ad(self, data):

        obs_df = data.index.to_frame(index=False).astype("category")

        return sc.AnnData(X=data.values, obs=obs_df, var=pd.DataFrame(index=data.columns))
    
    def normalize_counts(self, data):

        data.layers['raw']=data.X
        # Why does the demo do this????
        # sc.pp.normalize_total(data, target_sum=1e4)
        sc.pp.log1p(data)

    def create_spot_data(self, third_dim=False, save_data=True):

        x_min, x_max = min(self.raw_xenium_data["x_location"]), max(self.raw_xenium_data["x_location"])
        y_min, y_max = min(self.raw_xenium_data["y_location"]), max(self.raw_xenium_data["y_location"])

        MIN_PAD = 1e-8

        x_values = np.arange(x_min - MIN_PAD, x_max + self.SPOT_SIZE, self.SPOT_SIZE)
        y_values = np.arange(y_min - MIN_PAD, y_max + self.SPOT_SIZE, self.SPOT_SIZE)
        

        self.xenium_spot_data = self.raw_xenium_data.copy()

        if third_dim:
            z_min, z_max = min(self.raw_xenium_data["z_location"]), max(self.raw_xenium_data["z_location"])
            z_values = np.arange(z_min - MIN_PAD, z_max + self.SPOT_SIZE, self.SPOT_SIZE)
            self.xenium_spot_data["col"] = np.searchsorted(x_values, self.xenium_spot_data["x_location"]) - 1
            self.xenium_spot_data["row"] = np.searchsorted(y_values, self.xenium_spot_data["y_location"]) - 1
            self.xenium_spot_data["z-index"] = np.searchsorted(z_values, self.xenium_spot_data["z_location"]) - 1
            self.xenium_spot_data["spot_number"] = (self.xenium_spot_data["col"] * len(y_values) * len(z_values)) + (self.xenium_spot_data["row"] * len(z_values)) + self.xenium_spot_data["z-index"]
        else:
            self.xenium_spot_data["col"] = np.searchsorted(x_values, self.xenium_spot_data["x_location"]) - 1
            self.xenium_spot_data["row"] = np.searchsorted(y_values, self.xenium_spot_data["y_location"]) - 1
            self.xenium_spot_data["spot_number"] = self.xenium_spot_data["col"] * len(y_values) + self.xenium_spot_data["row"]

        counts = self.xenium_spot_data.groupby(['spot_number', 'feature_name']).size().reset_index(name='count')

        counts_pivot = counts.pivot_table(index='spot_number', 
                                  columns='feature_name', 
                                  values='count', 
                                  fill_value=0)
        
        location_means = self.xenium_spot_data.groupby('spot_number').agg({
            'row': 'mean',
            'col': 'mean',
            'x_location': 'mean',
            'y_location': 'mean',
            'z_location': 'mean'
        }).reset_index()

        self.xenium_spot_data = location_means.join(counts_pivot, on='spot_number')

        if save_data:
            self.xenium_spot_data.to_csv(f"{self.spot_data_location}/{self.dataset_name}_SPOTSIZE={self.SPOT_SIZE}um_z={third_dim}.csv")

        self.xenium_spot_data.set_index(["spot_number", "x_location", "y_location", "z_location", "row", "col"], inplace=True)

        self.xenium_spot_data = self.convert_pd_to_ad(self.xenium_spot_data)

    def generate_neighborhood_graph(self, data: ad.AnnData, n_neighbors=15, n_pcs=20, plot_pcas=True):
        
        # generate the neigborhood graph based on pca
        sc.tl.pca(data, svd_solver='arpack')
        if plot_pcas:
            sc.pl.pca_variance_ratio(data, log=True)
        sc.pp.neighbors(data, n_neighbors=n_neighbors, n_pcs=n_pcs)

    def filter_only_high_variable_genes(self, data: ad.AnnData, min_mean: float=0.3, max_mean: float=7, min_disp: float=-0.5, plot_highly_variable_genes: bool = False):
        sc.pp.highly_variable_genes(data, min_mean=min_mean, max_mean=max_mean, min_disp=min_disp)
        if plot_highly_variable_genes:
            sc.pl.highly_variable_genes(data)

    def pca(self, data: ad.AnnData, num_pcs: int):
        sc.pp.pca(data, num_pcs)

    def Leiden(
            self,
            data: ad.AnnData,
            resolutions: List[float],
            save_plot: bool = False,
            embedding: str = "umap", 
            **kwargs
        ):

        for resolution in resolutions:

            key_added = f'leiden_{resolution}'

            sc.tl.leiden(data, resolution=resolution, key_added=key_added)

            # calculate embedding
            get_embedding(data, embedding, **kwargs)

            # plot embedding
            _ = plot_embedding(data, key_added, embedding, **kwargs)

            # save plot
            if save_plot:
                # Directory where you want to save the file
                directory = f"results/{self.dataset_name}/Leiden/"

                # Use os.makedirs to create the directory if it does not exist
                os.makedirs(directory, exist_ok=True)

                plt.savefig(f'{directory}{self.SPOT_SIZE}um_{embedding}_{resolution}_Z={self.THIRD_DIM}.png')

                colors = [int(x) for x in data.obs[key_added]]

                unique_clusters = np.unique(colors)

                for cluster_id in unique_clusters:

                    indices = np.where(colors == cluster_id)[0]
        
                    plt.scatter(
                        self.xenium_spot_data.obs["x_location"].iloc[indices],
                        self.xenium_spot_data.obs["y_location"].iloc[indices],
                        s=2,
                        label=f'Cluster {cluster_id}'
                    )

                plt.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xlabel("x_coord")
                plt.ylabel("y_coord")
                _ = plt.title(f"Leiden Cluster Visualization: Resolution = {resolution}, Spot Size = {self.SPOT_SIZE}")
                plt.savefig(f'{directory}{self.SPOT_SIZE}um_{resolution}_Z={self.THIRD_DIM}_CLUSTERS.png', bbox_inches='tight')

        return {resolution: data.obs[f'leiden_{resolution}'] for resolution in resolutions}

    def Louvain(
            self,
            data: ad.AnnData,
            resolutions: List[float],
            save_plot: bool = False,
            embedding: str = "umap", 
            **kwargs
        ):

        for resolution in resolutions:

            key_added = f'louvain_{resolution}'

            sc.tl.louvain(data, resolution=resolution, key_added=key_added)

            # calculate embedding
            get_embedding(data, embedding, **kwargs)

            # plot embedding
            _ = plot_embedding(data, key_added, embedding, **kwargs)

            # save plot
            if save_plot:
                # Directory where you want to save the file
                directory = f"results/{self.dataset_name}/Louvain/"

                # Use os.makedirs to create the directory if it does not exist
                os.makedirs(directory, exist_ok=True)

                plt.savefig(f'{directory}{self.SPOT_SIZE}um_{embedding}_{resolution}_Z={self.THIRD_DIM}.png')

                colors = [int(x) for x in data.obs[key_added]]

                unique_clusters = np.unique(colors)

                for cluster_id in unique_clusters:

                    indices = np.where(colors == cluster_id)[0]
        
                    plt.scatter(
                        self.xenium_spot_data.obs["x_location"].iloc[indices],
                        self.xenium_spot_data.obs["y_location"].iloc[indices],
                        s=2,
                        label=f'Cluster {cluster_id}'
                    )

                plt.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xlabel("x_coord")
                plt.ylabel("y_coord")
                _ = plt.title(f"Louvain Cluster Visualization: Resolution = {resolution}, Spot Size = {self.SPOT_SIZE}")
                plt.savefig(f'{directory}{self.SPOT_SIZE}um_{resolution}_Z={self.THIRD_DIM}_CLUSTERS.png')

        return {resolution: data.obs[f'louvain_{resolution}'] for resolution in resolutions}

    def Hierarchical(
            self,
            data: ad.AnnData,
            groupby: List[str] = ["spot_number"],
            save_plot: bool = False,
            embedding: str = "umap",
            **kwargs
        ):

        key_added = f'dendrogram_{groupby}'
        
        # calculate cluster assignment
        sc.tl.dendrogram(data, groupby=groupby, key_added=key_added)
        linkage_matrix = data.uns[key_added]['linkage']

        # Decide on the number of clusters
        num_clusters = 3  # Example: aiming for 3 clusters

        # Form clusters from the dendrogram
        cluster_labels = sch.fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')

        # Assign cluster labels to observations
        data.obs[key_added] = cluster_labels

        # plot dendrogram
        # sc.pl.dendrogram(data, groupby=groupby)

        # calculate embedding
        get_embedding(data, embedding, **kwargs)

        # plot embedding
        _ = plot_embedding(data, key_added, embedding, **kwargs)

        # save plot
        if save_plot:

            # Directory where you want to save the file
            directory = f"results/{self.dataset_name}/hierarchical/"

            # Use os.makedirs to create the directory if it does not exist
            os.makedirs(directory, exist_ok=True)

            plt.savefig(f'{directory}{self.SPOT_SIZE}um_{embedding}_Z={self.THIRD_DIM}.png')

            colors = [int(x) for x in data.obs[key_added]]

            unique_clusters = np.unique(colors)

            for cluster_id in unique_clusters:

                indices = np.where(colors == cluster_id)[0]
    
                plt.scatter(
                    self.xenium_spot_data.obs["x_location"].iloc[indices],
                    self.xenium_spot_data.obs["y_location"].iloc[indices],
                    s=2,
                    label=f'Cluster {cluster_id}'
                )

            plt.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xlabel("x_coord")
            plt.ylabel("y_coord")
            _ = plt.title(f"Hierarchical Cluster Visualization, Spot Size = {self.SPOT_SIZE}")
            plt.savefig(f'{directory}{self.SPOT_SIZE}um_Z={self.THIRD_DIM}_CLUSTERS.png')

        return data.obs[key_added]
