import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from typing import List

class XeniumCluster:

    # TO DO
    # add option to only include high variable genes
    # create a metric that performs cluster assignment comparison 

    SPOT_SIZE = 100

    def __init__(self, data: pd.DataFrame, dataset_name: str) -> None:
        
        self.raw_xenium_data = data
        self.xenium_spot_data = None
        self.dataset_name = dataset_name
        self.spot_data_location = f"data/spot_data/{dataset_name}/{dataset_name}_SPOTSIZE={self.SPOT_SIZE}um.csv"

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

        return sc.AnnData(data)
    
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
            self.xenium_spot_data["spot_number"] = (np.searchsorted(x_values, self.xenium_spot_data["x_location"]) - 1) * len(y_values) * len(z_values) + (np.searchsorted(y_values, self.xenium_spot_data["y_location"]) - 1) * len(z_values) + (np.searchsorted(y_values, self.xenium_spot_data["z_location"]) - 1)
        else:
            self.xenium_spot_data["spot_number"] = (np.searchsorted(x_values, self.xenium_spot_data["x_location"]) - 1) * len(y_values) + (np.searchsorted(y_values, self.xenium_spot_data["y_location"]) - 1)

        counts = self.xenium_spot_data.groupby(['spot_number', 'feature_name']).size().reset_index(name='count')

        counts_pivot = counts.pivot_table(index='spot_number', 
                                  columns='feature_name', 
                                  values='count', 
                                  fill_value=0)
        
        location_means = self.xenium_spot_data.groupby('spot_number').agg({
            'x_location': 'mean',
            'y_location': 'mean',
            'z_location': 'mean'
        }).reset_index()

        self.xenium_spot_data = location_means.join(counts_pivot, on='spot_number')

        if save_data:
            self.xenium_spot_data.to_csv(self.spot_data_location)

        self.xenium_spot_data = self.convert_pd_to_ad(self.xenium_spot_data)

    def generate_neighborhood_graph(self, data: ad.AnnData, plot_pcas=True):
        
        # generate the neigborhood graph based on pca
        sc.tl.pca(data, svd_solver='arpack')
        if plot_pcas:
            sc.pl.pca_variance_ratio(data, log=True)
        sc.pp.neighbors(data, n_neighbors=15, n_pcs=20)
        
    def get_embedding(
            data: ad.AnnData,
            embedding: str = "umap",
            **kwargs
        ):
    
        if embedding == "umap":
            sc.tl.umap(data, min_dist=0.1)
        elif embedding == "tsne":
            sc.tl.tsne(data, **kwargs)
        elif embedding == "pca":
            sc.tl.pca(data, **kwargs)
        elif embedding == "diffmap":
            sc.tl.diffmap(data, **kwargs)

    def plot_embedding(
        data: ad.AnnData,
        embedding: str = "umap",
        **kwargs
    ):
        
        if embedding == "umap":
            sc.pl.umap(data, min_dist=0.1)
        elif embedding == "tsne":
            sc.pl.tsne(data, **kwargs)
        elif embedding == "pca":
            sc.pl.pca(data, **kwargs)
        elif embedding == "diffmap":
            sc.pl.diffmap(data, **kwargs)

    def BayesSpace(data: ad.AnnData):
        pass

    def Leiden(
            self,
            data: ad.AnnData,
            resolutions: List[float],
            save_plot: bool = False,
            embedding: str = "umap", 
            **kwargs
        ):

        for resolution in resolutions:

            sc.tl.leiden(data, resolution=resolution, key_added=f'leiden_{resolution}')
            print(f"Cluster with resolutions {resolution}", data.obs[f'leiden_{resolution}'], sep=" ")

            # calculate embedding
            self.get_embedding(data, embedding, **kwargs)

            # plot embedding
            self.plot_embedding(data, embedding, **kwargs)

            # save plot
            if save_plot:
                plt.savefig(f'results/{self.dataset_name}/Leiden/{embedding}_{resolution}.png')

        return {resolution: data.obs[f'leiden_{resolution}'] for resolution in resolutions}

    def Louvian(
            self,
            data: ad.AnnData,
            resolutions: List[float],
            save_plot: bool = False,
            embedding: str = "umap", 
            **kwargs
        ):

        for resolution in resolutions:

            sc.tl.louvain(data, resolution=resolution, key_added=f'louvain_{resolution}')

            # calculate embedding
            self.get_embedding(data, embedding, **kwargs)

            # plot embedding
            self.plot_embedding(data, embedding, **kwargs)

            # save plot
            if save_plot:
                plt.savefig(f'results/{self.dataset_name}/Louvain/{embedding}_{resolution}.png')

        return {resolution: data.obs[f'louvain_{resolution}'] for resolution in resolutions}

    def Hierarchical(
            self,
            data: ad.AnnData,
            save_plot: bool = False,
            embedding: str = "umap",
            **kwargs
        ):
        
        # calculate embedding
        sc.tl.dendrogram(data, key_added=f'dendrogram')

        # plot dendrogram
        sc.pl.dendrogram(data)

        # calculate embedding
        self.get_embedding(data, embedding, **kwargs)

        # plot embedding
        self.plot_embedding(data, embedding, **kwargs)

        # save plot
        if save_plot:
            plt.savefig(f'results/{self.dataset_name}/hierarchical/{embedding}.png')

        return data.obs['dendrogram']


