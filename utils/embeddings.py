import scanpy as sc
import anndata as ad


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
        cluster_key: str,
        embedding: str = "umap",
        **kwargs # will make this something you can customize later
    ):

    if embedding == "umap":
        sc.pl.umap(data,size=30,color=cluster_key,legend_loc='on data',legend_fontsize=3,legend_fontoutline=1,show=False,palette="rainbow")
    elif embedding == "tsne":
        sc.pl.tsne(data,size=30,color=cluster_key,legend_loc='on data',legend_fontsize=3,legend_fontoutline=1,show=False,palette="rainbow")
    elif embedding == "pca":
        sc.pl.pca(data,size=30,color=cluster_key,legend_loc='on data',legend_fontsize=3,legend_fontoutline=1,show=False,palette="rainbow")
    elif embedding == "diffmap":
        sc.pl.diffmap(data,size=30,color=cluster_key,legend_loc='on data',legend_fontsize=3,legend_fontoutline=1,show=False,palette="rainbow")