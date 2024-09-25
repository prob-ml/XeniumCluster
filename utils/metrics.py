import numpy as np
import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

MARKER_GENES = ["BANK1", "CEACAM6", "FASN", "FGL2", "IL7R", "KRT6B", "POSTN", "TCIM"]


def NMI(cluster_pred_1, cluster_pred_2):
    return normalized_mutual_info_score(cluster_pred_1, cluster_pred_2)

def ARI(cluster_pred_1, cluster_pred_2):
    return adjusted_rand_score(cluster_pred_1, cluster_pred_2)

def gene_morans_i(clustering, locations, clusters, num_neighbors=100, marker_genes=MARKER_GENES, untransformed_counts=None):
    print("Starting Moran's I Calculation.")
    moran_clusters = ad.AnnData(locations)
    sc.pp.neighbors(moran_clusters, n_pcs=0, n_neighbors=num_neighbors)
    print("Neighbors calculated.")

    # Create a binary adjacency matrix indicating if points are in the same cluster
    cluster_labels = clusters.values
    same_cluster = (cluster_labels[:, None] == cluster_labels).astype(int)
    print(moran_clusters.obsp["connectivities"].shape, same_cluster.shape)
    moran_clusters.obsp["connectivities"] = moran_clusters.obsp["connectivities"].multiply(csr_matrix(same_cluster))
    print("Connectivities formed.")

    # Calculate Moran's I for the genes
    if untransformed_counts is not None:
        morans_i = sc.metrics.morans_i(moran_clusters, vals=untransformed_counts.T)
    else:
        morans_i = sc.metrics.morans_i(moran_clusters, vals=clustering.xenium_spot_data.X.T)

    morans_i_dict = dict(zip(clustering.xenium_spot_data.var.index, morans_i))

    for gene in marker_genes:
        print(num_neighbors, gene, morans_i_dict[gene])

    return morans_i_dict

def gene_gearys_c(clustering, locations, clusters, num_neighbors=100, untransformed_counts=None):
    print("Starting Geary's C Calculation.")
    gearys_clusters = ad.AnnData(locations)
    sc.pp.neighbors(gearys_clusters, n_pcs=0, n_neighbors=num_neighbors)
    print("Neighbors calculated.")

    # Create a binary adjacency matrix indicating if points are in the same cluster
    cluster_labels = clusters.values
    same_cluster = (cluster_labels[:, None] == cluster_labels).astype(int)
    gearys_clusters.obsp["connectivities"] = gearys_clusters.obsp["connectivities"].multiply(csr_matrix(same_cluster))
    print("Connectivities formed.")

    # Calculate Geary's C for the genes
    if untransformed_counts is not None:
        gearys_c = sc.metrics.gearys_c(gearys_clusters, vals=untransformed_counts.T)
    else:
        gearys_c = sc.metrics.gearys_c(gearys_clusters, vals=clustering.xenium_spot_data.X.T)

    gearys_c_dict = dict(zip(clustering.xenium_spot_data.var.index, gearys_c))

    return gearys_c_dict

def log2FC(original_adata, clusters, genes=["BANK1"]):

    if clusters.ndim == 2 and clusters.shape[1] == 1:
        clusters = clusters.squeeze()
    elif clusters.ndim > 2:
        raise ValueError("Clusters must be able to be converted to a vector.")

    # total gene counts per spot
    spot_sums = original_adata.xenium_spot_data.X.sum(axis=1)

    # size factor per cell
    spot_size_factor = spot_sums/np.median(spot_sums)

    log2FC_by_cluster = {}

    for gene in genes:
        log2FC_by_cluster[gene] = {}
        gene_index = original_adata.xenium_spot_data.var.index.get_loc(gene)
        gene_expression = original_adata.xenium_spot_data.X[:, gene_index]
        for val in np.unique(clusters):
            val_spots = (clusters == val)
            opposing_spots = (clusters != val)

            cluster_size_factor = np.sum(spot_size_factor[val_spots])
            other_clusters_size_factor = np.sum(spot_size_factor[opposing_spots])
            cluster_gene_sum = np.sum(gene_expression[val_spots])
            other_clusters_gene_sum = np.sum(gene_expression[opposing_spots])

            log2FC_value = np.log2( (1 + cluster_gene_sum) / (1 + cluster_size_factor) ) - np.log2( (1 + other_clusters_gene_sum) / (1 + other_clusters_size_factor) )

            log2FC_by_cluster[gene][val] = log2FC_value

    return log2FC_by_cluster