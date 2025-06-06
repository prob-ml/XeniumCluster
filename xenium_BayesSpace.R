# if (!requireNamespace("BiocManager", quietly = TRUE)) {
#     install.packages("BiocManager")
#     BiocManager::install("BayesSpace")
# }

library(SingleCellExperiment)
library(ggplot2)
library(BayesSpace)
library(Matrix)

# sce <- readVisium("data/sce")

BayesSpace <- function(dataset="hBreast", SPOT_SIZE=100, init_method="mclust", num_pcs=15, K=NULL, grid_search=TRUE) {
  
  rowData <- read.csv(paste0("data/BayesSpace/", dataset, "_rowData_SPOT_SIZE=", SPOT_SIZE, ".csv"), stringsAsFactors=FALSE, row.names=1)
  colData <- read.csv(paste0("data/BayesSpace/", dataset, "_colData_SPOT_SIZE=", SPOT_SIZE, ".csv"), stringsAsFactors=FALSE, row.names=1)
  countsData <- t(read.csv(paste0("data/BayesSpace/", dataset, "_counts_SPOT_SIZE=", SPOT_SIZE, ".csv"), row.names=1, check.names=F, stringsAsFactors=FALSE))
  
  # Create unique row names from the first column, then remove it from countsData
  rownames(countsData) <- rownames(rowData)
  colnames(countsData) <- rownames(colData)
  
  if (grid_search) {
    gamma_list <- seq(1.0, 3.0, by = 0.25)
  } else {
    gamma_list <- c(2)
  }
  
  for (gamma in gamma_list) {
    
    is_already_log_transformed = TRUE
    if (is_already_log_transformed) {
      sce <- SingleCellExperiment(assays=list(logcounts=as(as.matrix(countsData), "dgCMatrix")),
                                  rowData=rowData,
                                  colData=colData)  
    } else {
      sce <- SingleCellExperiment(assays=list(counts=as(as.matrix(countsData), "dgCMatrix")),
                                  rowData=rowData,
                                  colData=colData)  
    }
    
    set.seed(102)
    
    if (dataset == "SYNTHETIC") {
      pca_data <- t(countsData)
      rownames(pca_data) <- colnames(sce)  # Ensure row names match sce colnames
      reducedDims(sce) <- list(PCA=pca_data)
    } else {
      if (is_already_log_transformed) {
        sce <- spatialPreprocess(sce, platform="ST", 
                                 n.PCs=num_pcs, n.HVGs=2000, log.normalize=FALSE)
      } else {
        sce <- spatialPreprocess(sce, platform="ST", 
                                 n.PCs=num_pcs, n.HVGs=2000, log.normalize=TRUE) 
      }
    }
    
    sce <- sce[, !grepl("^(BLANK_|NegControl)", colnames(sce))]
    
    set.seed(149)
    if (is.null(K)) {
      sce <- qTune(sce, qs=seq(2, 15), platform="ST", d=num_pcs)
      qPlot(sce)
      q_optimal <- attr(sce, "q.logliks")[which.max(attr(sce, "q.logliks")$loglik),]$q
    }
    else
      q_optimal <- K
    
    NREP = 1000
    sce <- spatialCluster(sce, q=q_optimal, platform="ST", d=num_pcs,
                          init.method=init_method, model="t", gamma=gamma,
                          nrep=NREP, burn.in=100,
                          save.chain=TRUE)
    chain <- mcmcChain(sce)
    final_means <- chain[(NREP-100):NREP, 26:50]
    final_scales <- chain[(NREP-100):NREP, 1:25]
    
    dir_path <- paste0("results/", dataset, "/BayesSpace/", num_pcs, "/", q_optimal, "/clusters/", init_method, "/", SPOT_SIZE, "/", format(gamma, nsmall = 2))
    if (!dir.exists(dir_path)) {
      dir.create(dir_path, recursive = TRUE)
    }
    
    means_path <- paste0(dir_path, "/means")
    if (!dir.exists(means_path)) {
      dir.create(means_path, recursive = TRUE)
    }
    
    scales_path <- paste0(dir_path, "/scales")
    if (!dir.exists(scales_path)) {
      dir.create(scales_path, recursive = TRUE)
    }
    cluster_data <- data.frame(sce$spatial.cluster)
    colnames(cluster_data) <- c("BayesSpace cluster")
    write.csv(cluster_data, paste0(dir_path, "/clusters_K=", q_optimal, ".csv"), row.names = TRUE)
    write.csv(final_means, paste0(dir_path, "/means", "/clusters_K=", q_optimal, ".csv"), row.names = TRUE)
    write.csv(final_scales, paste0(dir_path, "/scales", "/clusters_K=", q_optimal, ".csv"), row.names = TRUE)
    
    clusterPlot(sce, palette=c("purple", "red", "blue", "yellow"), color="black") +
      theme_bw() +
      xlab("Column") +
      ylab("Row") +
      labs(fill="BayesSpace\ncluster", title="Spatial clustering of ST_mel1_rep2")
  }
  return(sce$spatial.cluster)
}

args <- commandArgs(trailingOnly = TRUE)
dataset_name <- args[1]
SPOT_SIZE <- as.numeric(args[2])
init_method <- args[3]
num_pcs <- as.numeric(args[4])
K <- ifelse(length(args) > 4, as.numeric(args[5]), NULL)
grid_search <- ifelse(length(args) > 5, as.logical(args[6]), NULL)
cat(c(SPOT_SIZE, num_pcs, K))
BayesSpace(dataset_name, SPOT_SIZE, init_method, num_pcs, K, grid_search)