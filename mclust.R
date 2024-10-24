if (!requireNamespace("mclust", quietly = TRUE)) {
    install.packages("mclust")
}
library(mclust)

mclust_wrapper <- function(data, G, num_pcs=10, SPOT_SIZE=50, dataset_name="hBreast") {
    res = Mclust(data, G=G, modelNames="EEE")

    dir_path <- paste0("results/", dataset_name, "/mclust/", num_pcs, "/", G, "/clusters/", SPOT_SIZE)
    if (!dir.exists(dir_path)) {
      dir.create(dir_path, recursive = TRUE)
    }

    cluster_data <- data.frame(res$classification)
    colnames(cluster_data) <- c("mclust cluster")
    write.csv(cluster_data, paste0(dir_path, "/clusters_K=", G, ".csv"), row.names = TRUE)

    return(res$G)
}

args <- commandArgs(trailingOnly = TRUE)
data <- as.matrix(read.csv(args[1], header = FALSE))
G <- as.numeric(args[2])
num_pcs <- as.numeric(args[3])
SPOT_SIZE <- as.numeric(args[4])
dataset_name <- ifelse(length(args) > 4, args[5], "hBreast")
mclust_wrapper(data, G, num_pcs, SPOT_SIZE, dataset_name)
