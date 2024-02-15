# Install devtools, if necessary
if (!requireNamespace("devtools", quietly = TRUE))
    install.packages("devtools")

devtools::install_github("edward130603/BayesSpace")

library(SingleCellExperiment)
library(ggplot2)
library(BayesSpace)
library(Matrix)

sce <- readVisium(".")

rowData <- read.csv("path/to/rowData.csv", stringsAsFactors=FALSE)
colData <- read.csv("path/to/colData.csv", stringsAsFactors=FALSE, row.names=1)
counts <- read.csv("path/to/counts.csv.gz",
                   row.names=1, check.names=F, stringsAsFactors=FALSE)

hBreast <- SingleCellExperiment(assays=list(counts=as(counts, "dgCMatrix")),
                            rowData=rowData,
                            colData=colData)

set.seed(102)
hBreast <- spatialPreprocess(hBreast, platform="ST", 
                              n.PCs=7, n.HVGs=2000, log.normalize=FALSE)

hBreast <- qTune(hBreast, qs=seq(2, 10), platform="ST", d=7)
qPlot(hBreast)

set.seed(149)
hBreast <- spatialCluster(hBreast, q=4, platform="ST", d=7,
                           init.method="mclust", model="t", gamma=2,
                           nrep=1000, burn.in=100,
                           save.chain=TRUE)

set.seed(149)
hBreast <- spatialCluster(hBreast, q=4, platform="ST", d=7,
                           init.method="mclust", model="t", gamma=2,
                           nrep=1000, burn.in=100,
                           save.chain=TRUE)

clusterPlot(hBreast)

clusterPlot(hBreast, palette=c("purple", "red", "blue", "yellow"), color="black") +
  theme_bw() +
  xlab("Column") +
  ylab("Row") +
  labs(fill="BayesSpace\ncluster", title="Spatial clustering of ST_mel1_rep2")

hBreast.enhanced <- spatialEnhance(hBreast, q=4, platform="ST", d=7,
                                    model="t", gamma=2,
                                    jitter_prior=0.3, jitter_scale=3.5,
                                    nrep=1000, burn.in=100,
                                    save.chain=TRUE)