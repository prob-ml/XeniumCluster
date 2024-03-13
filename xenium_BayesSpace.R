if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("BayesSpace")

library(SingleCellExperiment)
library(ggplot2)
library(BayesSpace)
library(Matrix)

# sce <- readVisium("data/hBreast")

rowData <- read.csv("../Downloads/rowData.csv", stringsAsFactors=FALSE, row.names=1)
colData <- read.csv("../Downloads/colData.csv", stringsAsFactors=FALSE, row.names=1)
countsData <- t(read.csv("../Downloads/counts.csv", row.names=1, check.names=F, stringsAsFactors=FALSE))

# Create unique row names from the first column, then remove it from countsData
rownames(countsData)<- rownames(rowData)
colnames(countsData) <-rownames(colData)

sce <- SingleCellExperiment(assays=list(counts=as(as.matrix(countsData), "dgCMatrix")),
                            rowData=rowData,
                            colData=colData)

hBreast <- sce

set.seed(102)
hBreast <- spatialPreprocess(hBreast, platform="ST", 
                              n.PCs=7, n.HVGs=2000, log.normalize=TRUE)

hBreast <- hBreast[, !grepl("^(BLANK_|NegControl)", colnames(hBreast))]
hBreast <- qTune(hBreast, qs=seq(3, 7), platform="ST", d=7)
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