import pandas as pd

def Xenium_to_BayesSpace(xenium_spot_data, dataset_name = "hBreast", spot_size = 100):

    rowData = xenium_spot_data.var.index
    colData = xenium_spot_data.obs
    counts = xenium_spot_data.X

    pd.DataFrame(rowData).to_csv(f"data/BayesSpace/{dataset_name}_rowData_SPOT_SIZE={spot_size}.csv", index=False)
    colData.to_csv(f"data/BayesSpace/{dataset_name}_colData_SPOT_SIZE={spot_size}.csv", index=False)
    pd.DataFrame(counts).to_csv(f"data/BayesSpace/{dataset_name}_counts_SPOT_SIZE={spot_size}.csv")