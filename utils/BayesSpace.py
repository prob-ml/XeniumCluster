import pandas as pd

def Xenium_to_BayesSpace(xenium_spot_data):

    rowData = xenium_spot_data.var
    colData = xenium_spot_data.obs
    counts = xenium_spot_data.X

    rowData.to_csv("rowData.csv", index=False)
    colData.to_csv("colData.csv", index=False)
    pd.DataFrame(counts).to_csv("counts.csv", index=False)