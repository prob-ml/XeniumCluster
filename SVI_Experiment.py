import subprocess

# List of argument sets
argument_sets = [
    {"kmeans_init": True, "neighborhood_size": 1, "num_clusters": 17, "spatial_init": True, "spatial_normalize": 0.05, "concentration_amplification": 1.0, "spot_size": 50, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9, "neighborhood_agg": "sum"},
    {"kmeans_init": True, "neighborhood_size": 1, "num_clusters": 17, "spatial_init": True, "spatial_normalize": 0.05, "concentration_amplification": 1.0, "spot_size": 50, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9, "neighborhood_agg": "mean"},
    {"kmeans_init": True, "neighborhood_size": 1, "num_clusters": 17, "spatial_init": True, "spatial_normalize": 0.1, "concentration_amplification": 1.0, "spot_size": 50, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9, "neighborhood_agg": "sum"},
    {"kmeans_init": True, "neighborhood_size": 1, "num_clusters": 17, "spatial_init": True, "spatial_normalize": 0.1, "concentration_amplification": 1.0, "spot_size": 50, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9, "neighborhood_agg": "mean"},
    {"kmeans_init": True, "neighborhood_size": 1, "num_clusters": 17, "spatial_init": True, "spatial_normalize": 0.25, "concentration_amplification": 1.0, "spot_size": 50, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9, "neighborhood_agg": "mean"},
    {"kmeans_init": True, "neighborhood_size": 1, "num_clusters": 17, "spatial_init": True, "spatial_normalize": 0.25, "concentration_amplification": 1.0, "spot_size": 50, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9, "neighborhood_agg": "sum"},
    {"kmeans_init": True, "neighborhood_size": 1, "num_clusters": 17, "spatial_init": True, "spatial_normalize": 0.5, "concentration_amplification": 1.0, "spot_size": 50, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9, "neighborhood_agg": "mean"},
    {"kmeans_init": True, "neighborhood_size": 1, "num_clusters": 17, "spatial_init": True, "spatial_normalize": 0.5, "concentration_amplification": 1.0, "spot_size": 50, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9, "neighborhood_agg": "sum"},
    {"kmeans_init": True, "neighborhood_size": 1, "num_clusters": 17, "spatial_init": False, "spatial_normalize": 0.5, "concentration_amplification": 1.0, "spot_size": 50, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9, "neighborhood_agg": "mean"},
    {"kmeans_init": True, "neighborhood_size": 1, "num_clusters": 17, "spatial_init": False, "spatial_normalize": 0.5, "concentration_amplification": 1.0, "spot_size": 50, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9, "neighborhood_agg": "sum"},
]

for args in argument_sets:
    command = [
        "python", "SVI.py",
        f"--kmeans_init={args['kmeans_init']}",
        f"--neighborhood_size={args['neighborhood_size']}",
        f"--num_clusters={args['num_clusters']}",
        f"--spatial_init={args['spatial_init']}",
        f"--spatial_normalize={args['spatial_normalize']}",
        f"--concentration_amplification={args['concentration_amplification']}",
        f"--spot_size={args['spot_size']}",
        f"--data_mode={args['data_mode']}",
        f"--num_pcs={args['num_pcs']}",
        f"--hvg_var_prop={args['hvg_var_prop']}",
        f"--neighborhood_agg={args['neighborhood_agg']}"
    ]
    subprocess.run(command)
