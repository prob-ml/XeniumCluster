import subprocess

# List of argument sets
argument_sets = [
    {"kmeans_init": True, "neighborhood_size": 1, "num_clusters": 17, "spatial_init": True, "sample_for_assignment": False, "spatial_normalize": 1.0, "concentration_amplification": 3, "spot_size": 50, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9,},
    {"kmeans_init": True, "neighborhood_size": 1, "num_clusters": 17, "spatial_init": True, "sample_for_assignment": False, "spatial_normalize": 1.0, "concentration_amplification": 3, "spot_size": 75, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9,},
    {"kmeans_init": True, "neighborhood_size": 1, "num_clusters": 17, "spatial_init": True, "sample_for_assignment": False, "spatial_normalize": 1.0, "concentration_amplification": 3, "spot_size": 100, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9,},
    # {"kmeans_init": True, "neighborhood_size": 2, "num_clusters": 17, "spatial_init": True, "sample_for_assignment": False, "spatial_normalize": 1.0, "concentration_amplification": 3, "spot_size": 50, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9,},
    # {"kmeans_init": True, "neighborhood_size": 2, "num_clusters": 17, "spatial_init": True, "sample_for_assignment": False, "spatial_normalize": 1.0, "concentration_amplification": 3, "spot_size": 75, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9,},
    # {"kmeans_init": True, "neighborhood_size": 2, "num_clusters": 17, "spatial_init": True, "sample_for_assignment": False, "spatial_normalize": 1.0, "concentration_amplification": 3, "spot_size": 100, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9,},
    # {"kmeans_init": True, "neighborhood_size": 3, "num_clusters": 17, "spatial_init": True, "sample_for_assignment": False, "spatial_normalize": 1.0, "concentration_amplification": 3, "spot_size": 50, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9,},
    # {"kmeans_init": True, "neighborhood_size": 3, "num_clusters": 17, "spatial_init": True, "sample_for_assignment": False, "spatial_normalize": 1.0, "concentration_amplification": 3, "spot_size": 75, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9,},
    # {"kmeans_init": True, "neighborhood_size": 3, "num_clusters": 17, "spatial_init": True, "sample_for_assignment": False, "spatial_normalize": 1.0, "concentration_amplification": 3, "spot_size": 100, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9,},
    # {"kmeans_init": True, "neighborhood_size": 4, "num_clusters": 17, "spatial_init": True, "sample_for_assignment": False, "spatial_normalize": 1.0, "concentration_amplification": 3, "spot_size": 50, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9,},
    # {"kmeans_init": True, "neighborhood_size": 4, "num_clusters": 17, "spatial_init": True, "sample_for_assignment": False, "spatial_normalize": 1.0, "concentration_amplification": 3, "spot_size": 75, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9,},
    # {"kmeans_init": True, "neighborhood_size": 4, "num_clusters": 17, "spatial_init": True, "sample_for_assignment": False, "spatial_normalize": 1.0, "concentration_amplification": 3, "spot_size": 100, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9,},
    # {"kmeans_init": True, "neighborhood_size": 5, "num_clusters": 17, "spatial_init": True, "sample_for_assignment": False, "spatial_normalize": 1.0, "concentration_amplification": 3, "spot_size": 50, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9,},
    # {"kmeans_init": True, "neighborhood_size": 5, "num_clusters": 17, "spatial_init": True, "sample_for_assignment": False, "spatial_normalize": 1.0, "concentration_amplification": 3, "spot_size": 75, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9,},
    # {"kmeans_init": True, "neighborhood_size": 5, "num_clusters": 17, "spatial_init": True, "sample_for_assignment": False, "spatial_normalize": 1.0, "concentration_amplification": 3, "spot_size": 100, "num_pcs": 25, "data_mode": "PCA", "hvg_var_prop": 0.9,},
]

for args in argument_sets:
    command = [
        "python", "SVI.py",
        f"--kmeans_init={args['kmeans_init']}",
        f"--neighborhood_size={args['neighborhood_size']}",
        f"--num_clusters={args['num_clusters']}",
        f"--spatial_init={args['spatial_init']}",
        f"--sample_for_assignment={args['sample_for_assignment']}",
        f"--spatial_normalize={args['spatial_normalize']}",
        f"--concentration_amplification={args['concentration_amplification']}",
        f"--spot_size={args['spot_size']}",
        f"--data_mode={args['data_mode']}",
        f"--num_pcs={args['num_pcs']}",
        f"--hvg_var_prop={args['hvg_var_prop']}",
    ]
    subprocess.run(command)
