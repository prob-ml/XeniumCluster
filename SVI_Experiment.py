import subprocess

# List of argument sets
argument_sets = [
    [
        {
            "custom_init": custom_init, 
            "neighborhood_size": neighborhood_size, 
            "num_clusters": 17, 
            "spatial_init": False, 
            "spatial_normalize": 0.0, 
            "concentration_amplification": concentration_amplification, 
            "spot_size": 50, 
            "num_pcs": num_pcs, 
            "data_mode": "PCA", 
            "hvg_var_prop": 0.9, 
            "neighborhood_agg": neighborhood_agg
        }
        for custom_init in ["mclust"]
        for concentration_amplification in [1.0]
        for neighborhood_agg in ["sum", "mean"]
    ]
    for neighborhood_size in range(1, 6)
    for num_pcs in [3]
]

# Function to format arguments into a command string
def format_command(args):
    return (
        f"python SVI.py "
        f"--custom_init={args['custom_init']} "
        f"--neighborhood_size={args['neighborhood_size']} "
        f"--num_clusters={args['num_clusters']} "
        f"--spatial_init={args['spatial_init']} "
        f"--spatial_normalize={args['spatial_normalize']} "
        f"--concentration_amplification={args['concentration_amplification']} "
        f"--spot_size={args['spot_size']} "
        f"--data_mode={args['data_mode']} "
        f"--num_pcs={args['num_pcs']} "
        f"--hvg_var_prop={args['hvg_var_prop']} "
        f"--neighborhood_agg={args['neighborhood_agg']}"
    )

# Loop through each argument set and create tmux sessions
for argument_set in argument_sets:
    # Get the common arguments (neighborhood_size, num_pcs)
    neighborhood_size = argument_set[0]['neighborhood_size']
    num_pcs = argument_set[0]['num_pcs']
    session_name = f"n{neighborhood_size}PCs{num_pcs}"  # Unique tmux session name

    # Create a list of commands to run within the tmux session
    commands = [format_command(args) for args in argument_set]
    
    # Combine all commands into a single command string, joined by `&&` to run sequentially
    full_command = " && ".join(commands)
    
    # Run the full command in a new detached tmux session
    tmux_command = ["tmux", "new-session", "-d", "-s", session_name, full_command]
    
    # Start the tmux session with the set of commands
    subprocess.run(tmux_command)

    print(f"Started tmux session {session_name} with {len(commands)} commands.")
