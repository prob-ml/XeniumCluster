import itertools
import os
import subprocess

# Define your parameter grid
parameter_grid = {
    "custom_init": ["mclust", "K-Means", "Leiden"],
    "neighborhood_size": range(1, 5),
    "num_clusters": [17],
    "concentration_amplification": [1.0, 3.0, 5.0],
    "spot_size": [50],
    "num_pcs": [3, 5, 10, 15, 25],
    "data_mode": ["PCA"],
    "hvg_var_prop": [0.9],
    "neighborhood_agg": ["mean"],
    "mu_prior_scale": [0.1, 1.0, 2.5],
    "sigma_prior_scale": [0.1, 1.0],
    "mu_q_scale": [0.1, 1.0, 2.5],
    "sigma_q_scale": [0.1, 1.0],
}

# Function to format arguments into a command string
def format_command(args):
    return (
        f"python SVI.py "
        f"--custom_init={args['custom_init']} "
        f"--neighborhood_size={args['neighborhood_size']} "
        f"--num_clusters={args['num_clusters']} "
        f"--concentration_amplification={args['concentration_amplification']} "
        f"--spot_size={args['spot_size']} "
        f"--data_mode={args['data_mode']} "
        f"--num_pcs={args['num_pcs']} "
        f"--hvg_var_prop={args['hvg_var_prop']} "
        f"--neighborhood_agg={args['neighborhood_agg']} "
        f"--mu_prior_scale={args['mu_prior_scale']} "
        f"--sigma_prior_scale={args['sigma_prior_scale']} "
        f"--mu_q_scale={args['mu_q_scale']} "
        f"--sigma_q_scale={args['sigma_q_scale']}"
    )

# Generate all combinations of parameters
params = parameter_grid.items()
keys, values = zip(*params)
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Generate the command for each combination
commands = [format_command(args) for args in combinations]

# Set the maximum number of tmux sessions and calculate batch size
num_sessions = 50  # Adjust based on your requirements
batch_size = len(commands) // num_sessions + 1  # Ensure all commands are included

# Split commands into batches
batches = [commands[i:i + batch_size] for i in range(0, len(commands), batch_size)]

# Create a directory to store the shell scripts
os.makedirs('scripts', exist_ok=True)

# Loop through each batch and create tmux sessions
for idx, batch in enumerate(batches):
    script_name = f'scripts/batch_{idx}.sh'
    # Write commands to the shell script
    with open(script_name, 'w') as f:
        f.write('#!/bin/bash\n')
        for cmd in batch:
            f.write(cmd + '\n')
    # Make the script executable
    os.chmod(script_name, 0o755)

    session_name = f"batch_{idx}"

    # Start tmux session that runs the script
    tmux_command = ["tmux", "new-session", "-d", "-s", session_name, f"bash {script_name}"]

    subprocess.run(tmux_command)

    print(f"Started tmux session {session_name} running {len(batch)} commands")
