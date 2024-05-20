# %% [markdown]
# [ ] Fix the latent variance calculations. (I think it's just taking total variance instead of the 2 norm of z vectors.)

# %%
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
import torch.nn.functional as F
import optuna
import tensorboard
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings("ignore")
from importlib import reload

import torchclustermetrics
reload(torchclustermetrics)
from torchclustermetrics import silhouette

# this ensures that I can update the class without losing my variables in my notebook
import xenium_cluster
reload(xenium_cluster)
from xenium_cluster import XeniumCluster
from utils.metrics import *

# %%
# Path to your .gz file
file_path = 'data/hBreast/transcripts.csv.gz'

# Read the gzipped CSV file into a DataFrame
df_transcripts = pd.read_csv(file_path, compression='gzip')
df_transcripts.head()

# %%
clustering = XeniumCluster(data=df_transcripts, dataset_name="hBreast")
clustering.set_spot_size(100)
clustering.create_spot_data(third_dim=False, save_data=True)

# %%
valid_genes_mask = ~clustering.xenium_spot_data.var_names.str.startswith('BLANK_') & ~clustering.xenium_spot_data.var_names.str.startswith('NegControl')
clustering.xenium_spot_data = clustering.xenium_spot_data[:, valid_genes_mask]

# %%
HIGHLY_VARIABLE = False
clustering.normalize_counts(clustering.xenium_spot_data)

# %%
HIGHLY_VARIABLE = True
NUM_GENES = 25
# generate the neigborhood graph based on pca
sc.tl.pca(clustering.xenium_spot_data, svd_solver='arpack')
sc.pl.pca_variance_ratio(clustering.xenium_spot_data)
if HIGHLY_VARIABLE:
    if NUM_GENES > 0:
        clustering.filter_only_high_variable_genes(clustering.xenium_spot_data, n_top_genes=NUM_GENES)
    else:
        clustering.filter_only_high_variable_genes(clustering.xenium_spot_data)
    clustering.xenium_spot_data = clustering.xenium_spot_data[:,clustering.xenium_spot_data.var.highly_variable==True]

# %%
clustering.xenium_spot_data.var, clustering.xenium_spot_data.var.shape

# %%
clustering.xenium_spot_data.obs, clustering.xenium_spot_data.obs.shape

# %%
expression_data = pd.DataFrame(clustering.xenium_spot_data.X, columns=clustering.xenium_spot_data.var.index)
clustering.xenium_spot_data.obs.index = clustering.xenium_spot_data.obs.index.astype(int)
input_data = clustering.xenium_spot_data.obs.join(expression_data)
input_data.head()

# %% [markdown]
# ### Reconstruction Loss
# 
# Input: $X$
# 
# Reconstruction: $X^{'}$
# 
# $$(X - X^{'})^2$$
# 
# ### Spatial Loss
# 
# Left Boundary $(l)$: $\max(0, j - n)$
# 
# Right Boundary $(r)$: $\min(0, j + n)$
# 
# Top Boundary $(t)$: $\max(0, i - n)$
# 
# Bottom Boundary $(b)$: $\min(0, i + n)$
# 
# $$\frac{1}{IJ} \sum_{i=1}^I \sum_{j=1}^{J} \frac{1}{(r-l+1)(b-t+1)}\sum_{i^{'} = b}^{t} \sum_{j^{'}=l}^{r} D_{KL}(C[i^{'}, j^{'}], C[i, j]) $$
# 
# ### Entropy Regularization
# 
# $$- \frac{1}{IJK} \sum_{i=1}^{I}\sum_{j=1}^{J}\sum_{k=1}^{K} p_{i,j}(k) \log(p_{i,j}(k)) $$

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "[1]"


# %%
class GumbelSoftmax(torch.nn.Module):
    def __init__(self, dim, temperature=1.0):
        super(GumbelSoftmax, self).__init__()
        self.dim = dim
        self.temperature = temperature

    def forward(self, logits):
        return F.gumbel_softmax(logits, tau=self.temperature, dim=self.dim)

    def extra_repr(self):
        return f'temperature={self.temperature}'

# %%
class ClassifierAutoEncoder(L.LightningModule):

    def __init__(
            self,
            spot_height,
            spot_width,
            spot_depth = None,
            within_cluster_distance_hyperparam = 1,
            within_cluster_penalty_hyperparam = 1, 
            spatial_penalty_hyperparam = 1,
            entropy_penalty_hyperparam = 1,
            cluster_util_penalty_hyperparam = 1,
            neighborhood_size = 5,
            num_clusters = 10,
            latent_dim = 10
        ):

        self.within_cluster_distance_hyperparam = within_cluster_distance_hyperparam
        self.within_cluster_penalty_hyperparam = within_cluster_penalty_hyperparam
        self.spatial_penalty_hyperparam = spatial_penalty_hyperparam
        self.entropy_penalty_hyperparam = entropy_penalty_hyperparam
        self.cluster_util_penalty_hyperparam = cluster_util_penalty_hyperparam
        self.neighborhood_size = neighborhood_size
        self.num_clusters = num_clusters
        self.input_size = NUM_GENES if HIGHLY_VARIABLE else 541
        self.kernel_size = 5
        self.stride = 2
        self.padding = 0
        self.spot_height = spot_height
        self.spot_width = spot_width
        self.spot_depth = spot_depth
        self.latent_dim = latent_dim

        self.loss_array = []
        self.reconstruction = []
        self.wcd = []
        self.wcl = []

        super(ClassifierAutoEncoder, self).__init__()

        # Encoder Layers
        self.enc_conv1 = torch.nn.Conv2d(self.input_size, 256, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        self.enc_batchnorm1 = torch.nn.BatchNorm2d(256)
        self.enc_conv2 = torch.nn.Conv2d(256, 256, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        self.enc_batchnorm2 = torch.nn.BatchNorm2d(256)
        self.enc_conv3 = torch.nn.Conv2d(256, self.latent_dim, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        self.enc_batchnorm3 = torch.nn.BatchNorm2d(self.latent_dim)
        self.enc_pool = torch.nn.AdaptiveAvgPool2d((25, 25))

        # Decoder Layers
        self.dec_conv1 = torch.nn.ConvTranspose2d(self.latent_dim, 256, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        self.dec_batchnorm1 = torch.nn.BatchNorm2d(256)
        self.dec_conv2 = torch.nn.ConvTranspose2d(256, 256, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        self.dec_batchnorm2 = torch.nn.BatchNorm2d(256)
        self.dec_conv3 = torch.nn.ConvTranspose2d(256, self.input_size, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        self.dec_batchnorm3 = torch.nn.BatchNorm2d(self.input_size)

        self.activation = torch.nn.LeakyReLU()

        # Bridging Layers
        self.upsample = torch.nn.Upsample((self.spot_height, self.spot_width), mode='bilinear', align_corners=False)

        # Cluster Layers
        self.clust_linear1 = torch.nn.Linear(self.latent_dim, 256)
        self.clust_linear2 = torch.nn.Linear(256, 128)
        self.clust_linear3 = torch.nn.Linear(128, self.num_clusters)
        self.clust_softmax = GumbelSoftmax(dim=1)

        self.encoder = torch.nn.Sequential(
            self.enc_conv1,
            self.enc_batchnorm1,
            self.activation,
            self.enc_conv2, 
            self.enc_batchnorm2,   
            self.activation,       
            self.enc_conv3,  
            self.enc_batchnorm3,
            self.enc_pool    
        )

        self.decoder = torch.nn.Sequential(
            self.dec_conv1,
            self.dec_batchnorm1,
            self.activation,
            self.dec_conv2, 
            self.dec_batchnorm2,
            self.activation,
            self.dec_conv3, 
            self.dec_batchnorm3,
        )

        self.cluster_assignment = torch.nn.Sequential(
            self.clust_linear1,
            self.activation,
            self.clust_linear2,
            self.activation,
            self.clust_linear3,
            self.clust_softmax 
        )

    def training_mask(self, x):

        batch, channels, height, width = x.shape

        zeros = x.view(channels, height, width)
        zeros = (zeros.sum(dim=0) != 0)
        zeros = torch.tensor(zeros, dtype=int)

        return zeros
        
    def training_step(self, batch, batch_idx):

        x, x_hat, z, cluster = self(batch)

        try:
            zeros = self.training_mask(x).unsqueeze(0).unsqueeze(0)

            # Apply mask to input and reconstruction
            x_masked = x * zeros
            x_hat_masked = x_hat * zeros
            z_masked = z * torch.flatten(zeros).view(-1,1)
            cluster_masked = cluster * torch.flatten(zeros).view(-1,1)
        except:
            # Apply mask to input and reconstruction
            x_masked = x
            x_hat_masked = x_hat
            z_masked = z
            cluster_masked = cluster
            print("The masking procedure did not work.")

        loss = self.spatial_loss_function(x_masked, x_hat_masked, z_masked, cluster_masked)

        formatted_loss = round(loss.item(), 4)
        print(formatted_loss)
        self.loss_array.append(formatted_loss)
        print(f"# of Clusters: {cluster_masked.argmax(dim=1).unique().numel()}")

        if self.current_epoch % 10 == 0:

            num_possible_clusters = cluster_masked.shape[1] + 1

            # Get unique values and create a colormap
            batch, channels, height, width = x.shape
            zeros = x.view(channels, height, width)
            zeros = (zeros.sum(dim=0) != 0)
            zeros = torch.tensor(zeros, dtype=int)
            clusters = cluster_masked.argmax(dim=1).view(height, width)
            clusters = (clusters + 1) * zeros
            clusters_np = clusters.numpy()

            colors = plt.cm.get_cmap('viridis', num_possible_clusters)
            colormap = ListedColormap(colors(np.linspace(0, 1, num_possible_clusters)))

            # Plotting
            plt.figure(figsize=(6, 6))
            plt.imshow(clusters_np, cmap=colormap, interpolation='nearest', origin='lower')
            plt.colorbar(ticks=range(num_possible_clusters), label='Cluster Values')
            plt.title(f'Cluster Visualization at Epoch #{self.current_epoch}')
            plt.savefig(f'results/hBreast/XeniumCluster/{self.current_epoch}.png')


        return loss

    def validation_step(self, batch, batch_idx):
        x, x_hat, z, cluster = self(batch)
        try:
            zeros = self.training_mask(x).unsqueeze(0).unsqueeze(0)

            # Apply mask to input and reconstruction
            x_masked = x * zeros
            x_hat_masked = x_hat * zeros
            z_masked = z * torch.flatten(zeros).view(-1,1)
            cluster_masked = cluster * torch.flatten(zeros).view(-1,1)
        except:
            # Apply mask to input and reconstruction
            x_masked = x
            x_hat_masked = x_hat
            z_masked = z
            cluster_masked = cluster
            print("The masking procedure did not work.")

        loss = self.spatial_loss_function(x_masked, x_hat_masked, z_masked, cluster_masked)

        return loss

    def test_step(self, batch, batch_idx):
        x, x_hat, z, cluster = self(batch)
        try:
            zeros = self.training_mask(x).unsqueeze(0).unsqueeze(0)

            # Apply mask to input and reconstruction
            x_masked = x * zeros
            x_hat_masked = x_hat * zeros
            z_masked = z * torch.flatten(zeros).view(-1,1)
            cluster_masked = cluster * torch.flatten(zeros).view(-1,1)
        except:
            # Apply mask to input and reconstruction
            x_masked = x
            x_hat_masked = x_hat
            z_masked = z
            cluster_masked = cluster
            print("The masking procedure did not work.")

        loss = self.spatial_loss_function(x_masked, x_hat_masked, z_masked, cluster_masked)

        return loss


    def spatial_loss_function(self, input, reconstruction, latents, cluster_probabilities):
        reconstruction_loss = F.mse_loss(input, reconstruction)
        within_cluster_penalty = self.within_cluster_average_latent_dissimilarity(input, latents, cluster_probabilities)
        spatial_cluster_penalty = self.neighboring_cluster_dissimilarity(cluster_probabilities, input)
        within_cluster_distance = self.within_cluster_distance(cluster_probabilities, input)
        print(f"""LOSS CONTRIBUTIONS: 
              (Reconstruct, Within, Within Distance, Spatial)
              ({reconstruction_loss}, {self.within_cluster_penalty_hyperparam * within_cluster_penalty}, {self.within_cluster_distance_hyperparam * within_cluster_distance}, {self.spatial_penalty_hyperparam * spatial_cluster_penalty})
              """)
        self.reconstruction.append(reconstruction_loss.item())
        self.wcd.append(within_cluster_distance.item())
        self.wcl.append(within_cluster_penalty.item())
        # return self.within_cluster_distance_hyperparam * within_cluster_distance
        return reconstruction_loss + self.within_cluster_penalty_hyperparam * within_cluster_penalty + self.within_cluster_distance_hyperparam * within_cluster_distance

    @staticmethod
    def augment(latent_value, sd=0.2):
        
        if isinstance(sd, (int, float)):
            latent_size = latent_value.shape
            return latent_value + torch.normal(0.0, sd, size=latent_size)
        else:
            return latent_value + torch.normal(0.0, sd)

    def within_cluster_average_latent_dissimilarity(self, x, z, cluster_probabilities, margin=5.0, samples_per_cluster=20, min_utilization_penalty=1.0):
        
        # total_loss = dissimilarity_loss
        # Utilization penalty: Soft minimum average probability per cluster
        average_cluster_probs = cluster_probabilities.mean(dim=0)  # Mean probability assigned to each cluster across all data points
        utilization_penalty = -torch.log(average_cluster_probs.clamp(min=1e-6)).mean()  # Soft penalty using negative log

        # Initialize triplet loss function
        triplet_loss = torch.nn.TripletMarginLoss(margin=margin)

        # Find unique clusters
        cluster_assignments = cluster_probabilities.argmax(dim=1)
        cluster_assignments = (cluster_assignments + 1) * (cluster_probabilities.max(dim=1) != 0)
        unique_clusters = cluster_assignments.unique()

        # Preparing lists to hold anchor, positive, and negative examples
        anchor_list = []
        positive_list = []
        negative_list = []

        # Loop through each cluster to select anchor and positive, and find negative from other clusters
        for cluster in unique_clusters:

            for _ in range(samples_per_cluster):

                # Indices for current cluster and other clusters
                current_cluster_indices = (cluster_assignments == cluster).nonzero().squeeze()

                anchor_index = torch.randperm(len(current_cluster_indices))[0]
                cluster_anchor_index = current_cluster_indices[anchor_index]
                anchor_example = z[cluster_anchor_index]

                # Check if there are enough samples
                for other_cluster in unique_clusters:
                    if cluster != other_cluster:
                        other_cluster_indices = (cluster_assignments == cluster).nonzero().squeeze()
                        for _ in range(samples_per_cluster):
                            if current_cluster_indices.numel() > 1 and other_cluster_indices.numel() > 1:
                                
                                # Randomly choose one sample to be the anchor and another to be the positive sample
                                positive_example = self.augment(anchor_example, sd=torch.abs(anchor_example / 10.0))

                                # Randomly choose one sample from another cluster to be the negative sample
                                negative_index = other_cluster_indices[torch.randperm(len(other_cluster_indices))[0]]

                                # Add to lists
                                anchor_list.append(anchor_example)
                                positive_list.append(positive_example)
                                negative_list.append(z[negative_index])

        # Stack lists to create tensors for the triplet loss calculation
        if anchor_list:
            anchor_tensor = torch.stack(anchor_list)
            positive_tensor = torch.stack(positive_list)
            negative_tensor = torch.stack(negative_list)

            # Calculate the triplet loss
            triplet_loss = triplet_loss(anchor_tensor, positive_tensor, negative_tensor)


        print(f"DISSIMILARITY, PENALTY - {triplet_loss}, {min_utilization_penalty * utilization_penalty}")
        total_loss = triplet_loss + min_utilization_penalty * utilization_penalty

        return total_loss


    def within_cluster_distance(self, cluster_probabilities, x):
        """
        Calculates a clustering loss which minimizes intra-cluster distance and maximizes inter-cluster distance.
        
        Parameters:
        - cluster_probabilities (torch.Tensor): Tensor of shape (num_spots, num_clusters)
            representing the probability of each data point belonging to each cluster.
        - x (torch.Tensor): Tensor of data points of shape (batch, channels, height, width).
        
        Returns:
        - torch.Tensor: The computed loss.
        """

        # Assuming spatial coordinates can be derived from the last two dimensions (height, width)
        zeros = self.training_mask(x)
        ignored_clusters = torch.nonzero(zeros == 0)
        _, _, H, W = x.shape
        spatial_coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), dim=-1)
        spatial_coords = spatial_coords.float().to(x.device)  # Convert coordinates to float and send to device
        spatial_coords_flat = spatial_coords.view(-1, 2)  # Flatten the coordinates

        original_shape = spatial_coords_flat.shape[0]

        # Creating a mask to filter spatial_coords_flat
        # We will check for each element in spatial_coords_flat if it exists in ignored_clusters
        mask = torch.ones(spatial_coords_flat.shape[0], dtype=torch.bool, device=x.device)  # Start with all True
        for coord in ignored_clusters:
            mask *= ~(spatial_coords_flat == coord).all(dim=1)  # Turn mask False for matching coordinates

        # Apply mask
        spatial_coords_flat = spatial_coords_flat[mask]
        cluster_probabilities = cluster_probabilities[mask]

        # Calculate centroids of clusters in spatial terms
        weighted_spatial_sum = torch.mm(cluster_probabilities.t(), spatial_coords_flat)
        cluster_sizes = cluster_probabilities.sum(dim=0, keepdim=True).t()
        spatial_centroids = weighted_spatial_sum / cluster_sizes

        # Intra-cluster variance (minimize)
        expanded_probs = cluster_probabilities.unsqueeze(2)
        expanded_spatial_centroids = spatial_centroids.unsqueeze(0)
        dists_to_spatial_centroids = torch.norm(spatial_coords_flat.unsqueeze(1) - expanded_spatial_centroids, dim=2, p=2)
        intra_cluster_variance = torch.mean(expanded_probs.squeeze() * (dists_to_spatial_centroids ** 2))

        # Inter-cluster variance (maximize)
        spatial_centroid_dists = torch.pdist(spatial_centroids, p=2)
        inter_cluster_variance = spatial_centroid_dists.mean()

        # Combine the losses
        print(f"INTRA, INTER - {intra_cluster_variance}, {inter_cluster_variance}")
        loss = intra_cluster_variance + intra_cluster_variance / inter_cluster_variance + 1.0 / inter_cluster_variance

        return loss

    
    def neighboring_cluster_dissimilarity(self, cluster, x, zero_correction = 1e-9):
        batch, channels, height, width = x.shape
        spots, num_clusters = cluster.shape
        spatial_cluster = cluster.view(height, width, num_clusters)
        dissimilarity_values = []
        for i in range(height):
            for j in range(width):

                left_boundary = max(0, j - self.neighborhood_size)
                right_boundary = min(width, j + self.neighborhood_size + 1)
                top_boundary = max(0, i - self.neighborhood_size)
                bottom_boundary = min(height, i + self.neighborhood_size + 1)
                neighborhood = spatial_cluster[top_boundary:bottom_boundary, left_boundary:right_boundary]

                central_pixel_distr = spatial_cluster[i, j, :].unsqueeze(0).unsqueeze(0)
                kl_divergences = F.kl_div((central_pixel_distr + zero_correction).log(), neighborhood + zero_correction, reduction='none', log_target=False)
                dissimilarity = kl_divergences.sum(dim=-1).mean(dim=(0, 1))

                dissimilarity_values.append(dissimilarity.unsqueeze(0))

        dissimilarities = torch.cat(dissimilarity_values).to(x.device)
        return torch.mean(dissimilarities)  
    
    def entropy_regularization(self, cluster_probs, x, zero_correction = 1e-9):
        batch, channels, height, width = x.shape
        spots, num_clusters = cluster_probs.shape
        spatial_cluster = cluster_probs.view(height, width, num_clusters)
        entropy = -torch.sum(spatial_cluster * torch.log(spatial_cluster + zero_correction), dim=2)
        entropy_loss = torch.mean(entropy)
        return entropy_loss

    def forward(self, x):

        batch_size, _, _, _ = x.shape

        z = self.encoder(x)
        # FIX THIS TO BE FOR SEVERAL BATCHES
        x_hat = self.upsample(self.decoder(z))
        z = self.upsample(z)
        z = z.view(batch_size * self.spot_height * self.spot_width, self.latent_dim)
        cluster = self.cluster_assignment(z)

        return x, x_hat, z, cluster

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
        return optimizer

# %%
gene_data = []

for gene in clustering.xenium_spot_data.var.index:
    gene_channel = pd.pivot(input_data, index = 'row', columns = 'col', values = f"{gene}").fillna(0)
    gene_tensor = torch.tensor(gene_channel.to_numpy())
    gene_data.append(gene_tensor)

input_tensor = torch.stack(gene_data, dim=0)
input_tensor.shape

# %%
channels, spot_height, spot_width = input_tensor.shape
input_tensor = input_tensor.float()
input_tensor.to(dtype=torch.float32)
print(input_tensor.shape)
dataset = [input_tensor]
dataloader = DataLoader(dataset, batch_size=1)

model = ClassifierAutoEncoder(spot_height, spot_width, num_clusters=10, latent_dim=20, entropy_penalty_hyperparam = 1, within_cluster_distance_hyperparam = 0.05, within_cluster_penalty_hyperparam = 1.0, spatial_penalty_hyperparam = 1.0, cluster_util_penalty_hyperparam = 1, neighborhood_size = 4)

# %%
import datetime

# Get the current date and time
current_datetime = datetime.datetime.now()

# Format the date and time
formatted_datetime = current_datetime.strftime('%Y/%m/%d %H:%M:%S')

logger = TensorBoardLogger(save_dir=f"{os.getcwd()}/lightning_logs", name=f"{formatted_datetime}")
# Create a checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='train_loss',    # or another metric that you're tracking
    dirpath='my_model/',   # directory to save the model
    filename='best_model', # filename for saving
    save_top_k=1,          # save only the best model
    mode='min',            # `min` if monitoring metric is expected to decrease (e.g., loss)
    save_weights_only=True # set False to save the whole model
)

trainer = L.Trainer(max_epochs=1500, callbacks=[checkpoint_callback])
trainer.fit(model=model, train_dataloaders=dataloader)

# %%
plt.plot(range(len(model.loss_array)), model.loss_array)

# %%
plt.plot(range(len(model.reconstruction)), model.reconstruction)

# %%
plt.plot(range(len(model.wcd)), model.wcd)

# %%
plt.plot(range(len(model.wcl)), model.wcl)

# %%
predictions = trainer.predict(model = model, dataloaders = dataloader)

# %%
# Plot the tensor
inputs, reconstruction, latents, clusters = predictions[0]
clusters

# %%
clusters.argmax(dim=1)
_ = plt.hist(clusters.argmax(dim=1), bins=range(1, 21))

# %%
inputs.shape

# %%
# Plot the tensor
inputs, reconstruction, latents, clusters = predictions[0]
num_possible_clusters = clusters.shape[1]

# Get unique values and create a colormap
batch, channels, height, width = inputs.shape
clusters = clusters.argmax(dim=1).view(height, width)
clusters_np = clusters.numpy()

colors = plt.cm.get_cmap('viridis', num_possible_clusters)
colormap = ListedColormap(colors(np.linspace(0, 1, num_possible_clusters)))

# Plotting
plt.figure(figsize=(6, 6))
plt.imshow(clusters_np, cmap=colormap, interpolation='nearest', origin='lower')
plt.colorbar(ticks=range(num_possible_clusters), label='Cluster Values')
plt.title('Cluster Visualization with Unique Colors')
plt.show()

# %%
zeros = inputs.view(channels, height, width)
zeros = (zeros.sum(dim=0) != 0)
zeros = torch.tensor(zeros, dtype=int)

plt.imshow(zeros, interpolation='nearest', origin='lower')

# %%
# Plot the tensor
inputs, reconstruction, latents, clusters = predictions[0]
num_possible_clusters = clusters.shape[1] + 1

# Get unique values and create a colormap
batch, channels, height, width = inputs.shape
zeros = inputs.view(channels, height, width)
zeros = (zeros.sum(dim=0) != 0)
zeros = torch.tensor(zeros, dtype=int)
clusters = clusters.argmax(dim=1).view(height, width)
clusters = (clusters + 1) * zeros
clusters_np = clusters.numpy()

colors = plt.cm.get_cmap('viridis', num_possible_clusters)
colormap = ListedColormap(colors(np.linspace(0, 1, num_possible_clusters)))

# Plotting
plt.figure(figsize=(6, 6))
plt.imshow(clusters_np, cmap=colormap, interpolation='nearest', origin='lower')
plt.colorbar(ticks=range(num_possible_clusters), label='Cluster Values')
plt.title('Cluster Visualization with Unique Colors')
plt.savefig("results/hBreast/XeniumCluster/cluster.png")
plt.show()

# %%
vis_inputs = inputs.squeeze(0).sum(dim=0)

# Setup a figure for subplots
plt.figure(figsize=(12, 6))  # Adjust the figure size as necessary

# First subplot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.imshow(vis_inputs, cmap=colormap, interpolation='nearest')
plt.colorbar(ticks=range(num_possible_clusters), label='Cluster Values')
plt.title('Input Values')

vis_reconstruction = reconstruction.squeeze(0).sum(dim=0)

plt.subplot(1, 2, 2)
plt.imshow(vis_reconstruction, cmap=colormap, interpolation='nearest')
plt.colorbar(ticks=range(num_possible_clusters), label='Cluster Values')
plt.title('Reconstruction Values')

# %% [markdown]
# Try: Contrastive Loss
# 
# Include: Post-Processing Clean Up (like Graph Cut or Markov Random Fields)

# %% [markdown]
# 1. Neighborhood Size and Spatial Penalty
#      - Conditional Random Fields
# 
# 4. Data Preprocessing and Batch Effects
# Make sure that the input data is appropriately preprocessed and normalized. Variations in data scaling or the presence of outliers can adversely affect clustering performance.
# 
# Normalization and Standardization: Ensure that your data input to the model is well-preprocessed, including tasks like normalization or standardization, especially if dealing with image data.
# Batch Effects: Sometimes, how data is batched during training can affect learning dynamics, especially in cases where spatial relationships are important. Make sure the batching process does not obscure spatial relationships.
# 

# %% [markdown]
# CRFs

# %% [markdown]
#  Use Strided Convolutions for Downsampling
# Instead of traditional pooling layers, you can use strided convolutions to reduce spatial dimensions. This method effectively reduces the height and width while increasing the depth (number of channels), which compensates for the reduced spatial resolution by capturing more detailed feature information in the channel dimension.
# 
# Upsampling in Decoder: In the decoder, you can use transposed convolutions (often called deconvolutions) with appropriate stride to upsample the spatial dimensions back to the original size.
# 2. Incorporate Dilated Convolutions
# Dilated convolutions allow you to expand the receptive field without reducing the spatial dimension of the output. This technique can be particularly useful if you want to capture broader context without losing resolution.
# 
# 3. Max-Pooling with Indices (Max-Unpooling)
# A specific technique that can be helpful is max-pooling with indices, often used in segmentation tasks:
# 
# Max-Pooling with Indices: This type of pooling keeps track of the positions of the maxima within each pooling window. This information is then used during the upsampling phase to place the values back into their original locations, allowing for precise reconstruction.
# Max-Unpooling: In the decoder, use the stored indices to perform max-unpooling, which restores the data to its original dimensions by placing the max values back into their recorded positions, while other values are typically set to zero.
# 4. PixelShuffle for Upsampling
# The PixelShuffle operation is an efficient and simple way to increase the resolution of an image or feature map. It works by rearranging elements from a low-resolution input into a high-resolution output, by reshuffling the tensor dimensions:
# 
# Increase Channels Instead of Downsampling: Modify the encoder to increase the number of channels rather than reducing the spatial dimensions. Then, in the decoder, use PixelShuffle to reduce the channel depth while increasing the spatial dimensions back to the original.
# 5. Bottleneck Layers
# Consider using bottleneck layers that first increase the channel dimensions (deepen) and then reduce them back:
# 
# Implementation: Use 1x1 convolutions to expand the channel dimensions followed by 3x3 convolutions to process features, then another 1x1 convolution to compress the channels back. This method helps in capturing complex features without changing the spatial dimensions.
# 6. Using Convolutional Layers with Padding and Custom Dilation
# Use a combination of custom dilation and padding settings to adjust how the field of view of each convolutional operation is spread across the input:
# 
# Custom Settings: By adjusting dilation and padding, you can manage to capture more contextual information without necessarily having to pool and thus reduce the spatial dimensions.


