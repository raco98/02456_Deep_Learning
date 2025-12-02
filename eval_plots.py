import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from sklearn.decomposition import PCA
from IPython.display import display, clear_output

def plot_training_curves(training_data, validation_data):
    """
    Plots the training and validation curves for ELBO, Log Likelihood, and KL Divergence.
    
    Args:
        training_data (dict): Dictionary containing lists of training metrics.
        validation_data (dict): Dictionary containing lists of validation metrics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ELBO
    if 'elbo' in training_data:
        axes[0].plot(training_data['elbo'], label='Train')
        if 'elbo' in validation_data:
            axes[0].plot(validation_data['elbo'], label='Validation')
        axes[0].set_title('ELBO')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('ELBO')
        axes[0].legend()
        axes[0].grid(True)
    
    # Log Likelihood (Reconstruction)
    if 'log_px' in training_data:
        axes[1].plot(training_data['log_px'], label='Train')
        if 'log_px' in validation_data:
            axes[1].plot(validation_data['log_px'], label='Validation')
        axes[1].set_title('Log Likelihood (Reconstruction)')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True)
    
    # KL Divergence
    if 'kl' in training_data:
        axes[2].plot(training_data['kl'], label='Train')
        if 'kl' in validation_data:
            axes[2].plot(validation_data['kl'], label='Validation')
        axes[2].set_title('KL Divergence')
        axes[2].set_xlabel('Epoch')
        axes[2].legend()
        axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()


def visualize_reconstructions_live(model, data_loader, device, epoch: int, num_samples: int = 3, cmap: str = 'jet', transpose: bool = True):
    """
    Live update of original images and reconstructions for notebook training.

    Intended to be called from the training loop every N epochs (e.g., if epoch % 10 == 0).

    Args:
        model (nn.Module): trained or training VAE model.
        data_loader (DataLoader): DataLoader for validation/test set.
        device (torch.device): device to run model on.
        epoch (int): current epoch (used for title).
        num_samples (int): number of examples to show side-by-side.
        cmap (str): matplotlib colormap to use (e.g., 'jet').
        transpose (bool): whether to apply the same transpose used elsewhere to match orientation.
    """
    model.eval()
    try:
        batch = next(iter(data_loader))
    except StopIteration:
        return

    x = batch[0] if isinstance(batch, (list, tuple)) else batch
    x = x[:num_samples].to(device)

    with torch.no_grad():
        outputs = model(x)
        px = outputs['px']
        x_hat = px.mean

    x_np = x.cpu().numpy()
    x_hat_np = x_hat.cpu().numpy()

    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
    if num_samples == 1:
        axes = axes.reshape(2, 1)

    for i in range(num_samples):
        img_orig = x_np[i, 0]
        if transpose:
            img_orig = img_orig.T[::-1, :]

        axes[0, i].imshow(img_orig, cmap=cmap, vmin=0.0, vmax=1.0)
        axes[0, i].set_title(f"Orig #{i}")
        axes[0, i].axis('off')

        img_recon = x_hat_np[i, 0]
        if transpose:
            img_recon = img_recon.T[::-1, :]

        axes[1, i].imshow(img_recon, cmap=cmap, vmin=0.0, vmax=1.0)
        axes[1, i].set_title(f"Recon #{i}")
        axes[1, i].axis('off')

    fig.suptitle(f"Epoch {epoch}: Original (top) vs Reconstruction (bottom)")
    plt.tight_layout()

    # Clear previous output and display the new figure (works in notebooks)
    clear_output(wait=True)
    display(fig)
    plt.close(fig)

def visualize_reconstructions(model, data_loader, device, num_samples=5, transpose=True):
    """
    Visualizes original images and their reconstructions from the VAE.
    
    Args:
        model (nn.Module): The trained VAE model.
        data_loader (DataLoader): DataLoader for the test/validation set.
        device (torch.device): Device to run the model on.
        num_samples (int): Number of samples to visualize.
        transpose (bool): If True, applies .T[::-1, :] to match BOUT++ data orientation.
    """
    model.eval()
    try:
        x = next(iter(data_loader))
    except StopIteration:
        return
        
    x = x[:num_samples].to(device)
    
    with torch.no_grad():
        outputs = model(x)
        px = outputs['px']
        # Use the mean of the distribution as the reconstruction
        x_hat = px.mean
    
    x = x.cpu().numpy()
    x_hat = x_hat.cpu().numpy()
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
    
    # Handle case where num_samples is 1 (axes is 1D array)
    if num_samples == 1:
        axes = axes.reshape(2, 1)
        
    for i in range(num_samples):
        # Original
        img_orig = x[i, 0] # Assuming shape [B, C, H, W] and taking first channel
        if transpose:
            img_orig = img_orig.T[::-1, :]
            
        axes[0, i].imshow(img_orig, cmap='viridis')
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        
        # Reconstruction
        img_recon = x_hat[i, 0]
        if transpose:
            img_recon = img_recon.T[::-1, :]
            
        axes[1, i].imshow(img_recon, cmap='viridis')
        axes[1, i].set_title("Reconstruction")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.show()

def visualize_latent_space(model, data_loader, device):
    """
    Visualizes the latent space. If latent_dim > 2, uses PCA to project to 2D.
    
    Args:
        model (nn.Module): The trained VAE model.
        data_loader (DataLoader): DataLoader for the test/validation set.
        device (torch.device): Device to run the model on.
    """
    model.eval()
    mus = []
    
    with torch.no_grad():
        for x in data_loader:
            x = x.to(device)
            qz = model.posterior(x)
            mus.append(qz.mu.cpu())
            
    mus = torch.cat(mus, dim=0).numpy()
    
    plt.figure(figsize=(8, 6))
    
    if mus.shape[1] == 2:
        plt.scatter(mus[:, 0], mus[:, 1], alpha=0.6, s=15, c='b', edgecolors='k', linewidth=0.5)
        plt.xlabel('Latent Dim 1')
        plt.ylabel('Latent Dim 2')
        plt.title('Latent Space Visualization')
    else:
        # PCA projection if dim > 2
        pca = PCA(n_components=2)
        mus_pca = pca.fit_transform(mus)
        plt.scatter(mus_pca[:, 0], mus_pca[:, 1], alpha=0.6, s=15, c='b', edgecolors='k', linewidth=0.5)
        plt.xlabel(f'PC 1 (Expl. Var: {pca.explained_variance_ratio_[0]:.2f})')
        plt.ylabel(f'PC 2 (Expl. Var: {pca.explained_variance_ratio_[1]:.2f})')
        plt.title(f'Latent Space PCA (Original Dim: {mus.shape[1]})')
        
    plt.grid(True, alpha=0.3)
    plt.show()

def generate_samples(model, device, num_samples=16, transpose=True):
    """
    Generates new samples from the prior distribution.
    
    Args:
        model (nn.Module): The trained VAE model.
        device (torch.device): Device to run the model on.
        num_samples (int): Number of samples to generate.
        transpose (bool): If True, applies .T[::-1, :] to match BOUT++ data orientation.
    """
    model.eval()
    with torch.no_grad():
        # Sample from prior p(z)
        if hasattr(model, 'sample_from_prior'):
            outputs = model.sample_from_prior(batch_size=num_samples)
            px = outputs['px']
            samples = px.mean
        else:
            # Fallback
            z = torch.randn(num_samples, model.latent_features).to(device)
            px = model.observation_model(z)
            samples = px.mean
            
    samples = samples.cpu().numpy()
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten()
    
    for i in range(grid_size * grid_size):
        if i < num_samples:
            img = samples[i, 0]
            if transpose:
                img = img.T[::-1, :]
            axes[i].imshow(img, cmap='viridis')
            axes[i].axis('off')
        else:
            axes[i].axis('off')
            
    plt.suptitle('Generated Samples from Prior')
    plt.tight_layout()
    plt.show()
