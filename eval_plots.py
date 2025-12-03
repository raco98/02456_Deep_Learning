import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from sklearn.decomposition import PCA
from IPython.display import display, clear_output

import os
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from IPython.display import Image, display, clear_output
from sklearn.manifold import TSNE
from torch import Tensor
from torch.distributions import Normal
from torchvision.utils import make_grid

def plot_training_curves(training_data, validation_data, save_path=None):
    """
    Plots the training and validation curves for ELBO, Log Likelihood, and KL Divergence.
    
    Args:
        training_data (dict): Dictionary containing lists of training metrics.
        validation_data (dict): Dictionary containing lists of validation metrics.
        save_path (str, optional): Path to save the plots (without extension). 
                                   If provided, saves as both .pdf and .png.
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
    
    if save_path:
        # Save combined plot
        plt.savefig(f"{save_path}_combined.pdf")
        plt.savefig(f"{save_path}_combined.png")
        
        # Save individual plots
        metrics = [
            ('elbo', 'ELBO', 'ELBO'),
            ('log_px', 'Log Likelihood (Reconstruction)', None),
            ('kl', 'KL Divergence', None)
        ]
        
        for key, title, ylabel in metrics:
            if key in training_data:
                plt.figure(figsize=(8, 6))
                plt.plot(training_data[key], label='Train')
                if key in validation_data:
                    plt.plot(validation_data[key], label='Validation')
                plt.title(title)
                plt.xlabel('Epoch')
                if ylabel:
                    plt.ylabel(ylabel)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"{save_path}_{key}.pdf")
                plt.savefig(f"{save_path}_{key}.png")
                plt.close()
        
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


def plot_autoencoder_stats(
        x: Tensor = None,
        x_hat: Tensor = None,
        z: Tensor = None,
        y: Tensor = None,
        epoch: int = None,
        train_loss: List = None,
        valid_loss: List = None,
        classes: List = None,
        dimensionality_reduction_op: Optional[Callable] = None,
) -> None:
    """
    An utility 
    """
    # -- Plotting --
    f, axarr = plt.subplots(2, 2, figsize=(20, 20))

    # Loss
    ax = axarr[0, 0]
    ax.set_title("Error")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')

    ax.plot(np.arange(epoch + 1), train_loss, color="black")
    ax.plot(np.arange(epoch + 1), valid_loss, color="gray", linestyle="--")
    ax.legend(['Training error', 'Validation error'])

    # Latent space
    ax = axarr[0, 1]

    ax.set_title('Latent space')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')

    # If you want to use a dimensionality reduction method you can use
    # for example TSNE by projecting on two principal dimensions
    # TSNE.fit_transform(z)
    if dimensionality_reduction_op is not None:
        z = dimensionality_reduction_op(z)

    colors = iter(plt.get_cmap('Set1')(np.linspace(0, 1.0, len(classes))))
    for c in classes:
        ax.scatter(*z[y.numpy() == c].T, c=next(colors), marker='o')

    ax.legend(classes)

    # Inputs
    ax = axarr[1, 0]
    ax.set_title('Inputs')
    ax.axis('off')

    rows = 8
    batch_size = x.size(0)
    columns = batch_size // rows

    canvas = np.zeros((28 * rows, columns * 28))
    for i in range(rows):
        for j in range(columns):
            idx = i % columns + rows * j
            canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = x[idx].reshape((28, 28))
    ax.imshow(canvas, cmap='gray')

    # Reconstructions
    ax = axarr[1, 1]
    ax.set_title('Reconstructions')
    ax.axis('off')

    canvas = np.zeros((28 * rows, columns * 28))
    for i in range(rows):
        for j in range(columns):
            idx = i % columns + rows * j
            canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = x_hat[idx].reshape((28, 28))

    ax.imshow(canvas, cmap='gray')

    tmp_img = "tmp_ae_out.png"
    plt.savefig(tmp_img)
    plt.close(f)
    display(Image(filename=tmp_img))
    clear_output(wait=True)

    os.remove(tmp_img)


def plot_samples(ax, x):
    x = x.to('cpu')
    nrow = int(np.sqrt(x.size(0)))
    x_grid = make_grid(x.view(-1, 1, 28, 28), nrow=nrow).permute(1, 2, 0)
    ax.imshow(x_grid)
    ax.axis('off')


def plot_interpolations(ax, vae):
    device = next(iter(vae.parameters())).device
    nrow = 10
    nsteps = 10
    prior_params = vae.prior_params.expand(2 * nrow, *vae.prior_params.shape[-1:])
    mu, log_sigma = prior_params.chunk(2, dim=-1)
    pz = Normal(mu, log_sigma.exp())
    z = pz.sample().view(nrow, 2, -1)
    t = torch.linspace(0, 1, 10, device=device)
    zs = t[None, :, None] * z[:, 0, None, :] + (1 - t[None, :, None]) * z[:, 1, None, :]
    px = vae.observation_model(zs.view(nrow * nsteps, -1))
    x = px.sample()
    x = x.to('cpu')
    x_grid = make_grid(x.view(-1, 1, 28, 28), nrow=nrow).permute(1, 2, 0)
    ax.imshow(x_grid)
    ax.axis('off')


def plot_grid(ax, vae):
    device = next(iter(vae.parameters())).device
    nrow = 10
    xv, yv = torch.meshgrid([torch.linspace(-3, 3, 10), torch.linspace(-3, 3, 10)])
    zs = torch.cat([xv[:, :, None], yv[:, :, None]], -1)
    zs = zs.to(device)
    px = vae.observation_model(zs.view(nrow * nrow, 2))
    x = px.sample()
    x = x.to('cpu')
    x_grid = make_grid(x.view(-1, 1, 28, 28), nrow=nrow).permute(1, 2, 0)
    ax.imshow(x_grid)
    ax.axis('off')


def plot_2d_latents(ax, qz, z, y):
    z = z.to('cpu')
    y = y.to('cpu')
    scale_factor = 2
    batch_size = z.shape[0]
    palette = sns.color_palette()
    colors = [palette[l] for l in y]

    # plot prior
    prior = plt.Circle((0, 0), scale_factor, color='gray', fill=True, alpha=0.1)
    ax.add_artist(prior)

    # plot data points
    mus, sigmas = qz.mu.to('cpu'), qz.sigma.to('cpu')
    mus = [mus[i].numpy().tolist() for i in range(batch_size)]
    sigmas = [sigmas[i].numpy().tolist() for i in range(batch_size)]

    posteriors = [
        plt.matplotlib.patches.Ellipse(mus[i], *(scale_factor * s for s in sigmas[i]), color=colors[i], fill=False,
                                       alpha=0.3) for i in range(batch_size)]
    for p in posteriors:
        ax.add_artist(p)

    ax.scatter(z[:, 0], z[:, 1], color=colors)

    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_aspect('equal', 'box')


def plot_latents(ax, z, y):
    z = z.to('cpu')
    palette = sns.color_palette()
    colors = [palette[l] for l in y]
    z = TSNE(n_components=2).fit_transform(z)
    ax.scatter(z[:, 0], z[:, 1], color=colors)


def make_vae_plots(vae, x, y, outputs, training_data, validation_data, tmp_img="tmp_vae_out.png", figsize=(18, 18)):
    fig, axes = plt.subplots(3, 3, figsize=figsize, squeeze=False)

    # plot the observation
    axes[0, 0].set_title(r'Observation $\mathbf{x}$')
    plot_samples(axes[0, 0], x)

    # plot the latent samples
    try:
        z = outputs['z']
        if z.shape[1] == 2:
            axes[0, 1].set_title(r'Latent Samples $\mathbf{z} \sim q_\phi(\mathbf{z} | \mathbf{x})$')
            qz = outputs['qz']
            plot_2d_latents(axes[0, 1], qz, z, y)
        else:
            axes[0, 1].set_title(r'Latent Samples $\mathbf{z} \sim q_\phi(\mathbf{z} | \mathbf{x})$ (t-SNE)')
            plot_latents(axes[0, 1], z, y)
    except Exception as e:
        print(f"Could not generate the plot of the latent sanples because of exception")
        print(e)

    # plot posterior samples
    axes[0, 2].set_title(
        r'Reconstruction $\mathbf{x} \sim p_\theta(\mathbf{x} | \mathbf{z}), \mathbf{z} \sim q_\phi(\mathbf{z} | \mathbf{x})$')
    px = outputs['px']
    x_sample = px.sample().to('cpu')
    plot_samples(axes[0, 2], x_sample)

    # plot ELBO
    ax = axes[1, 0]
    ax.set_title(r'ELBO: $\mathcal{L} ( \mathbf{x} )$')
    ax.plot(training_data['elbo'], label='Training')
    ax.plot(validation_data['elbo'], label='Validation')
    ax.legend()

    # plot KL
    ax = axes[1, 1]
    ax.set_title(r'$\mathcal{D}_{\operatorname{KL}}\left(q_\phi(\mathbf{z}|\mathbf{x})\ |\ p(\mathbf{z})\right)$')
    ax.plot(training_data['kl'], label='Training')
    ax.plot(validation_data['kl'], label='Validation')
    ax.legend()

    # plot NLL
    ax = axes[1, 2]
    ax.set_title(r'$\log p_\theta(\mathbf{x} | \mathbf{z})$')
    ax.plot(training_data['log_px'], label='Training')
    ax.plot(validation_data['log_px'], label='Validation')
    ax.legend()

    # plot prior samples
    axes[2, 0].set_title(r'Samples $\mathbf{x} \sim p_\theta(\mathbf{x} | \mathbf{z}), \mathbf{z} \sim p(\mathbf{z})$')
    px = vae.sample_from_prior(batch_size=x.size(0))['px']
    x_samples = px.sample()
    plot_samples(axes[2, 0], x_samples)

    # plot interpolations samples
    axes[2, 1].set_title(
        r'Latent Interpolations: $\mathbf{x} \sim p_\theta(\mathbf{x} | t \cdot \mathbf{z}_1 + (1-t) \cdot \mathbf{z}_2), \mathbf{z}_1, \mathbf{z}_2 \sim p(\mathbf{z}), t=0 \dots 1$')
    plot_interpolations(axes[2, 1], vae)

    # plot samples (sampling from a grid instead of the prior)
    if vae.latent_features == 2:
        axes[2, 2].set_title(
            r'Samples: $\mathbf{x} \sim p_\theta(\mathbf{x} | \mathbf{z}), \mathbf{z} \sim \operatorname{grid}(-3:3, -3:3)$')
        px = vae.sample_from_prior(batch_size=x.size(0))['px']
        x_samples = px.sample()
        plot_grid(axes[2, 2], vae)

    # display
    plt.tight_layout()
    plt.savefig(tmp_img)
    plt.close(fig)
    display(Image(filename=tmp_img))
    clear_output(wait=True)

    os.remove(tmp_img)

def make_plasma_vae_plots(vae, x, outputs, training_data, validation_data,
                          tmp_img="tmp_vae_out.png", figsize=(18, 12)):

    px = outputs["px"]       # Bernoulli distribution
    qz = outputs["qz"]
    z = outputs["z"]         # latent samples


    x = x.detach().cpu()
    reconstruction = px.mean.detach().cpu()
    z = z.detach().cpu()

    B, C, H, W = x.shape

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # ---------------------------------------------------------
    # 1. Show one input image
    # ---------------------------------------------------------
    axes[0, 0].set_title("Input (first sample)")
    img = x[0, 0].numpy().T[::-1, :]   # transpose + flip vertically
    axes[0, 0].imshow(img, cmap='rocket')
    axes[0, 0].axis("off")

    # ---------------------------------------------------------
    # 2. Show reconstruction
    # ---------------------------------------------------------
    axes[0, 1].set_title("Reconstruction (first sample)")
    img_hat = reconstruction[0, 0].numpy().T[::-1, :]
    axes[0, 1].imshow(img_hat, cmap='rocket')
    axes[0, 1].axis("off")

    # ---------------------------------------------------------
    # 3. Latent space (if dim=2)
    # ---------------------------------------------------------
    if z.shape[1] == 2:
        axes[0, 2].set_title("Latent space")
        axes[0, 2].scatter(z[:, 0], z[:, 1], s=10, alpha=0.6)
        axes[0, 2].set_xlabel("z1")
        axes[0, 2].set_ylabel("z2")
    else:
        axes[0, 2].set_title("Latent space (dim > 2)")
        axes[0, 2].text(0.5, 0.5, "Latent dim > 2", ha="center", va="center")
        axes[0, 2].axis("off")

    # ---------------------------------------------------------
    # 4. ELBO
    # ---------------------------------------------------------
    ax = axes[1, 0]
    ax.set_title("ELBO")
    ax.plot(training_data["elbo"], label="train")
    ax.plot(validation_data["elbo"], label="val")
    ax.legend()

    # ---------------------------------------------------------
    # 5. KL
    # ---------------------------------------------------------
    ax = axes[1, 1]
    ax.set_title("KL")
    ax.plot(training_data["kl"], label="train")
    ax.plot(validation_data["kl"], label="val")
    ax.legend()

    # ---------------------------------------------------------
    # 6. log p(x|z)
    # ---------------------------------------------------------
    ax = axes[1, 2]
    ax.set_title("log p(x | z)")
    ax.plot(training_data["log_px"], label="train")
    ax.plot(validation_data["log_px"], label="val")
    ax.legend()

    # ---------------------------------------------------------
    # Finalize and show
    # ---------------------------------------------------------
    plt.tight_layout()
    plt.savefig(tmp_img)
    plt.close(fig)

    display(Image(filename=tmp_img))
    clear_output(wait=True)
    os.remove(tmp_img)