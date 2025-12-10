#!/usr/bin/env python3
"""
CLI: PET Image Reconstruction

Command-line interface for PET image reconstruction from processed list-mode data.

Usage:
    cli_reconstruct_pet.py [OPTIONS] LISTMODE_FILE

Example:
    # Basic reconstruction
    cli_reconstruct_pet.py output/processed/listmode_data.npz

    # Custom image dimensions
    cli_reconstruct_pet.py --image-size 256,256,128 listmode_data.npz

    # Adjust voxel size
    cli_reconstruct_pet.py --voxel-size 1.0,1.0,1.0 listmode_data.npz
"""

import sys
from pathlib import Path

import click
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

try:
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    click.echo("WARNING: PyTorch not installed (GPU acceleration unavailable)", err=True)

try:
    from pytomography.metadata import ObjectMeta, PETScannerParams
    from pytomography.algorithms import OSEM
    HAS_PYTOMOGRAPHY = True
except ImportError:
    HAS_PYTOMOGRAPHY = False


# ==============================================================================
# RECONSTRUCTION FUNCTIONS
# ==============================================================================

def load_listmode_data(filepath, verbose=True):
    """Load processed list-mode data."""
    if verbose:
        click.echo(f"Loading list-mode data...")

    data = np.load(filepath)

    if verbose:
        click.echo(f"  ✓ Events: {len(data['x1'])}")
        click.echo(f"  LOR length: {np.min(data['lor_length']):.1f} - {np.max(data['lor_length']):.1f} mm")

    return dict(data)


def create_sinogram(listmode_data, num_bins_radial=128, num_bins_angular=180,
                    image_shape=(128, 128, 64), verbose=True):
    """Convert list-mode to sinogram."""
    if verbose:
        click.echo("\nCreating sinogram...")

    x1, y1, z1 = listmode_data['x1'], listmode_data['y1'], listmode_data['z1']
    x2, y2, z2 = listmode_data['x2'], listmode_data['y2'], listmode_data['z2']

    # LOR parameters
    x_mid = (x1 + x2) / 2
    y_mid = (y1 + y2) / 2
    z_mid = (z1 + z2) / 2

    dx = x2 - x1
    dy = y2 - y1

    # Radial distance (perpendicular to LOR)
    radial_dist = np.abs(x_mid * dy - y_mid * dx) / np.sqrt(dx**2 + dy**2 + 1e-10)

    # Angle
    angle = np.arctan2(dy, dx) * 180 / np.pi
    angle = (angle + 180) % 180

    # Axial slice
    z_slice = ((z_mid - np.min(z_mid)) / (np.max(z_mid) - np.min(z_mid) + 1e-10) *
               (image_shape[2] - 1)).astype(int)
    z_slice = np.clip(z_slice, 0, image_shape[2] - 1)

    # Create sinogram
    radial_bins = np.linspace(0, np.max(radial_dist) * 1.1, num_bins_radial)
    angular_bins = np.linspace(0, 180, num_bins_angular)

    sinogram = np.zeros((num_bins_angular, num_bins_radial, image_shape[2]))

    # Histogram LORs
    with click.progressbar(length=len(x1), label='Binning LORs') as bar:
        for i in range(len(x1)):
            r_idx = np.digitize(radial_dist[i], radial_bins) - 1
            a_idx = np.digitize(angle[i], angular_bins) - 1
            z_idx = z_slice[i]

            if 0 <= r_idx < num_bins_radial and 0 <= a_idx < num_bins_angular:
                sinogram[a_idx, r_idx, z_idx] += 1

            if i % 1000 == 0:
                bar.update(1000)

    if verbose:
        click.echo(f"  ✓ Sinogram shape: {sinogram.shape}")
        click.echo(f"  Total counts: {np.sum(sinogram):.0f}")
        click.echo(f"  Mean (non-zero): {np.mean(sinogram[sinogram > 0]):.2f}")

    return sinogram


def visualize_sinogram(sinogram, output_path, verbose=True):
    """Visualize sinogram data."""
    if verbose:
        click.echo("\nGenerating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("PET Sinogram Analysis", fontsize=16, fontweight='bold')

    mid_slice = sinogram.shape[2] // 2

    # Sinogram (middle slice)
    ax = axes[0, 0]
    im = ax.imshow(sinogram[:, :, mid_slice], aspect='auto', cmap='hot')
    ax.set_xlabel('Radial Bin')
    ax.set_ylabel('Angular Bin')
    ax.set_title(f'Sinogram (Slice {mid_slice})')
    plt.colorbar(im, ax=ax, label='Counts')

    # Projection (sum over angles)
    ax = axes[0, 1]
    projection = np.sum(sinogram, axis=0)
    im = ax.imshow(projection, aspect='auto', cmap='hot')
    ax.set_xlabel('Radial Bin')
    ax.set_ylabel('Axial Slice')
    ax.set_title('Sinogram (Angular Sum)')
    plt.colorbar(im, ax=ax, label='Counts')

    # Radial profile
    ax = axes[1, 0]
    profile = np.sum(sinogram[:, :, mid_slice], axis=0)
    ax.plot(profile)
    ax.set_xlabel('Radial Bin')
    ax.set_ylabel('Counts')
    ax.set_title('Radial Profile')
    ax.grid(True, alpha=0.3)

    # Axial profile
    ax = axes[1, 1]
    axial_profile = np.sum(np.sum(sinogram, axis=0), axis=0)
    ax.plot(axial_profile)
    ax.set_xlabel('Axial Slice')
    ax.set_ylabel('Counts')
    ax.set_title('Axial Profile')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if verbose:
        click.echo(f"  ✓ Saved: {output_path.name}")
    plt.close()


def show_osem_template(verbose=True):
    """Display OSEM reconstruction template."""
    if not verbose:
        return

    click.echo("\n" + "=" * 70)
    click.echo("OSEM RECONSTRUCTION TEMPLATE (PyTomography)")
    click.echo("=" * 70)

    template = """
For full OSEM reconstruction with PyTomography:

1. Define scanner geometry:
   scanner_params = PETScannerParams(...)

2. Create system matrix:
   system_matrix = PETSystemMatrix(
       scanner_params=scanner_params,
       object_meta=object_meta
   )

3. Run OSEM:
   osem = OSEM(system_matrix)
   reconstructed = osem(
       listmode_data,
       initial_image,
       num_iterations=10,
       num_subsets=8
   )

Documentation:
  https://pytomography.readthedocs.io/
  https://github.com/PyTomography/PyTomography/tree/main/examples
"""

    click.echo(template)


# ==============================================================================
# CLI INTERFACE
# ==============================================================================

@click.command()
@click.argument('listmode_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='../output/reconstruction',
              help='Output directory for reconstruction results')
@click.option('--image-size', default='128,128,64',
              help='Image dimensions as x,y,z (default: 128,128,64)')
@click.option('--voxel-size', default='2.0,2.0,2.0',
              help='Voxel size in mm as dx,dy,dz (default: 2.0,2.0,2.0)')
@click.option('--num-bins-radial', default=128,
              help='Number of radial bins in sinogram (default: 128)')
@click.option('--num-bins-angular', default=180,
              help='Number of angular bins in sinogram (default: 180)')
@click.option('--plot/--no-plot', default=True,
              help='Generate visualization plots')
@click.option('--show-template/--no-template', default=True,
              help='Show OSEM reconstruction template')
@click.option('--verbose/--quiet', '-v/-q', default=True,
              help='Verbose output')
def reconstruct_pet(listmode_file, output_dir, image_size, voxel_size,
                    num_bins_radial, num_bins_angular, plot, show_template, verbose):
    """
    Reconstruct PET images from processed list-mode data.

    Creates sinogram representation and provides OSEM reconstruction template.
    Full reconstruction requires PyTomography with scanner calibration.

    \b
    Examples:
        # Basic reconstruction
        cli_reconstruct_pet.py output/processed/listmode_data.npz

        # Custom image parameters
        cli_reconstruct_pet.py --image-size 256,256,128 --voxel-size 1.0,1.0,1.0 data.npz

        # High-resolution sinogram
        cli_reconstruct_pet.py --num-bins-radial 256 --num-bins-angular 360 data.npz
    """
    click.echo("=" * 70)
    click.echo("PET IMAGE RECONSTRUCTION")
    click.echo("=" * 70)

    try:
        # Parse parameters
        img_shape = tuple(map(int, image_size.split(',')))
        vox_size = tuple(map(float, voxel_size.split(',')))

        if len(img_shape) != 3:
            raise click.BadParameter("image-size must be x,y,z")
        if len(vox_size) != 3:
            raise click.BadParameter("voxel-size must be dx,dy,dz")

        if verbose:
            click.echo(f"\nConfiguration:")
            click.echo(f"  Image size: {img_shape}")
            click.echo(f"  Voxel size: {vox_size} mm")
            click.echo(f"  Sinogram bins: {num_bins_angular} × {num_bins_radial}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Load data
        if verbose:
            click.echo("\n[1/3] Loading Data")
        listmode_data = load_listmode_data(listmode_file, verbose)

        # Step 2: Create sinogram
        if verbose:
            click.echo("\n[2/3] Creating Sinogram")
        sinogram = create_sinogram(
            listmode_data,
            num_bins_radial=num_bins_radial,
            num_bins_angular=num_bins_angular,
            image_shape=img_shape,
            verbose=verbose
        )

        # Save sinogram
        sino_path = output_path / "sinogram.npy"
        np.save(sino_path, sinogram)
        if verbose:
            click.echo(f"  ✓ Saved: {sino_path.name}")

        # Step 3: Visualization
        if plot:
            if verbose:
                click.echo("\n[3/3] Generating Visualization")
            plot_path = output_path / "sinogram_analysis.png"
            visualize_sinogram(sinogram, plot_path, verbose)

        # Show template
        if show_template:
            show_osem_template(verbose)

        # Summary
        click.echo("\n" + "=" * 70)
        click.echo("RECONSTRUCTION PREPARATION COMPLETE")
        click.echo("=" * 70)
        click.echo(f"\n✓ Output saved to: {output_path.absolute()}")
        click.echo(f"✓ Sinogram shape: {sinogram.shape}")
        click.echo(f"✓ Total counts: {np.sum(sinogram):.0f}")

        if not HAS_PYTOMOGRAPHY:
            click.echo("\nℹ For full OSEM reconstruction:")
            click.echo("  pip install pytomography torch")
            click.echo("  See: https://pytomography.readthedocs.io/")

    except Exception as e:
        raise click.ClickException(str(e))


if __name__ == "__main__":
    reconstruct_pet()
