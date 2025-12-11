#!/usr/bin/env python3
"""
PET Image Reconstruction using PyTomography

This script performs OSEM reconstruction of PET data processed from GATE simulations.

Based on PyTomography documentation:
- https://pytomography.readthedocs.io/
- https://github.com/PyTomography/PyTomography

Reference: PyTomography paper (2024)
- "PyTomography: A Python Library for Quantitative Medical Image Reconstruction"
- https://www.sciencedirect.com/science/article/pii/S235271102400390X
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Check dependencies
try:
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch device: {device}")
except ImportError:
    print("ERROR: PyTorch not installed.")
    print("Install with: pip install torch")
    sys.exit(1)

try:
    from pytomography.metadata import ObjectMeta, PETScannerParams
    from pytomography.projectors.system_matrix import SystemMatrix
    from pytomography.algorithms import OSEM
    from pytomography.likelihoods import PoissonLogLikelihood
    HAS_PYTOMOGRAPHY = True
except ImportError:
    print("ERROR: PyTomography not installed.")
    print("Install with: pip install pytomography")
    print("\nFor full installation:")
    print("  pip install torch pytomography matplotlib numpy")
    sys.exit(1)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Input/Output paths
LISTMODE_FILE = "../output/processed/listmode_data.npz"
OUTPUT_DIR = "../output/reconstruction"

# Image space parameters (adjust based on your FOV)
IMAGE_SHAPE = [128, 128, 64]  # [x, y, z] voxels
IMAGE_SPACING = [2.0, 2.0, 2.0]  # mm per voxel [dx, dy, dz]

# Scanner parameters (from geometry_pet.py)
SCANNER_CONFIG = {
    'crystal_size_x': 2.0,  # mm
    'crystal_size_y': 2.0,  # mm
    'crystal_size_z': 19.0,  # mm (thickness)
    'detector_radius': 130.0,  # mm (FOV_RADIUS * 1.3 from geometry)
    'num_crystals_transaxial': 50,
    'num_crystals_axial': 50,
    'num_panels': 2,
    'mean_interaction_depth': 9.5,  # mm (half of crystal thickness)
}

# Reconstruction parameters
RECON_CONFIG = {
    'num_subsets': 8,
    'num_iterations': 10,
    'save_iterations': [1, 3, 5, 10],  # Save images at these iterations
} ff


# ==============================================================================
# SIMPLIFIED SYSTEM MATRIX
# ==============================================================================

class SimplePETSystemMatrix:
    """
    Simplified PET system matrix for two-panel geometry.

    This is a basic implementation. For production use, consider:
    - Full geometric calibration
    - Attenuation correction
    - Scatter correction
    - Normalization
    """

    def __init__(self, scanner_config, image_shape, image_spacing):
        """
        Initialize system matrix.

        Args:
            scanner_config: Scanner geometry parameters
            image_shape: [nx, ny, nz] image dimensions
            image_spacing: [dx, dy, dz] voxel sizes in mm
        """
        self.scanner_config = scanner_config
        self.image_shape = image_shape
        self.image_spacing = image_spacing

        # Calculate image extent
        self.image_extent = [
            image_shape[i] * image_spacing[i] for i in range(3)
        ]

        print("\nSystem Matrix Configuration:")
        print(f"  Image shape: {image_shape}")
        print(f"  Voxel size: {image_spacing} mm")
        print(f"  Image extent: {self.image_extent} mm")
        print(f"  Scanner radius: {scanner_config['detector_radius']} mm")

    def forward_project(self, image):
        """
        Forward projection: image → sinogram/listmode.

        Args:
            image: 3D image array

        Returns:
            Projected data
        """
        # Placeholder for forward projection
        # In production, use PyTomography's projector or parallelproj
        raise NotImplementedError("Forward projection requires full PyTomography setup")

    def back_project(self, data):
        """
        Back projection: sinogram/listmode → image.

        Args:
            data: Projection data

        Returns:
            Back-projected image
        """
        # Placeholder for back projection
        raise NotImplementedError("Back projection requires full PyTomography setup")


# ==============================================================================
# RECONSTRUCTION FUNCTIONS
# ==============================================================================

def load_listmode_data(filepath):
    """
    Load processed list-mode data.

    Args:
        filepath: Path to .npz file with list-mode data

    Returns:
        Dictionary with LOR coordinates and properties
    """
    print(f"\nLoading list-mode data from {filepath}...")

    data = np.load(filepath)

    print(f"  Events loaded: {len(data['x1'])}")
    print(f"  LOR length range: {np.min(data['lor_length']):.1f} - {np.max(data['lor_length']):.1f} mm")
    print(f"  Energy range: {np.min(data['energy1']):.1f} - {np.max(data['energy2']):.1f} keV")

    return dict(data)


def create_sinogram_from_listmode(listmode_data, num_bins_radial=128, num_bins_angular=180):
    """
    Convert list-mode data to sinogram representation.

    Args:
        listmode_data: Dictionary with LOR data
        num_bins_radial: Number of radial bins
        num_bins_angular: Number of angular bins

    Returns:
        Sinogram array [num_bins_angular, num_bins_radial, num_slices]
    """
    print("\nCreating sinogram from list-mode data...")

    # Extract LOR endpoints
    x1, y1, z1 = listmode_data['x1'], listmode_data['y1'], listmode_data['z1']
    x2, y2, z2 = listmode_data['x2'], listmode_data['y2'], listmode_data['z2']

    # Calculate LOR parameters (Siddon's algorithm concepts)
    # Mid-point of LOR
    x_mid = (x1 + x2) / 2
    y_mid = (y1 + y2) / 2
    z_mid = (z1 + z2) / 2

    # LOR direction
    dx = x2 - x1
    dy = y2 - y1

    # Calculate radial distance (distance from center to LOR)
    # Using perpendicular distance formula
    radial_dist = np.abs(x_mid * dy - y_mid * dx) / np.sqrt(dx**2 + dy**2)

    # Calculate angle
    angle = np.arctan2(dy, dx) * 180 / np.pi  # Convert to degrees
    angle = (angle + 180) % 180  # Normalize to [0, 180]

    # Axial position (z coordinate)
    z_slice = ((z_mid - np.min(z_mid)) / (np.max(z_mid) - np.min(z_mid)) *
               (IMAGE_SHAPE[2] - 1)).astype(int)
    z_slice = np.clip(z_slice, 0, IMAGE_SHAPE[2] - 1)

    # Bin into sinogram
    radial_bins = np.linspace(0, np.max(radial_dist) * 1.1, num_bins_radial)
    angular_bins = np.linspace(0, 180, num_bins_angular)

    sinogram = np.zeros((num_bins_angular, num_bins_radial, IMAGE_SHAPE[2]))

    # Histogram LORs into sinogram
    for i in range(len(x1)):
        r_idx = np.digitize(radial_dist[i], radial_bins) - 1
        a_idx = np.digitize(angle[i], angular_bins) - 1
        z_idx = z_slice[i]

        if 0 <= r_idx < num_bins_radial and 0 <= a_idx < num_bins_angular:
            sinogram[a_idx, r_idx, z_idx] += 1

    print(f"  Sinogram shape: {sinogram.shape}")
    print(f"  Total counts: {np.sum(sinogram):.0f}")
    print(f"  Mean counts per bin: {np.mean(sinogram[sinogram > 0]):.2f}")

    return sinogram


def simple_fbp_reconstruction(sinogram):
    """
    Simple Filtered Back-Projection reconstruction.

    This is a basic implementation for visualization.
    For quantitative imaging, use OSEM with full corrections.

    Args:
        sinogram: [num_angles, num_bins, num_slices] sinogram

    Returns:
        Reconstructed 3D image
    """
    print("\nPerforming FBP reconstruction...")

    num_angles, num_bins, num_slices = sinogram.shape

    # Initialize image
    image = np.zeros(IMAGE_SHAPE)

    # Simplified back-projection (sum over angles)
    # Note: This is NOT proper FBP - it's unfiltered back-projection
    # For demonstration purposes only

    for slice_idx in range(num_slices):
        # Back-project this slice
        slice_sinogram = sinogram[:, :, slice_idx]

        if np.sum(slice_sinogram) == 0:
            continue

        # Simple back-projection: sum projections
        # (Proper FBP requires ramp filter and correct geometric weighting)
        slice_recon = np.sum(slice_sinogram, axis=0)

        # Map to image space (very simplified)
        # This is a placeholder - proper reconstruction needs geometric mapping

    print("  Note: This is simplified back-projection, not full FBP")
    print("  For quantitative results, use OSEM with PyTomography")

    return image


def osem_reconstruction_template():
    """
    Template for OSEM reconstruction with PyTomography.

    This is a code template showing the PyTomography workflow.
    Actual implementation requires full scanner calibration.
    """
    print("\n" + "="*70)
    print("OSEM RECONSTRUCTION TEMPLATE")
    print("="*70)

    print("\nPyTomography OSEM Reconstruction Workflow:")
    print("\n1. Define Object Metadata:")
    print("   object_meta = ObjectMeta(")
    print("       dr=[2.0, 2.0, 2.0],  # voxel size in mm")
    print("       shape=[128, 128, 64]  # image dimensions")
    print("   )")

    print("\n2. Define Scanner Parameters:")
    print("   scanner_params = PETScannerParams(")
    print("       scanner_lut=...,  # Look-up table for detector pairs")
    print("       tof_meta=...,      # Time-of-flight metadata")
    print("   )")

    print("\n3. Create System Matrix:")
    print("   system_matrix = SystemMatrix(")
    print("       obj2obj_transforms=[...],  # Attenuation, etc.")
    print("       proj2proj_transforms=[...], # Scatter, randoms, etc.")
    print("       object_meta=object_meta,")
    print("       scanner_params=scanner_params")
    print("   )")

    print("\n4. Initialize OSEM:")
    print("   osem = OSEM(system_matrix)")
    print("   initial_image = torch.ones(image_shape) * initial_value")

    print("\n5. Run Reconstruction:")
    print("   reconstructed = osem(")
    print("       listmode_data,")
    print("       initial_image,")
    print("       num_iterations=10,")
    print("       num_subsets=8")
    print("   )")

    print("\n" + "="*70)
    print("For complete implementation, see:")
    print("  https://pytomography.readthedocs.io/")
    print("  https://github.com/PyTomography/PyTomography/tree/main/examples")
    print("="*70)


def visualize_results(sinogram, image=None, save_dir=OUTPUT_DIR):
    """
    Visualize sinogram and reconstructed image.

    Args:
        sinogram: Sinogram data
        image: Reconstructed image (optional)
        save_dir: Directory to save figures
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print("\nGenerating visualization...")

    # Create figure
    if image is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes = axes.reshape(1, -1)

    fig.suptitle("PET Reconstruction Results", fontsize=16, fontweight='bold')

    # Sinogram views
    mid_slice = sinogram.shape[2] // 2

    # Sinogram (middle slice)
    ax = axes[0, 0]
    im = ax.imshow(sinogram[:, :, mid_slice], aspect='auto', cmap='hot')
    ax.set_xlabel('Radial Bin')
    ax.set_ylabel('Angular Bin')
    ax.set_title(f'Sinogram (Slice {mid_slice})')
    plt.colorbar(im, ax=ax, label='Counts')

    # Sinogram projection (sum over angles)
    ax = axes[0, 1]
    projection = np.sum(sinogram, axis=0)
    im = ax.imshow(projection, aspect='auto', cmap='hot')
    ax.set_xlabel('Radial Bin')
    ax.set_ylabel('Axial Slice')
    ax.set_title('Sinogram (Angular Sum)')
    plt.colorbar(im, ax=ax, label='Counts')

    # Radial profile
    ax = axes[0, 2]
    profile = np.sum(sinogram[:, :, mid_slice], axis=0)
    ax.plot(profile)
    ax.set_xlabel('Radial Bin')
    ax.set_ylabel('Counts')
    ax.set_title('Radial Profile (Middle Slice)')
    ax.grid(True, alpha=0.3)

    # Reconstructed image (if available)
    if image is not None:
        mid_x = image.shape[0] // 2
        mid_y = image.shape[1] // 2
        mid_z = image.shape[2] // 2

        # XY slice
        ax = axes[1, 0]
        im = ax.imshow(image[:, :, mid_z].T, origin='lower', cmap='hot')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Reconstructed (Z={mid_z})')
        plt.colorbar(im, ax=ax, label='Activity')

        # XZ slice
        ax = axes[1, 1]
        im = ax.imshow(image[:, mid_y, :].T, origin='lower', cmap='hot')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title(f'Reconstructed (Y={mid_y})')
        plt.colorbar(im, ax=ax, label='Activity')

        # YZ slice
        ax = axes[1, 2]
        im = ax.imshow(image[mid_x, :, :].T, origin='lower', cmap='hot')
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.set_title(f'Reconstructed (X={mid_x})')
        plt.colorbar(im, ax=ax, label='Activity')

    plt.tight_layout()

    fig_path = save_path / "reconstruction_results.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {fig_path}")

    plt.show()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main reconstruction workflow."""

    print("="*70)
    print("PET IMAGE RECONSTRUCTION")
    print("="*70)

    # Check if listmode file exists
    listmode_path = Path(LISTMODE_FILE)
    if not listmode_path.exists():
        print(f"\nERROR: List-mode file not found: {listmode_path}")
        print("Please run event processing first:")
        print("  python scripts/process_events.py")
        return

    # Load data
    print("\n" + "="*70)
    print("STEP 1: Loading Data")
    print("="*70)
    listmode_data = load_listmode_data(listmode_path)

    # Create sinogram
    print("\n" + "="*70)
    print("STEP 2: Sinogram Creation")
    print("="*70)
    sinogram = create_sinogram_from_listmode(listmode_data)

    # Save sinogram
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / "sinogram.npy", sinogram)
    print(f"  ✓ Saved sinogram to: {output_path / 'sinogram.npy'}")

    # Visualize
    print("\n" + "="*70)
    print("STEP 3: Visualization")
    print("="*70)
    visualize_results(sinogram)

    # Display OSEM reconstruction template
    print("\n" + "="*70)
    print("STEP 4: OSEM Reconstruction")
    print("="*70)
    osem_reconstruction_template()

    print("\n" + "="*70)
    print("RECONSTRUCTION WORKFLOW COMPLETE")
    print("="*70)
    print(f"\nOutput saved to: {output_path.absolute()}")
    print("\nNext steps for quantitative reconstruction:")
    print("  1. Configure scanner geometry in PyTomography")
    print("  2. Implement attenuation correction")
    print("  3. Add scatter and random corrections")
    print("  4. Run OSEM with full system matrix")
    print("\nSee PyTomography documentation:")
    print("  https://pytomography.readthedocs.io/")


if __name__ == "__main__":
    main()
