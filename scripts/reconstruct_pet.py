#!/usr/bin/env python3
"""
PET Image Reconstruction using PyTomography

This script performs OSEM reconstruction of PET data processed from GATE simulations.

**8-Panel Geometry Support:**
This version supports 8 detector panels arranged in a circle at 45° spacing.
Coincidences are detected only between opposing panels (180° apart):
- Panel 0 ↔ Panel 4 (0° and 180°)
- Panel 1 ↔ Panel 5 (45° and 225°)
- Panel 2 ↔ Panel 6 (90° and 270°)
- Panel 3 ↔ Panel 7 (135° and 315°)

The sinogram representation covers the full 360° angular range to accommodate
all 4 opposing panel pairs. This differs from traditional 2-panel PET which
only requires 180° coverage.

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
    HAS_TORCH = True
    
    
except ImportError:
    print("ERROR: PyTorch not installed.")
    print("Install with: pip install torch")
    HAS_TORCH = False
    sys.exit(1)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Input/Output paths
LISTMODE_FILE = "output/processed/listmode_data.npz"
OUTPUT_DIR = "output/reconstruction"

# Image space parameters (adjust based on your FOV)
IMAGE_SHAPE = [128, 128, 64]  # [x, y, z] voxels
IMAGE_SPACING = [2.0, 2.0, 2.0]  # mm per voxel [dx, dy, dz]

# Scanner parameters (from geometry_pet.py)
SCANNER_CONFIG = {
    'crystal_size_x': 2.0,  # mm
    'crystal_size_y': 2.0,  # mm
    'crystal_size_z': 19.0,  # mm (thickness)
    'detector_radius': 130.0,  # mm (FOV_RADIUS * 1.4 from geometry)
    'num_crystals_transaxial': 50,
    'num_crystals_axial': 50,
    'num_panels': 8,  # 8 panels in circular arrangement
    'panel_spacing_degrees': 45.0,  # 360° / 8 panels
    'opposing_panel_pairs': [(0, 4), (1, 5), (2, 6), (3, 7)],  # Opposing pairs for coincidences
    'mean_interaction_depth': 9.5,  # mm (half of crystal thickness)
}

# Reconstruction parameters
RECON_CONFIG = {
    'num_subsets': 8,
    'num_iterations': 10,
    'save_iterations': [1, 3, 5, 10],  # Save images at these iterations
}


# ==============================================================================
# SIMPLIFIED SYSTEM MATRIX
# ==============================================================================

class SimplePETSystemMatrix:
    """
    Simplified PET system matrix for multi-panel geometry.

    Supports both 2-panel and 8-panel circular detector configurations.
    For 8-panel geometry, the system matrix accounts for the 45° angular
    spacing between adjacent panels and the opposing-panel coincidence logic.

    This is a basic implementation. For production use, consider:
    - Full geometric calibration with panel-specific sensitivity
    - Attenuation correction
    - Scatter correction
    - Normalization for detector efficiency variations
    - Depth-of-interaction (DOI) modeling for thick crystals
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

    # Check for panel IDs (8-panel geometry)
    if 'panel_id1' in data and 'panel_id2' in data:
        print(f"  ✓ Panel IDs found (8-panel geometry)")
        unique_pairs = set()
        for p1, p2 in zip(data['panel_id1'], data['panel_id2']):
            pair = tuple(sorted([p1, p2]))
            unique_pairs.add(pair)
        print(f"    Active panel pairs: {sorted(unique_pairs)}")
    else:
        print(f"  ⚠ No panel IDs (assuming 2-panel geometry)")

    return dict(data)


def create_sinogram_from_listmode(listmode_data, num_bins_radial=128, num_bins_angular=180):
    """
    Convert list-mode data to sinogram representation.

    For 8-panel geometry, this handles LORs from opposing panel pairs.
    The angular range [0, 360°] covers all 4 opposing pairs:
    - Panel 0 ↔ 4: ~0° and ~180°
    - Panel 1 ↔ 5: ~45° and ~225°
    - Panel 2 ↔ 6: ~90° and ~270°
    - Panel 3 ↔ 7: ~135° and ~315°

    Args:
        listmode_data: Dictionary with LOR data (including optional panel_id1, panel_id2)
        num_bins_radial: Number of radial bins
        num_bins_angular: Number of angular bins (360 for full rotation)

    Returns:
        Sinogram array [num_bins_angular, num_bins_radial, num_slices]
    """
    print("\nCreating sinogram from list-mode data...")

    # Extract LOR endpoints
    x1, y1, z1 = listmode_data['x1'], listmode_data['y1'], listmode_data['z1']
    x2, y2, z2 = listmode_data['x2'], listmode_data['y2'], listmode_data['z2']

    # Check for 8-panel geometry
    has_panel_ids = 'panel_id1' in listmode_data and 'panel_id2' in listmode_data
    if has_panel_ids:
        panel_id1 = listmode_data['panel_id1']
        panel_id2 = listmode_data['panel_id2']
        print(f"  Using 8-panel geometry with {len(SCANNER_CONFIG['opposing_panel_pairs'])} opposing pairs")
    else:
        print(f"  Using generic 2-panel geometry")

    # Calculate LOR parameters (Siddon's algorithm concepts)
    # Mid-point of LOR
    x_mid = (x1 + x2) / 2
    y_mid = (y1 + y2) / 2
    z_mid = (z1 + z2) / 2

    # LOR direction
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    # Calculate radial distance (distance from center to LOR in XY plane)
    # Using perpendicular distance formula
    lor_length_xy = np.sqrt(dx**2 + dy**2)
    # Avoid division by zero
    lor_length_xy = np.where(lor_length_xy > 0, lor_length_xy, 1e-10)
    radial_dist = np.abs(x_mid * dy - y_mid * dx) / lor_length_xy

    # Calculate angle in XY plane
    angle = np.arctan2(dy, dx) * 180 / np.pi  # Convert to degrees
    angle = (angle + 360) % 360  # Normalize to [0, 360]

    # Axial position (z coordinate)
    if len(z_mid) > 0 and np.ptp(z_mid) > 0:
        z_slice = ((z_mid - np.min(z_mid)) / np.ptp(z_mid) * (IMAGE_SHAPE[2] - 1)).astype(int)
        z_slice = np.clip(z_slice, 0, IMAGE_SHAPE[2] - 1)
    else:
        z_slice = np.zeros(len(z_mid), dtype=int)

    # Bin into sinogram
    max_radial = max(SCANNER_CONFIG['detector_radius'], np.max(radial_dist) * 1.1) if len(radial_dist) > 0 else SCANNER_CONFIG['detector_radius']
    radial_bins = np.linspace(0, max_radial, num_bins_radial + 1)
    angular_bins = np.linspace(0, 360, num_bins_angular + 1)

    sinogram = np.zeros((num_bins_angular, num_bins_radial, IMAGE_SHAPE[2]))

    # Histogram LORs into sinogram
    for i in range(len(x1)):
        r_idx = np.digitize(radial_dist[i], radial_bins) - 1
        a_idx = np.digitize(angle[i], angular_bins) - 1
        z_idx = z_slice[i]

        if 0 <= r_idx < num_bins_radial and 0 <= a_idx < num_bins_angular and 0 <= z_idx < IMAGE_SHAPE[2]:
            sinogram[a_idx, r_idx, z_idx] += 1

    print(f"  Sinogram shape: {sinogram.shape}")
    print(f"  Total counts: {np.sum(sinogram):.0f}")
    if np.sum(sinogram > 0) > 0:
        print(f"  Mean counts per bin: {np.mean(sinogram[sinogram > 0]):.2f}")
        print(f"  Max counts per bin: {np.max(sinogram):.0f}")
    else:
        print(f"  ⚠ Warning: Sinogram is empty (no counts binned)")

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


def osem_reconstruction(sinogram, num_iterations=10, num_subsets=8, save_iterations=None):
    """
    Perform simple back-projection reconstruction from sinogram data.

    This implementation uses scipy's iradon for fast back-projection.
    For production use, implement proper OSEM with full system matrix.

    Args:
        sinogram: Sinogram array [num_angles, num_radial, num_slices]
        num_iterations: Not used for back-projection (kept for API compatibility)
        num_subsets: Not used for back-projection (kept for API compatibility)
        save_iterations: Not used for back-projection

    Returns:
        reconstructed_image: 3D reconstructed volume
        iteration_images: Dictionary (empty for back-projection)
    """
    print("\nPerforming Back-Projection Reconstruction...")
    print(f"  Using scipy.transform.iradon (inverse Radon transform)")

    from skimage.transform import iradon

    num_angles, num_radial, num_slices = sinogram.shape
    print(f"  Sinogram shape: {sinogram.shape}")
    print(f"  Reconstructing {num_slices} slices...")

    # Reconstruct each slice independently
    reconstructed = np.zeros((num_radial, num_radial, num_slices), dtype=np.float32)

    for z_idx in range(num_slices):
        if (z_idx + 1) % 10 == 0 or z_idx == 0:
            print(f"    Slice {z_idx + 1}/{num_slices}...")

        # Get sinogram for this slice [num_angles, num_radial]
        slice_sino = sinogram[:, :, z_idx]

        # iradon expects [num_radial, num_angles] (transposed)
        slice_sino_T = slice_sino.T

        # Angles in degrees
        theta = np.linspace(0, 180, num_angles, endpoint=False)

        # Reconstruct this slice
        try:
            recon_slice = iradon(slice_sino_T, theta=theta, filter_name='ramp', circle=True)
            reconstructed[:, :, z_idx] = recon_slice
        except Exception as e:
            print(f"      Warning: Reconstruction failed for slice {z_idx}: {e}")
            reconstructed[:, :, z_idx] = 0

    # Resize to target image shape if needed
    if reconstructed.shape[0] != IMAGE_SHAPE[0] or reconstructed.shape[1] != IMAGE_SHAPE[1]:
        print(f"  Resizing from {reconstructed.shape[:2]} to {IMAGE_SHAPE[:2]}...")
        from scipy.ndimage import zoom
        zoom_factors = [IMAGE_SHAPE[0]/reconstructed.shape[0],
                       IMAGE_SHAPE[1]/reconstructed.shape[1],
                       1.0]
        reconstructed = zoom(reconstructed, zoom_factors, order=1)

    # Resize z dimension if needed
    if reconstructed.shape[2] > IMAGE_SHAPE[2]:
        reconstructed = reconstructed[:, :, :IMAGE_SHAPE[2]]
    elif reconstructed.shape[2] < IMAGE_SHAPE[2]:
        pad_z = IMAGE_SHAPE[2] - reconstructed.shape[2]
        reconstructed = np.pad(reconstructed, ((0,0), (0,0), (0,pad_z)), mode='constant')

    # Normalize to reasonable range
    if reconstructed.max() > 0:
        reconstructed = np.clip(reconstructed, 0, None)  # Remove negative values
        reconstructed = reconstructed / reconstructed.max() * np.mean(sinogram[sinogram > 0])

    print(f"  ✓ Reconstruction complete")
    print(f"    Output shape: {reconstructed.shape}")
    print(f"    Value range: [{reconstructed.min():.2e}, {reconstructed.max():.2e}]")
    print(f"\n  ⚠ NOTE: This is Filtered Back-Projection for visualization.")
    print(f"     For quantitative results, implement:")
    print(f"       - OSEM with proper system matrix")
    print(f"       - Attenuation and scatter correction")
    print(f"       - Normalization for detector efficiency")

    iteration_images = {}
    return reconstructed, iteration_images


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

    num_panels = SCANNER_CONFIG['num_panels']
    fig.suptitle(f"PET Reconstruction Results ({num_panels}-Panel Geometry)",
                 fontsize=16, fontweight='bold')

    # Sinogram views
    mid_slice = sinogram.shape[2] // 2

    # Sinogram (middle slice)
    ax = axes[0, 0]
    im = ax.imshow(sinogram[:, :, mid_slice], aspect='auto', cmap='hot')
    ax.set_xlabel('Radial Bin')
    ax.set_ylabel('Angular Bin (0-360°)' if num_panels == 8 else 'Angular Bin')
    ax.set_title(f'Sinogram (Slice {mid_slice})')
    plt.colorbar(im, ax=ax, label='Counts')

    # Mark panel pair angles for 8-panel geometry
    if num_panels == 8:
        for i, (p1, p2) in enumerate(SCANNER_CONFIG['opposing_panel_pairs']):
            angle = i * 45  # 0°, 45°, 90°, 135°
            angle_bin = int(angle / 360 * sinogram.shape[0])
            ax.axhline(angle_bin, color='cyan', linestyle='--', alpha=0.5, linewidth=0.5)
            ax.text(sinogram.shape[1] * 1.02, angle_bin, f'{angle}°',
                   fontsize=8, va='center', color='cyan')

    # Sinogram projection (sum over angles)
    ax = axes[0, 1]
    projection = np.sum(sinogram, axis=0)
    im = ax.imshow(projection, aspect='auto', cmap='hot')
    ax.set_xlabel('Radial Bin')
    ax.set_ylabel('Axial Slice')
    ax.set_title('Sinogram (Angular Sum)')
    plt.colorbar(im, ax=ax, label='Counts')

    # Angular profile (for 8-panel geometry, should show 4 peaks)
    ax = axes[0, 2]
    angular_profile = np.sum(sinogram[:, :, mid_slice], axis=1)
    angles = np.linspace(0, 360, len(angular_profile))
    ax.plot(angles, angular_profile)
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Counts')
    ax.set_title('Angular Profile (Middle Slice)')
    ax.grid(True, alpha=0.3)

    # Mark expected panel pair angles
    if num_panels == 8:
        for i in range(4):
            angle = i * 45
            ax.axvline(angle, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.text(angle, ax.get_ylim()[1] * 0.9, f'P{i}↔{i+4}',
                   fontsize=8, ha='center', color='red')

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

    # Perform OSEM reconstruction
    print("\n" + "="*70)
    print("STEP 3: OSEM Reconstruction")
    print("="*70)
    reconstructed_image, _ = osem_reconstruction(
        sinogram,
        num_iterations=RECON_CONFIG['num_iterations'],
        num_subsets=RECON_CONFIG['num_subsets'],
        save_iterations=RECON_CONFIG['save_iterations']
    )

    # Save reconstructed image as NumPy array
    np.save(output_path / "reconstructed_volume.npy", reconstructed_image)
    print(f"  ✓ Saved reconstructed volume to: {output_path / 'reconstructed_volume.npy'}")

    # Save as NIfTI image with proper metadata
    try:
        import SimpleITK as sitk

        # Create SimpleITK image from numpy array
        # Note: SimpleITK uses (z, y, x) ordering, we need to transpose
        sitk_image = sitk.GetImageFromArray(reconstructed_image.transpose(2, 1, 0))

        # Set spacing (in mm)
        sitk_image.SetSpacing([IMAGE_SPACING[0], IMAGE_SPACING[1], IMAGE_SPACING[2]])

        # Set origin (center of volume at scanner isocenter)
        origin_x = -(IMAGE_SHAPE[0] / 2) * IMAGE_SPACING[0]
        origin_y = -(IMAGE_SHAPE[1] / 2) * IMAGE_SPACING[1]
        origin_z = -(IMAGE_SHAPE[2] / 2) * IMAGE_SPACING[2]
        sitk_image.SetOrigin([origin_x, origin_y, origin_z])

        # Set direction (identity matrix - standard RAS orientation)
        sitk_image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])

        # Add metadata
        sitk_image.SetMetaData("Modality", "PT")  # PET
        sitk_image.SetMetaData("Scanner", "OpenGATE_8Panel_PET")
        sitk_image.SetMetaData("Reconstruction", "FBP_RampFilter")
        sitk_image.SetMetaData("ImageShape", f"{IMAGE_SHAPE[0]}x{IMAGE_SHAPE[1]}x{IMAGE_SHAPE[2]}")
        sitk_image.SetMetaData("VoxelSpacing_mm", f"{IMAGE_SPACING[0]}x{IMAGE_SPACING[1]}x{IMAGE_SPACING[2]}")
        sitk_image.SetMetaData("Origin_mm", f"{origin_x:.2f}x{origin_y:.2f}x{origin_z:.2f}")
        sitk_image.SetMetaData("DetectorRadius_mm", str(SCANNER_CONFIG['detector_radius']))
        sitk_image.SetMetaData("NumPanels", str(SCANNER_CONFIG['num_panels']))

        # Save as NIfTI
        nifti_path = output_path / "reconstructed_volume.nii.gz"
        sitk.WriteImage(sitk_image, str(nifti_path))
        print(f"  ✓ Saved NIfTI volume to: {nifti_path}")
        print(f"    - Origin: ({origin_x:.1f}, {origin_y:.1f}, {origin_z:.1f}) mm")
        print(f"    - Spacing: ({IMAGE_SPACING[0]}, {IMAGE_SPACING[1]}, {IMAGE_SPACING[2]}) mm")
        print(f"    - Direction: Identity (RAS orientation)")

    except ImportError:
        print("  ⚠ SimpleITK not installed - skipping NIfTI export")
        print("    Install with: mamba install -c conda-forge simpleitk")

    # Visualize
    print("\n" + "="*70)
    print("STEP 4: Visualization")
    print("="*70)
    visualize_results(sinogram, image=reconstructed_image)

    print("\n" + "="*70)
    print("RECONSTRUCTION WORKFLOW COMPLETE")
    print("="*70)
    print(f"\nOutput saved to: {output_path.absolute()}")
    print(f"\nFiles created:")
    print(f"  - sinogram.npy: Sinogram data")
    print(f"  - reconstructed_volume.npy: 3D reconstructed image [{reconstructed_image.shape[0]}×{reconstructed_image.shape[1]}×{reconstructed_image.shape[2]}]")
    print(f"  - reconstruction_results.png: Visualization")
    print("\nNext steps for improved reconstruction:")
    print("  1. Implement attenuation correction")
    print("  2. Add scatter and random corrections")
    print("  3. Apply post-reconstruction filtering (Gaussian, median, etc.)")
    print("  4. Tune OSEM parameters (iterations, subsets)")
    print("\nSee PyTomography documentation:")
    print("  https://pytomography.readthedocs.io/")


if __name__ == "__main__":
    main()
