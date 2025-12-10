#!/usr/bin/env python3
"""
PET Event Processing and Reconstruction Preparation

This script processes ROOT files from GATE/OpenGATE simulations and prepares
data for tomographic reconstruction using PyTomography.

Based on established PyTomography workflows:
- https://pytomography.readthedocs.io/
- https://github.com/PyTomography/PyTomography

Requirements:
    pip install pytomography uproot awkward matplotlib numpy torch
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

try:
    import uproot
except ImportError:
    print("ERROR: uproot not installed. Install with: pip install uproot")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("WARNING: torch not installed. Install with: pip install torch")
    torch = None

try:
    from pytomography.io.PET import gate
    from pytomography.metadata import PETScannerParams
    from pytomography.projectors import PETSystemMatrix
    from pytomography.algorithms import OSEM
    HAS_PYTOMOGRAPHY = True
except ImportError:
    print("WARNING: pytomography not installed.")
    print("For full reconstruction: pip install pytomography")
    HAS_PYTOMOGRAPHY = False


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# File paths
ROOT_FILE = "../output/events.root"
GEOMETRY_SCRIPT = "geometry_pet.py"
OUTPUT_DIR = "../output/processed"

# Coincidence detection parameters
COINCIDENCE_WINDOW = 10.0  # ns (10 ns timing window)
ENERGY_WINDOW_LOW = 400  # keV (lower energy threshold)
ENERGY_WINDOW_HIGH = 600  # keV (upper energy threshold)

# Reconstruction parameters
NUM_SUBSETS = 8
NUM_ITERATIONS = 10
IMAGE_SIZE = [128, 128, 64]  # [x, y, z] voxels


# ==============================================================================
# EVENT PROCESSING FUNCTIONS
# ==============================================================================

class GATEEventProcessor:
    """Process GATE ROOT files for PET reconstruction."""

    def __init__(self, root_file_path):
        """
        Initialize event processor.

        Args:
            root_file_path: Path to ROOT file from GATE simulation
        """
        self.root_file_path = Path(root_file_path)
        self.hits_data = None
        self.coincidences = None
        self.valid_events = None

        if not self.root_file_path.exists():
            raise FileNotFoundError(f"ROOT file not found: {root_file_path}")

    def load_hits(self, tree_name="Hits"):
        """
        Load hits data from ROOT file.

        Args:
            tree_name: Name of the tree in ROOT file (default: "Hits")

        Returns:
            Dictionary containing hit data arrays
        """
        print(f"Loading hits from {self.root_file_path}...")

        try:
            with uproot.open(self.root_file_path) as file:
                # List available trees
                print(f"Available trees: {file.keys()}")

                if tree_name not in file:
                    print(f"Warning: '{tree_name}' tree not found.")
                    print("Attempting to find alternative tree names...")
                    # Try common alternatives
                    for alt_name in ["Hits", "Reads", "Singles", "Coincidences"]:
                        if alt_name in file:
                            tree_name = alt_name
                            print(f"Using tree: {tree_name}")
                            break
                    else:
                        raise KeyError(f"Could not find hits tree in ROOT file")

                tree = file[tree_name]

                # Extract relevant branches
                self.hits_data = {
                    'event_id': tree["EventID"].array(library="np") if "EventID" in tree else None,
                    'energy': tree["TotalEnergyDeposit"].array(library="np"),
                    'time': tree["GlobalTime"].array(library="np"),
                    'position_x': tree["PostPosition.fX"].array(library="np"),
                    'position_y': tree["PostPosition.fY"].array(library="np"),
                    'position_z': tree["PostPosition.fZ"].array(library="np"),
                    'volume_id': tree["PreStepUniqueVolumeID"].array(library="np") if "PreStepUniqueVolumeID" in tree else None,
                }

                # Remove None entries
                self.hits_data = {k: v for k, v in self.hits_data.items() if v is not None}

                print(f"Loaded {len(self.hits_data['energy'])} hits")
                print(f"Energy range: {np.min(self.hits_data['energy']):.2f} - {np.max(self.hits_data['energy']):.2f} keV")
                print(f"Time range: {np.min(self.hits_data['time']):.2f} - {np.max(self.hits_data['time']):.2f} ns")

                return self.hits_data

        except Exception as e:
            print(f"Error loading ROOT file: {e}")
            raise

    def apply_energy_window(self, energy_low=ENERGY_WINDOW_LOW, energy_high=ENERGY_WINDOW_HIGH):
        """
        Apply energy window to filter valid events.

        Args:
            energy_low: Lower energy threshold (keV)
            energy_high: Upper energy threshold (keV)

        Returns:
            Mask of valid events
        """
        if self.hits_data is None:
            raise ValueError("No hits data loaded. Call load_hits() first.")

        energy = self.hits_data['energy']

        # Apply energy window
        mask = (energy >= energy_low) & (energy <= energy_high)

        print(f"\nEnergy window filtering:")
        print(f"  Window: {energy_low} - {energy_high} keV")
        print(f"  Events before: {len(energy)}")
        print(f"  Events after: {np.sum(mask)}")
        print(f"  Acceptance: {100 * np.sum(mask) / len(energy):.2f}%")

        return mask

    def detect_coincidences(self, time_window=COINCIDENCE_WINDOW):
        """
        Detect coincidence events within time window.

        Args:
            time_window: Coincidence time window in ns

        Returns:
            Array of coincidence pairs (indices into hits_data)
        """
        if self.hits_data is None:
            raise ValueError("No hits data loaded. Call load_hits() first.")

        print(f"\nDetecting coincidences (window: {time_window} ns)...")

        # First apply energy window
        valid_mask = self.apply_energy_window()
        valid_indices = np.where(valid_mask)[0]

        times = self.hits_data['time'][valid_mask]
        energies = self.hits_data['energy'][valid_mask]

        coincidences = []

        # Sort by time for efficient searching
        time_sorted_idx = np.argsort(times)
        times_sorted = times[time_sorted_idx]

        # Find coincidences
        for i in range(len(times_sorted) - 1):
            t1 = times_sorted[i]

            # Find all events within time window
            j = i + 1
            while j < len(times_sorted) and (times_sorted[j] - t1) <= time_window:
                # Valid coincidence found
                idx1 = valid_indices[time_sorted_idx[i]]
                idx2 = valid_indices[time_sorted_idx[j]]

                coincidences.append([idx1, idx2])
                j += 1

        self.coincidences = np.array(coincidences)

        print(f"  Found {len(self.coincidences)} coincidence pairs")
        print(f"  Coincidence rate: {len(self.coincidences) / len(times) * 100:.2f}%")

        return self.coincidences

    def create_listmode_data(self):
        """
        Create list-mode data structure for PyTomography.

        Returns:
            Dictionary with list-mode event data
        """
        if self.coincidences is None:
            raise ValueError("No coincidences detected. Call detect_coincidences() first.")

        print("\nCreating list-mode data structure...")

        # Extract coincidence event data
        idx1 = self.coincidences[:, 0]
        idx2 = self.coincidences[:, 1]

        listmode_data = {
            # Detector 1 coordinates
            'x1': self.hits_data['position_x'][idx1],
            'y1': self.hits_data['position_y'][idx1],
            'z1': self.hits_data['position_z'][idx1],

            # Detector 2 coordinates
            'x2': self.hits_data['position_x'][idx2],
            'y2': self.hits_data['position_y'][idx2],
            'z2': self.hits_data['position_z'][idx2],

            # Event properties
            'energy1': self.hits_data['energy'][idx1],
            'energy2': self.hits_data['energy'][idx2],
            'time1': self.hits_data['time'][idx1],
            'time2': self.hits_data['time'][idx2],
            'time_diff': self.hits_data['time'][idx2] - self.hits_data['time'][idx1],
        }

        # Calculate LOR (Line of Response) properties
        dx = listmode_data['x2'] - listmode_data['x1']
        dy = listmode_data['y2'] - listmode_data['y1']
        dz = listmode_data['z2'] - listmode_data['z1']

        listmode_data['lor_length'] = np.sqrt(dx**2 + dy**2 + dz**2)

        print(f"  List-mode events: {len(idx1)}")
        print(f"  Mean LOR length: {np.mean(listmode_data['lor_length']):.2f} mm")
        print(f"  Mean time difference: {np.mean(np.abs(listmode_data['time_diff'])):.3f} ns")

        return listmode_data

    def save_processed_data(self, output_dir=OUTPUT_DIR):
        """
        Save processed data to files.

        Args:
            output_dir: Directory to save output files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving processed data to {output_path}...")

        # Save hits data
        if self.hits_data is not None:
            np.savez(
                output_path / "hits_data.npz",
                **self.hits_data
            )
            print(f"  ✓ Saved hits data")

        # Save coincidences
        if self.coincidences is not None:
            np.save(output_path / "coincidences.npy", self.coincidences)
            print(f"  ✓ Saved coincidence pairs")

        # Save list-mode data
        try:
            listmode = self.create_listmode_data()
            np.savez(output_path / "listmode_data.npz", **listmode)
            print(f"  ✓ Saved list-mode data")
        except Exception as e:
            print(f"  ✗ Could not create list-mode data: {e}")

        print(f"\nAll data saved to: {output_path.absolute()}")

    def plot_statistics(self, save_fig=True):
        """
        Plot event statistics and quality metrics.

        Args:
            save_fig: If True, save figure to file
        """
        if self.hits_data is None:
            raise ValueError("No data to plot. Load hits first.")

        print("\nGenerating plots...")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("PET Event Statistics", fontsize=16, fontweight='bold')

        # Energy histogram
        ax = axes[0, 0]
        ax.hist(self.hits_data['energy'], bins=100, alpha=0.7, edgecolor='black')
        ax.axvline(ENERGY_WINDOW_LOW, color='r', linestyle='--', label='Lower threshold')
        ax.axvline(ENERGY_WINDOW_HIGH, color='r', linestyle='--', label='Upper threshold')
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel('Counts')
        ax.set_title('Energy Spectrum')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Time histogram
        ax = axes[0, 1]
        ax.hist(self.hits_data['time'], bins=100, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Counts')
        ax.set_title('Time Distribution')
        ax.grid(True, alpha=0.3)

        # Spatial distribution (XY plane)
        ax = axes[0, 2]
        ax.scatter(self.hits_data['position_x'], self.hits_data['position_y'],
                   alpha=0.1, s=1)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('Hit Positions (XY Plane)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Spatial distribution (XZ plane)
        ax = axes[1, 0]
        ax.scatter(self.hits_data['position_x'], self.hits_data['position_z'],
                   alpha=0.1, s=1)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Z (mm)')
        ax.set_title('Hit Positions (XZ Plane)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Coincidence time differences (if available)
        if self.coincidences is not None:
            ax = axes[1, 1]
            listmode = self.create_listmode_data()
            ax.hist(listmode['time_diff'], bins=100, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Time Difference (ns)')
            ax.set_ylabel('Counts')
            ax.set_title('Coincidence Time Differences')
            ax.grid(True, alpha=0.3)

            # LOR length distribution
            ax = axes[1, 2]
            ax.hist(listmode['lor_length'], bins=100, alpha=0.7, edgecolor='black')
            ax.set_xlabel('LOR Length (mm)')
            ax.set_ylabel('Counts')
            ax.set_title('Line of Response Length')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_fig:
            output_path = Path(OUTPUT_DIR)
            output_path.mkdir(parents=True, exist_ok=True)
            fig_path = output_path / "event_statistics.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved figure: {fig_path}")

        plt.show()


# ==============================================================================
# PYTOMOGRAPHY RECONSTRUCTION
# ==============================================================================

def reconstruct_with_pytomography(listmode_file, geometry_script=GEOMETRY_SCRIPT):
    """
    Perform PET reconstruction using PyTomography.

    Args:
        listmode_file: Path to list-mode data file (.npz)
        geometry_script: Path to geometry definition script

    Returns:
        Reconstructed image
    """
    if not HAS_PYTOMOGRAPHY:
        print("ERROR: PyTomography not installed.")
        print("Install with: pip install pytomography")
        return None

    print("\n" + "="*70)
    print("PET RECONSTRUCTION WITH PYTOMOGRAPHY")
    print("="*70)

    # Load list-mode data
    print(f"\nLoading list-mode data from {listmode_file}...")
    data = np.load(listmode_file)

    # Extract scanner geometry from GATE script
    # Note: This requires the geometry_pet.py to be a GATE macro file
    # For OpenGATE Python scripts, we'll use manual configuration
    print("\nConfiguring scanner geometry...")

    # Manual scanner configuration (adjust based on your geometry)
    # These values should match your geometry_pet.py configuration
    scanner_params = {
        'scanner_type': 'generic',  # Generic PET scanner
        'radius': 100.0,  # mm (FOV_RADIUS from geometry)
        'num_crystals_x': 50,  # From PETGeometry
        'num_crystals_y': 50,
        'crystal_size_x': 2.0,  # mm
        'crystal_size_y': 2.0,  # mm
        'crystal_size_z': 19.0,  # mm
        'num_panels': 2,  # Two-panel geometry
    }

    print(f"  Scanner type: {scanner_params['scanner_type']}")
    print(f"  Detector radius: {scanner_params['radius']} mm")
    print(f"  Crystal array: {scanner_params['num_crystals_x']}×{scanner_params['num_crystals_y']}")

    # Create system matrix
    print("\nCreating system matrix...")
    # Note: Actual implementation depends on PyTomography version
    # This is a template - adjust based on your PyTomography installation

    print("  WARNING: Full PyTomography reconstruction requires:")
    print("    1. Scanner geometry configuration (PETScannerParams)")
    print("    2. System matrix creation (PETSystemMatrix)")
    print("    3. OSEM reconstruction algorithm")
    print("\n  Please refer to PyTomography documentation:")
    print("  https://pytomography.readthedocs.io/")

    return None


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main processing workflow."""

    print("="*70)
    print("PET EVENT PROCESSING PIPELINE")
    print("="*70)

    # Check if ROOT file exists
    root_path = Path(ROOT_FILE)
    if not root_path.exists():
        print(f"\nERROR: ROOT file not found: {root_path}")
        print("Please run a GATE simulation first to generate events.")
        print("Example: python scripts/example_ring_source.py")
        return

    # Initialize processor
    processor = GATEEventProcessor(ROOT_FILE)

    # Step 1: Load hits
    print("\n" + "="*70)
    print("STEP 1: Loading Event Data")
    print("="*70)
    processor.load_hits()

    # Step 2: Detect coincidences
    print("\n" + "="*70)
    print("STEP 2: Coincidence Detection")
    print("="*70)
    processor.detect_coincidences(time_window=COINCIDENCE_WINDOW)

    # Step 3: Save processed data
    print("\n" + "="*70)
    print("STEP 3: Saving Processed Data")
    print("="*70)
    processor.save_processed_data()

    # Step 4: Generate plots
    print("\n" + "="*70)
    print("STEP 4: Statistical Analysis")
    print("="*70)
    processor.plot_statistics(save_fig=True)

    # Step 5: Reconstruction (if PyTomography available)
    if HAS_PYTOMOGRAPHY:
        print("\n" + "="*70)
        print("STEP 5: Image Reconstruction")
        print("="*70)
        listmode_path = Path(OUTPUT_DIR) / "listmode_data.npz"
        if listmode_path.exists():
            reconstruct_with_pytomography(listmode_path)
    else:
        print("\n" + "="*70)
        print("STEP 5: Image Reconstruction - SKIPPED")
        print("="*70)
        print("Install PyTomography for reconstruction:")
        print("  pip install pytomography torch")

    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70)
    print(f"\nProcessed data saved to: {Path(OUTPUT_DIR).absolute()}")
    print("\nNext steps:")
    print("  1. Review event_statistics.png")
    print("  2. Load listmode_data.npz for reconstruction")
    print("  3. Use PyTomography for OSEM/BSREM reconstruction")


if __name__ == "__main__":
    main()
