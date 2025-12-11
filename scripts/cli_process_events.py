#!/usr/bin/env python3
"""
CLI: PET Event Processing

Command-line interface for processing GATE ROOT files and preparing data
for tomographic reconstruction.

IMPORTANT: ROOT files store energy in MeV, but this tool uses keV.
           Energy conversion (MeV → keV) is automatic during loading.
           All energy parameters are specified in keV.

Usage:
    cli_process_events.py [OPTIONS] INPUT_ROOT_FILE

Example:
    # Basic processing (400-600 keV window)
    cli_process_events.py output/events.root

    # Custom energy window (in keV)
    cli_process_events.py --energy-min 350 --energy-max 650 output/events.root

    # Tight photopeak window (480-540 keV around 511 keV)
    cli_process_events.py --energy-min 480 --energy-max 540 output/events.root

    # Adjust coincidence window
    cli_process_events.py --time-window 5.0 output/events.root
"""

import sys
from pathlib import Path

import click
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

try:
    import uproot
except ImportError:
    click.echo("ERROR: uproot not installed. Install with: pip install uproot", err=True)
    sys.exit(1)


# ==============================================================================
# CORE PROCESSING CLASS
# ==============================================================================

class GATEEventProcessor:
    """Process GATE ROOT files for PET reconstruction."""

    def __init__(self, root_file_path, verbose=True):
        self.root_file_path = Path(root_file_path)
        self.hits_data = None
        self.coincidences = None
        self.verbose = verbose

        if not self.root_file_path.exists():
            raise FileNotFoundError(f"ROOT file not found: {root_file_path}")

    def log(self, message):
        """Print message if verbose mode enabled."""
        if self.verbose:
            click.echo(message)

    @staticmethod
    def list_all_keys(directory, indent=0):
        prefix = "  " * indent
        for key, obj in directory.items():
            click.echo(f"{prefix}- {key}")
            if isinstance(obj, uproot.ReadOnlyDirectory):
                GATEEventProcessor.list_all_keys(obj, indent + 1)

    def load_hits(self, tree_name="Hits"):
        """Load hits data from ROOT file."""
        self.log(f"Loading hits from {self.root_file_path.name}...")

        try:
            with uproot.open(self.root_file_path) as file:
                # Find hits tree
                if tree_name not in file:
                    self.log(f"Warning: '{tree_name}' tree not found. Available trees:")
                    self.list_all_keys(file)
                    
                    self.log("Searching for alternative tree names...")
                    for alt_name in ["Hits", "Reads", "Singles", "Coincidences"]:
                        if alt_name in file:
                            tree_name = alt_name
                            self.log(f"Using tree: {tree_name}")
                            break
                    else:
                        raise KeyError(f"Could not find hits tree in ROOT file")

                tree = file[tree_name]

                # Extract branches
                self.hits_data = {
                    'event_id': tree["EventID"].array(library="np") if "EventID" in tree else None,
                    'energy': tree["TotalEnergyDeposit"].array(library="np"),  # MeV in ROOT
                    'time': tree["GlobalTime"].array(library="np"),
                    'position_x': tree["PostPosition_X"].array(library="np"),
                    'position_y': tree["PostPosition_Y"].array(library="np"),
                    'position_z': tree["PostPosition_Z"].array(library="np"),
                    'volume_id': tree["PreStepUniqueVolumeID"].array(library="np") if "PreStepUniqueVolumeID" in tree else None,
                }

                # Remove None entries
                self.hits_data = {k: v for k, v in self.hits_data.items() if v is not None}

                # Convert energy from MeV to keV (ROOT stores in MeV, PET uses keV)
                self.hits_data['energy'] = self.hits_data['energy'] * 1000.0

                self.log(f"  ✓ Loaded {len(self.hits_data['energy'])} hits")
                self.log(f"  Energy: {np.min(self.hits_data['energy']):.2f} - {np.max(self.hits_data['energy']):.2f} keV")
                self.log(f"  Time: {np.min(self.hits_data['time']):.2f} - {np.max(self.hits_data['time']):.2f} ns")

                return self.hits_data

        except Exception as e:
            raise click.ClickException(f"Error loading ROOT file: {e}")

    def apply_energy_window(self, energy_low, energy_high):
        """
        Apply energy window filter.

        Args:
            energy_low: Lower threshold in keV
            energy_high: Upper threshold in keV

        Returns:
            Boolean mask of valid events

        Note:
            Energy in self.hits_data is already converted to keV from MeV
        """
        if self.hits_data is None:
            raise ValueError("No hits data loaded.")

        energy = self.hits_data['energy']  # Already in keV
        mask = (energy >= energy_low) & (energy <= energy_high)

        self.log(f"\nEnergy window: {energy_low:.1f} - {energy_high:.1f} keV")
        self.log(f"  Before: {len(energy)} events")
        self.log(f"  After: {np.sum(mask)} events")
        self.log(f"  Acceptance: {100 * np.sum(mask) / len(energy):.2f}%")

        return mask

    def detect_coincidences(self, time_window, energy_low=400, energy_high=600):
        """
        Detect coincidence events.

        Args:
            time_window: Coincidence time window in ns
            energy_low: Lower energy threshold in keV (default: 400)
            energy_high: Upper energy threshold in keV (default: 600)
        """
        if self.hits_data is None:
            raise ValueError("No hits data loaded.")

        self.log(f"\nDetecting coincidences (time window: {time_window} ns)...")

        # Apply energy window first
        valid_mask = self.apply_energy_window(energy_low, energy_high)
        valid_indices = np.where(valid_mask)[0]

        times = self.hits_data['time'][valid_mask]

        coincidences = []

        # Time-sorted search
        time_sorted_idx = np.argsort(times)
        times_sorted = times[time_sorted_idx]

        with click.progressbar(length=len(times_sorted)-1,
                              label='Processing coincidences') as bar:
            for i in range(len(times_sorted) - 1):
                t1 = times_sorted[i]
                j = i + 1
                while j < len(times_sorted) and (times_sorted[j] - t1) <= time_window:
                    idx1 = valid_indices[time_sorted_idx[i]]
                    idx2 = valid_indices[time_sorted_idx[j]]
                    coincidences.append([idx1, idx2])
                    j += 1
                bar.update(1)

        self.coincidences = np.array(coincidences)

        self.log(f"  ✓ Found {len(self.coincidences)} coincidence pairs")
        self.log(f"  Coincidence rate: {len(self.coincidences) / len(times) * 100:.2f}%")

        return self.coincidences

    def create_listmode_data(self):
        """Create list-mode data structure."""
        if self.coincidences is None:
            raise ValueError("No coincidences detected.")

        self.log("\nCreating list-mode data...")

        idx1 = self.coincidences[:, 0]
        idx2 = self.coincidences[:, 1]

        listmode_data = {
            'x1': self.hits_data['position_x'][idx1],
            'y1': self.hits_data['position_y'][idx1],
            'z1': self.hits_data['position_z'][idx1],
            'x2': self.hits_data['position_x'][idx2],
            'y2': self.hits_data['position_y'][idx2],
            'z2': self.hits_data['position_z'][idx2],
            'energy1': self.hits_data['energy'][idx1],
            'energy2': self.hits_data['energy'][idx2],
            'time1': self.hits_data['time'][idx1],
            'time2': self.hits_data['time'][idx2],
            'time_diff': self.hits_data['time'][idx2] - self.hits_data['time'][idx1],
        }

        # LOR properties
        dx = listmode_data['x2'] - listmode_data['x1']
        dy = listmode_data['y2'] - listmode_data['y1']
        dz = listmode_data['z2'] - listmode_data['z1']
        listmode_data['lor_length'] = np.sqrt(dx**2 + dy**2 + dz**2)

        self.log(f"  ✓ {len(idx1)} list-mode events")
        self.log(f"  Mean LOR length: {np.mean(listmode_data['lor_length']):.2f} mm")

        return listmode_data

    def save_processed_data(self, output_dir):
        """Save processed data."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.log(f"\nSaving to {output_path}...")

        if self.hits_data is not None:
            np.savez(output_path / "hits_data.npz", **self.hits_data)
            self.log("  ✓ hits_data.npz")

        if self.coincidences is not None:
            np.save(output_path / "coincidences.npy", self.coincidences)
            self.log("  ✓ coincidences.npy")

        try:
            listmode = self.create_listmode_data()
            np.savez(output_path / "listmode_data.npz", **listmode)
            self.log("  ✓ listmode_data.npz")
        except Exception as e:
            self.log(f"  ✗ Could not create list-mode: {e}")

    def plot_statistics(self, output_path):
        """Generate statistics plots."""
        if self.hits_data is None:
            return

        self.log("\nGenerating plots...")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("PET Event Statistics", fontsize=16, fontweight='bold')

        # Energy histogram
        ax = axes[0, 0]
        ax.hist(self.hits_data['energy'], bins=100, alpha=0.7, edgecolor='black')
        ax.axvline(400, color='r', linestyle='--', label='Lower threshold')
        ax.axvline(600, color='r', linestyle='--', label='Upper threshold')
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

        # Spatial XY
        ax = axes[0, 2]
        ax.scatter(self.hits_data['position_x'], self.hits_data['position_y'],
                   alpha=0.1, s=1)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('Hit Positions (XY)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Spatial XZ
        ax = axes[1, 0]
        ax.scatter(self.hits_data['position_x'], self.hits_data['position_z'],
                   alpha=0.1, s=1)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Z (mm)')
        ax.set_title('Hit Positions (XZ)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Coincidence time differences
        if self.coincidences is not None:
            ax = axes[1, 1]
            listmode = self.create_listmode_data()
            ax.hist(listmode['time_diff'], bins=100, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Time Difference (ns)')
            ax.set_ylabel('Counts')
            ax.set_title('Coincidence Timing')
            ax.grid(True, alpha=0.3)

            # LOR lengths
            ax = axes[1, 2]
            ax.hist(listmode['lor_length'], bins=100, alpha=0.7, edgecolor='black')
            ax.set_xlabel('LOR Length (mm)')
            ax.set_ylabel('Counts')
            ax.set_title('Line of Response')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        self.log(f"  ✓ {output_path.name}")
        plt.close()


# ==============================================================================
# DETECTOR CAMERA VIEW VISUALIZATION
# ==============================================================================

class DetectorCameraView:
    """Visualize PET events as detector camera views."""

    def __init__(self, data_source, verbose=True):
        """
        Initialize with event data.

        Args:
            data_source: Either:
                - Dictionary with listmode data (x1, y1, z1, x2, y2, z2, etc.)
                - GATEEventProcessor instance (direct access, more efficient)
            verbose: Enable verbose output
        """
        self.verbose = verbose

        # Support both direct processor access and listmode dictionary
        if isinstance(data_source, GATEEventProcessor):
            # Direct access - no intermediate conversion (efficient)
            self._init_from_processor(data_source)
        else:
            # Listmode dictionary - for standalone use or loaded files
            self._init_from_listmode(data_source)

        self._analyze_geometry()

    def _init_from_processor(self, processor):
        """Initialize directly from GATEEventProcessor (efficient)."""
        if processor.coincidences is None:
            raise ValueError("No coincidences detected in processor")

        idx1 = processor.coincidences[:, 0]
        idx2 = processor.coincidences[:, 1]

        # Direct extraction - skips intermediate listmode creation
        self.detector1 = {
            'x': processor.hits_data['position_x'][idx1],
            'y': processor.hits_data['position_y'][idx1],
            'z': processor.hits_data['position_z'][idx1],
            'energy': processor.hits_data['energy'][idx1],
            'time': processor.hits_data['time'][idx1]
        }
        self.detector2 = {
            'x': processor.hits_data['position_x'][idx2],
            'y': processor.hits_data['position_y'][idx2],
            'z': processor.hits_data['position_z'][idx2],
            'energy': processor.hits_data['energy'][idx2],
            'time': processor.hits_data['time'][idx2]
        }

    def _init_from_listmode(self, listmode_data):
        """Initialize from listmode dictionary (for standalone/file loading)."""
        self.detector1 = {
            'x': listmode_data['x1'],
            'y': listmode_data['y1'],
            'z': listmode_data['z1'],
            'energy': listmode_data['energy1'],
            'time': listmode_data['time1']
        }
        self.detector2 = {
            'x': listmode_data['x2'],
            'y': listmode_data['y2'],
            'z': listmode_data['z2'],
            'energy': listmode_data['energy2'],
            'time': listmode_data['time2']
        }

    def log(self, message):
        """Print message if verbose enabled."""
        if self.verbose:
            click.echo(message)

    def _analyze_geometry(self):
        """Analyze detector positions to determine orientation."""
        det1_mean = np.array([
            np.mean(self.detector1['x']),
            np.mean(self.detector1['y']),
            np.mean(self.detector1['z'])
        ])
        det2_mean = np.array([
            np.mean(self.detector2['x']),
            np.mean(self.detector2['y']),
            np.mean(self.detector2['z'])
        ])

        if self.verbose:
            self.log(f"  Panel 1 center: ({det1_mean[0]:.1f}, {det1_mean[1]:.1f}, {det1_mean[2]:.1f}) mm")
            self.log(f"  Panel 2 center: ({det2_mean[0]:.1f}, {det2_mean[1]:.1f}, {det2_mean[2]:.1f}) mm")

        # Determine separation axis
        separation = np.abs(det2_mean - det1_mean)
        self.separation_axis = np.argmax(separation)
        axis_names = ['X', 'Y', 'Z']

        if self.verbose:
            self.log(f"  Separation axis: {axis_names[self.separation_axis]}")
            self.log(f"  Panel distance: {separation[self.separation_axis]:.1f} mm")

    def create_camera_views(self, output_path, bins=50, colormap='viridis', log_scale=False):
        """Create detector camera view visualization."""
        fig = plt.figure(figsize=(16, 12))

        # Determine coordinates based on separation axis
        if self.separation_axis == 0:  # Separated in X
            coord1_name, coord2_name = 'Y', 'Z'
            det1_u, det1_v = self.detector1['y'], self.detector1['z']
            det2_u, det2_v = self.detector2['y'], self.detector2['z']
        elif self.separation_axis == 1:  # Separated in Y
            coord1_name, coord2_name = 'X', 'Z'
            det1_u, det1_v = self.detector1['x'], self.detector1['z']
            det2_u, det2_v = self.detector2['x'], self.detector2['z']
        else:  # Separated in Z
            coord1_name, coord2_name = 'X', 'Y'
            det1_u, det1_v = self.detector1['x'], self.detector1['y']
            det2_u, det2_v = self.detector2['x'], self.detector2['y']

        from matplotlib.colors import LogNorm
        norm = LogNorm() if log_scale else None

        # Panel 1 camera view
        ax1 = plt.subplot(2, 3, 1)
        _, _, _, im1 = ax1.hist2d(det1_u, det1_v, bins=bins, cmap=colormap, norm=norm)
        ax1.set_xlabel(f'{coord1_name} position (mm)')
        ax1.set_ylabel(f'{coord2_name} position (mm)')
        ax1.set_title('Detector Panel 1 - Camera View', fontweight='bold')
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1, label='Hit counts')

        # Panel 2 camera view
        ax2 = plt.subplot(2, 3, 2)
        _, _, _, im2 = ax2.hist2d(det2_u, det2_v, bins=bins, cmap=colormap, norm=norm)
        ax2.set_xlabel(f'{coord1_name} position (mm)')
        ax2.set_ylabel(f'{coord2_name} position (mm)')
        ax2.set_title('Detector Panel 2 - Camera View', fontweight='bold')
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2, label='Hit counts')

        # Combined view
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist2d(det1_u, det1_v, bins=bins, cmap='Reds', alpha=0.5)
        ax3.hist2d(det2_u, det2_v, bins=bins, cmap='Blues', alpha=0.5)
        ax3.set_xlabel(f'{coord1_name} position (mm)')
        ax3.set_ylabel(f'{coord2_name} position (mm)')
        ax3.set_title('Combined (Red: Panel 1, Blue: Panel 2)', fontweight='bold')
        ax3.set_aspect('equal')

        # Energy-weighted Panel 1
        ax4 = plt.subplot(2, 3, 4)
        _, _, _, im4 = ax4.hist2d(det1_u, det1_v, bins=bins,
                                  weights=self.detector1['energy'],
                                  cmap=colormap, norm=norm)
        ax4.set_xlabel(f'{coord1_name} position (mm)')
        ax4.set_ylabel(f'{coord2_name} position (mm)')
        ax4.set_title('Panel 1 - Energy Weighted', fontweight='bold')
        ax4.set_aspect('equal')
        plt.colorbar(im4, ax=ax4, label='Total energy (keV)')

        # Energy-weighted Panel 2
        ax5 = plt.subplot(2, 3, 5)
        _, _, _, im5 = ax5.hist2d(det2_u, det2_v, bins=bins,
                                  weights=self.detector2['energy'],
                                  cmap=colormap, norm=norm)
        ax5.set_xlabel(f'{coord1_name} position (mm)')
        ax5.set_ylabel(f'{coord2_name} position (mm)')
        ax5.set_title('Panel 2 - Energy Weighted', fontweight='bold')
        ax5.set_aspect('equal')
        plt.colorbar(im5, ax=ax5, label='Total energy (keV)')

        # LOR projection
        ax6 = plt.subplot(2, 3, 6)
        n_lors = min(1000, len(det1_u))
        sample_idx = np.random.choice(len(det1_u), n_lors, replace=False)

        for idx in sample_idx:
            ax6.plot([det1_u[idx], det2_u[idx]],
                    [det1_v[idx], det2_v[idx]],
                    'b-', alpha=0.02, linewidth=0.5)

        ax6.scatter(det1_u[sample_idx], det1_v[sample_idx],
                   c='red', s=10, alpha=0.3, label='Panel 1')
        ax6.scatter(det2_u[sample_idx], det2_v[sample_idx],
                   c='blue', s=10, alpha=0.3, label='Panel 2')

        ax6.set_xlabel(f'{coord1_name} position (mm)')
        ax6.set_ylabel(f'{coord2_name} position (mm)')
        ax6.set_title(f'Lines of Response ({n_lors} samples)', fontweight='bold')
        ax6.set_aspect('equal')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.suptitle('PET Detector Camera Views', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.log(f"  ✓ {output_path.name}")


# ==============================================================================
# CLI INTERFACE
# ==============================================================================

from trogon import tui

@tui()
@click.command()
@click.argument('input_root', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='../output/processed',
              help='Output directory for processed data')
@click.option('--time-window', '-t', default=10.0,
              help='Coincidence time window in ns (default: 10.0)')
@click.option('--energy-min', default=400.0,
              help='Minimum energy threshold in keV (default: 400)')
@click.option('--energy-max', default=600.0,
              help='Maximum energy threshold in keV (default: 600)')
@click.option('--tree-name', default='Hits',
              help='ROOT tree name (default: Hits)')
@click.option('--plot/--no-plot', default=True,
              help='Generate statistics plots')
@click.option('--camera-view/--no-camera-view', default=True,
              help='Generate detector camera view visualization')
@click.option('--camera-bins', default=50,
              help='Histogram bins for camera views (default: 50)')
@click.option('--camera-colormap', default='viridis',
              help='Colormap for camera views (default: viridis)')
@click.option('--camera-log-scale/--camera-linear-scale', default=False,
              help='Use logarithmic color scale for camera views')
@click.option('--verbose/--quiet', '-v/-q', default=True,
              help='Verbose output')
def process_events(input_root, output_dir, time_window, energy_min, energy_max,
                   tree_name, plot, camera_view, camera_bins, camera_colormap,
                   camera_log_scale, verbose):
    """
    Process GATE ROOT files for PET reconstruction.

    Extracts hits, detects coincidences, and creates list-mode data
    suitable for tomographic reconstruction with PyTomography.

    Note: ROOT files store energy in MeV, but this tool works with keV.
          Energy values are automatically converted during loading.

    
    Examples:
        # Basic processing (400-600 keV window, 10 ns coincidence)
        cli_process_events.py output/events.root

        # Custom parameters
        cli_process_events.py -t 5.0 --energy-min 350 --energy-max 650 output/events.root

        # Tight energy window around 511 keV photopeak
        cli_process_events.py --energy-min 480 --energy-max 540 output/events.root

        # High-resolution detector camera views with hot colormap
        cli_process_events.py --camera-bins 100 --camera-colormap hot output/events.root

        # Logarithmic scale for wide dynamic range
        cli_process_events.py --camera-log-scale output/events.root

        # Skip camera views (faster processing)
        cli_process_events.py --no-camera-view output/events.root

        # Specify output location
        cli_process_events.py -o results/processed output/events.root
    """
    click.echo("=" * 70)
    click.echo("PET EVENT PROCESSING")
    click.echo("=" * 70)

    if verbose:
        click.echo(f"\nInput: {input_root}")
        click.echo(f"Output: {output_dir}")
        click.echo(f"Energy window: {energy_min:.1f} - {energy_max:.1f} keV")
        click.echo(f"Time window: {time_window} ns")

    try:
        # Initialize processor
        processor = GATEEventProcessor(input_root, verbose=verbose)

        # Step 1: Load hits (MeV → keV conversion happens here)
        if verbose:
            click.echo("\n[1/4] Loading Event Data")
            click.echo("  (Converting energy from MeV to keV)")
        processor.load_hits(tree_name)

        # Step 2: Detect coincidences
        if verbose:
            click.echo("\n[2/4] Detecting Coincidences")
        processor.detect_coincidences(time_window, energy_min, energy_max)

        # Step 3: Save data
        if verbose:
            click.echo("\n[3/4] Saving Processed Data")
        processor.save_processed_data(output_dir)

        # Step 4: Generate statistics plots
        step_num = 4
        total_steps = 4 + (1 if camera_view else 0)

        if plot:
            if verbose:
                click.echo(f"\n[{step_num}/{total_steps}] Generating Statistics Plots")
            plot_path = Path(output_dir) / "event_statistics.png"
            processor.plot_statistics(plot_path)
            step_num += 1

        # Step 5: Generate detector camera views
        if camera_view:
            if verbose:
                click.echo(f"\n[{step_num}/{total_steps}] Generating Detector Camera Views")

            try:
                # Direct processor access - no intermediate listmode creation
                # This is more efficient than: listmode = processor.create_listmode_data()
                camera_viz = DetectorCameraView(processor, verbose=verbose)
                camera_path = Path(output_dir) / "detector_camera_views.png"
                camera_viz.create_camera_views(
                    camera_path,
                    bins=camera_bins,
                    colormap=camera_colormap,
                    log_scale=camera_log_scale
                )
            except Exception as e:
                if verbose:
                    click.echo(f"  ✗ Could not generate camera views: {e}")

        # Summary
        click.echo("\n" + "=" * 70)
        click.echo("PROCESSING COMPLETE")
        click.echo("=" * 70)
        click.echo(f"\n✓ Output saved to: {Path(output_dir).absolute()}")
        click.echo(f"✓ Total hits: {len(processor.hits_data['energy'])}")
        click.echo(f"✓ Coincidences: {len(processor.coincidences)}")

        output_path = Path(output_dir)
        if (output_path / "listmode_data.npz").exists():
            click.echo(f"\nGenerated files:")
            click.echo(f"  • listmode_data.npz - List-mode event data")
            if plot:
                click.echo(f"  • event_statistics.png - Event statistics")
            if camera_view and (output_path / "detector_camera_views.png").exists():
                click.echo(f"  • detector_camera_views.png - Detector camera views")
            click.echo(f"\nNext step:")
            click.echo(f"  cli_reconstruct_pet.py {output_path / 'listmode_data.npz'}")

    except Exception as e:
        raise click.ClickException(str(e))


if __name__ == "__main__":
    process_events()
