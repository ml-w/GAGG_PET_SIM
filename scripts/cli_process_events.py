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
@click.option('--verbose/--quiet', '-v/-q', default=True,
              help='Verbose output')
def process_events(input_root, output_dir, time_window, energy_min, energy_max,
                   tree_name, plot, verbose):
    """
    Process GATE ROOT files for PET reconstruction.

    Extracts hits, detects coincidences, and creates list-mode data
    suitable for tomographic reconstruction with PyTomography.

    Note: ROOT files store energy in MeV, but this tool works with keV.
          Energy values are automatically converted during loading.

    \b
    Examples:
        # Basic processing (400-600 keV window, 10 ns coincidence)
        cli_process_events.py output/events.root

        # Custom parameters
        cli_process_events.py -t 5.0 --energy-min 350 --energy-max 650 output/events.root

        # Tight energy window around 511 keV photopeak
        cli_process_events.py --energy-min 480 --energy-max 540 output/events.root

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

        # Step 4: Generate plots
        if plot:
            if verbose:
                click.echo("\n[4/4] Generating Plots")
            plot_path = Path(output_dir) / "event_statistics.png"
            processor.plot_statistics(plot_path)

        # Summary
        click.echo("\n" + "=" * 70)
        click.echo("PROCESSING COMPLETE")
        click.echo("=" * 70)
        click.echo(f"\n✓ Output saved to: {Path(output_dir).absolute()}")
        click.echo(f"✓ Total hits: {len(processor.hits_data['energy'])}")
        click.echo(f"✓ Coincidences: {len(processor.coincidences)}")

        output_path = Path(output_dir)
        if (output_path / "listmode_data.npz").exists():
            click.echo(f"\nNext step:")
            click.echo(f"  cli_reconstruct_pet.py {output_path / 'listmode_data.npz'}")

    except Exception as e:
        raise click.ClickException(str(e))


if __name__ == "__main__":
    process_events()
