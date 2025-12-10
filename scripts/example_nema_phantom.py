#!/usr/bin/env python3
"""
Example: NEMA IQ Phantom Simulation

This script demonstrates PET imaging with the NEMA NU 2-2012 Image Quality phantom.
The phantom contains 6 fillable spheres and a lung insert for assessing:
- Spatial resolution
- Contrast recovery
- Image uniformity
- Scatter correction accuracy
"""

import sys
import pathlib

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import opengate as gate
from geometry_pet import PETGeometry, add_materials, MBq, cm

def main():
    print("=" * 70)
    print("NEMA IQ Phantom PET Simulation")
    print("=" * 70)

    # Create simulation
    sim = gate.Simulation()

    # Configure world
    add_materials(sim)
    sim.check_volumes_overlap = True
    sim.world.material = "Air"
    sim.world.color = [1, 0, 0, 0]

    # Create PET scanner
    print("\n[1/4] Building PET detector geometry...")
    pet = PETGeometry(sim, debug=False)  # Use debug=True for faster testing
    pet.add_pet()
    print("  ✓ Two-panel detector created (50x50 crystals per panel)")

    # Configure physics
    print("\n[2/4] Setting up physics...")
    pet.setup_physics("QGSP_BERT_EMV")
    print("  ✓ Physics list: QGSP_BERT_EMV")
    print("  ✓ Production cuts: 1 mm")

    # Add NEMA phantom
    print("\n[3/4] Creating NEMA IQ phantom...")
    phantom = pet.add_nema_iq_phantom()
    print(f"  ✓ Body: Cylindrical, 19 cm diameter, 18 cm length")
    print(f"  ✓ Spheres: {len(phantom['spheres'])} fillable spheres")
    print(f"     Diameters: 10, 13, 17, 22, 28, 37 mm")
    print(f"  ✓ Lung insert: 51 mm diameter cylinder (low density)")

    # Add F-18 source
    print("\n[4/4] Adding radioactive source...")
    source = pet.add_phantom_source(activity=10 * MBq, isotope="F18")
    print(f"  ✓ Isotope: F-18 (half-life: 109.77 min)")
    print(f"  ✓ Activity: 10 MBq")
    print(f"  ✓ Distribution: Uniform in phantom spheres")

    # Simulation settings
    print("\n" + "=" * 70)
    print("Simulation Configuration")
    print("=" * 70)
    sim.number_of_threads = 1
    sim.random_seed = 123456
    print(f"  Threads: {sim.number_of_threads}")
    print(f"  Random seed: {sim.random_seed}")

    # Visualization mode
    print("\n  Mode: Visualization (interactive)")
    print("  Note: Close window to end simulation")
    sim.visu = True
    sim.visu_type = "qt"

    # For production runs, use:
    # sim.visu = False
    # sim.run_timing_intervals = [[0, 60]]  # 60 second acquisition
    # sim.number_of_events = 1e7  # 10 million events
    # print(f"  Acquisition time: 60 seconds")
    # print(f"  Number of events: 10 million")
    # print(f"  Output file: output/events.root")

    # Run simulation
    print("\n" + "=" * 70)
    print("Starting simulation...")
    print("=" * 70 + "\n")

    sim.run()

    print("\n" + "=" * 70)
    print("Simulation Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
