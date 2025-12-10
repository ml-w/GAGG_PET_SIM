#!/usr/bin/env python3
"""
Example: Ring Source Configuration

This script demonstrates a ring source configuration for simulating
rotating/limited-angle PET detector systems. Useful for:
- Two-panel coincidence detectors with rotation
- Limited-angle PET systems
- Calibration and sensitivity measurements
"""

import sys
import pathlib

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import opengate as gate
from geometry_pet import PETGeometry, add_materials, MBq, cm

def main():
    print("=" * 70)
    print("Ring Source PET Simulation")
    print("=" * 70)

    # Create simulation
    sim = gate.Simulation()

    # Configure world
    add_materials(sim)
    sim.check_volumes_overlap = True
    sim.world.material = "Air"
    sim.world.color = [1, 0, 0, 0]

    # Create PET scanner
    print("\n[1/3] Building PET detector geometry...")
    pet = PETGeometry(sim, debug=False)
    pet.add_pet()
    print("  ✓ Two-panel detector created")
    print("  ✓ Panel separation: 10 cm radius from center")

    # Configure physics
    print("\n[2/3] Setting up physics...")
    pet.setup_physics("QGSP_BERT_EMV")
    print("  ✓ Physics configured for PET imaging")

    # Add ring source
    print("\n[3/3] Creating ring source...")
    ring_radius = 8 * cm
    source = pet.add_ring_source(radius=ring_radius, activity=50 * MBq, isotope="F18")
    print(f"  ✓ Ring radius: {ring_radius / cm:.1f} cm")
    print(f"  ✓ Activity: 50 MBq F-18")
    print(f"  ✓ Axial extent: 5 cm")
    print(f"  ✓ Emission: Isotropic")

    # Simulation settings
    print("\n" + "=" * 70)
    print("Simulation Configuration")
    print("=" * 70)
    sim.number_of_threads = 1
    sim.random_seed = 345678
    print(f"  Threads: {sim.number_of_threads}")
    print(f"  Random seed: {sim.random_seed}")

    print("\n  Use Case: Limited-angle PET with rotation")
    print("  Note: For full reconstruction, rotate detector or use")
    print("        multiple acquisition angles")

    # Visualization
    print("\n  Mode: Visualization (interactive)")
    sim.visu = True
    sim.visu_type = "qt"

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
