#!/usr/bin/env python3
"""
Example: Derenzo (Hot Rod) Phantom Simulation

This script demonstrates PET spatial resolution testing with a Derenzo phantom.
The phantom contains rods of varying diameters arranged in 6 wedge sectors
for evaluating the scanner's ability to resolve closely-spaced objects.
"""

import sys
import pathlib

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import opengate as gate
from geometry_pet import PETGeometry, add_materials, MBq, cm

def main():
    print("=" * 70)
    print("Derenzo (Hot Rod) Phantom PET Simulation")
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
    pet = PETGeometry(sim, debug=False)
    pet.add_pet()
    print("  ✓ Two-panel detector created (50x50 crystals per panel)")

    # Configure physics
    print("\n[2/4] Setting up physics...")
    pet.setup_physics("QGSP_BERT_EMV")
    print("  ✓ Physics list: QGSP_BERT_EMV")

    # Add Derenzo phantom
    print("\n[3/4] Creating Derenzo phantom...")
    rod_pattern = "micro"  # or "clinical"
    phantom = pet.add_derenzo_phantom(rod_pattern=rod_pattern)
    print(f"  ✓ Pattern: {rod_pattern}")
    print(f"  ✓ Total rods: {len(phantom['rods'])}")

    if rod_pattern == "micro":
        print(f"  ✓ Rod diameters: 1.0, 1.5, 2.0, 2.5, 3.0, 4.0 mm")
        print(f"     (6 sectors, rod spacing = 2× diameter)")
    else:
        print(f"  ✓ Rod diameters: 3.5, 4.0, 4.5, 5.0, 5.5, 6.0 mm")
        print(f"     (6 sectors, rod spacing = 2× diameter)")

    print(f"  ✓ Phantom size: 12 cm diameter, 4 cm height")

    # Add F-18 source
    print("\n[4/4] Adding radioactive source...")
    source = pet.add_phantom_source(activity=20 * MBq, isotope="F18")
    print(f"  ✓ Isotope: F-18")
    print(f"  ✓ Activity: 20 MBq")
    print(f"  ✓ Distribution: Hot rods (cold background)")

    # Simulation settings
    print("\n" + "=" * 70)
    print("Simulation Configuration")
    print("=" * 70)
    sim.number_of_threads = 1
    sim.random_seed = 789012
    print(f"  Threads: {sim.number_of_threads}")
    print(f"  Random seed: {sim.random_seed}")

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
    print("\nExpected Results:")
    print("  - Smallest resolvable rod diameter indicates spatial resolution")
    print("  - Sectors with larger rods should be clearly separated")
    print("  - Sectors with smaller rods may merge (resolution limit)")

if __name__ == "__main__":
    main()
