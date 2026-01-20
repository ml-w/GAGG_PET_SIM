"""
Derenzo Phantom for PET Spatial Resolution Testing - Explicit Volume Version

This is a modified version of phantom_derenzo.py that creates each hot rod as an
individual volume instead of using translation lists. This allows the phantom to
be voxelized properly with OpenGATE's sim.voxelize_geometry().

Key differences from original:
- Each cylinder is created as a separate volume (e.g., tubs_1_rod_0, tubs_1_rod_1, etc.)
- No translation lists - each volume has a single translation
- Fully compatible with OpenGATE voxelization

Reference: https://doi.org/10.3390/diagnostics15111387
"""

import opengate as gate
from scipy.spatial.transform import Rotation as R

# Define the units used in the simulation set-up
cm = gate.g4_units.cm
keV = gate.g4_units.keV
mm = gate.g4_units.mm


def add_derenzo_phantom(sim, name="derenzo", scale_factor=1.0):
    """
    Add a Derenzo phantom with 6 sets of cylinders of different sizes.
    Each cylinder is created as an individual volume for voxelization compatibility.

    Phantom housing is a cylindrical Tubs (tube) geometry filled with PMMA.

    Aware that using translation list vectors also works for the simulation, but it prevents
    the phantom to be saved. That's why we are using for loop to create this phantom.

    Args:
        sim: OpenGATE simulation object
        name: Name for the phantom volume (default "derenzo")
        scale_factor: Scaling factor for all dimensions (default 1.0)

    Returns:
        phantom_body: The main phantom volume object
    """
    # Configs
    PHANTOM_DIAMETER = 28 * cm  # Outer diameter of cylindrical phantom
    PHANTOM_THICKNESS = 8 * cm  # Axial extent
    ROD_LENGTH = (PHANTOM_THICKNESS - 2 * cm)/2.

    # FOV
    FOV = sim.add_volume("Box", "FOV")
    FOV.size = [30 * cm * scale_factor, 30 * cm * scale_factor, PHANTOM_THICKNESS * scale_factor]
    FOV.mother = "world"
    FOV.material = "G4_AIR"
    FOV.color = [0, 1, 0, 0.1]  # Semi-transparent green
    
    # Create phantom body (PMMA-filled cylindrical tube)
    phantom_body = sim.add_volume("Tubs", name)
    phantom_body.rmin = 0 * mm  # Solid cylinder (not hollow)
    phantom_body.rmax = PHANTOM_DIAMETER / 2 * scale_factor
    phantom_body.dz = PHANTOM_THICKNESS / 2 * scale_factor  # Half-length in GATE
    phantom_body.mother = FOV.name
    phantom_body.material = "PMMA"

    # Rotate to align with typical PET scanner orientation
    # Phantom axis along Z, rods oriented axially
    rot = R.from_euler("x", [-90], degrees=True)
    phantom_body.rotation = rot.as_matrix()
    yellow = [1, 1, 0, 0.5]


    # Sector 1: 14.5 mm diameter rods (3 rods)
    sector_1_positions = [
        [29 * mm * scale_factor, 105 * mm * scale_factor, 0],
        [-29 * mm * scale_factor, 105 * mm * scale_factor, 0],
        [0 * mm * scale_factor, 54.77 * mm * scale_factor, 0],
    ]

    for i, pos in enumerate(sector_1_positions):
        rod = sim.add_volume("Tubs", f"{phantom_body.name}_tubs_1_rod_{i}")
        rod.material = "G4_WATER"
        rod.mother = phantom_body.name
        rod.rmin = 0 * mm
        rod.rmax = 14.5 * mm * scale_factor
        rod.dz = ROD_LENGTH * scale_factor
        rod.color = yellow
        rod.translation = pos

    # Sector 2: 9.3 mm diameter rods (6 rods)
    sector_2_positions = [
        [66 * mm * scale_factor, 25 * mm * scale_factor, 0],
        [66 * mm * scale_factor, 89.44 * mm * scale_factor, 0],
        [28.8 * mm * scale_factor, 25 * mm * scale_factor, 0],
        [103.2 * mm * scale_factor, 25 * mm * scale_factor, 0],
        [84.5 * mm * scale_factor, 57.21 * mm * scale_factor, 0],
        [47.5 * mm * scale_factor, 57.21 * mm * scale_factor, 0],
    ]

    for i, pos in enumerate(sector_2_positions):
        rod = sim.add_volume("Tubs", f"{phantom_body.name}_tubs_2_rod_{i}")
        rod.material = "G4_WATER"
        rod.mother = phantom_body.name
        rod.rmin = 0 * mm
        rod.rmax = 9.3 * mm * scale_factor
        rod.dz = ROD_LENGTH * scale_factor
        rod.color = yellow
        rod.translation = pos

    # Sector 3: 7.85 mm diameter rods (10 rods)
    sector_3_positions = [
        [66 * mm * scale_factor, -45.19 * mm * scale_factor, 0],
        [97.4 * mm * scale_factor, -45.19 * mm * scale_factor, 0],
        [34.6 * mm * scale_factor, -45.19 * mm * scale_factor, 0],
        [18.9 * mm * scale_factor, -18 * mm * scale_factor, 0],
        [50.3 * mm * scale_factor, -18 * mm * scale_factor, 0],
        [81.7 * mm * scale_factor, -18 * mm * scale_factor, 0],
        [113.1 * mm * scale_factor, -18 * mm * scale_factor, 0],
        [50.3 * mm * scale_factor, -72.38 * mm * scale_factor, 0],
        [81.7 * mm * scale_factor, -72.38 * mm * scale_factor, 0],
        [66 * mm * scale_factor, -99.57 * mm * scale_factor, 0],
    ]

    for i, pos in enumerate(sector_3_positions):
        rod = sim.add_volume("Tubs", f"{phantom_body.name}_tubs_3_rod_{i}")
        rod.material = "G4_WATER"
        rod.mother = phantom_body.name
        rod.rmin = 0 * mm
        rod.rmax = 7.85 * mm * scale_factor
        rod.dz = ROD_LENGTH * scale_factor
        rod.color = yellow
        rod.translation = pos

    # Sector 4: 6.5 mm diameter rods (10 rods)
    sector_4_positions = [
        [0 * mm * scale_factor, -91.69 * mm * scale_factor, 0],
        [0 * mm * scale_factor, -46.67 * mm * scale_factor, 0],
        [26 * mm * scale_factor, -91.69 * mm * scale_factor, 0],
        [-26 * mm * scale_factor, -91.69 * mm * scale_factor, 0],
        [-13 * mm * scale_factor, -69.18 * mm * scale_factor, 0],
        [13 * mm * scale_factor, -69.18 * mm * scale_factor, 0],
        [13 * mm * scale_factor, -114.2 * mm * scale_factor, 0],
        [-13 * mm * scale_factor, -114.2 * mm * scale_factor, 0],
        [-39 * mm * scale_factor, -114.2 * mm * scale_factor, 0],
        [39 * mm * scale_factor, -114.2 * mm * scale_factor, 0],
    ]

    for i, pos in enumerate(sector_4_positions):
        rod = sim.add_volume("Tubs", f"{phantom_body.name}_tubs_4_rod_{i}")
        rod.material = "G4_WATER"
        rod.mother = phantom_body.name
        rod.rmin = 0 * mm
        rod.rmax = 6.5 * mm * scale_factor
        rod.dz = ROD_LENGTH * scale_factor
        rod.color = yellow
        rod.translation = pos

    # Sector 5: 5.75 mm diameter rods (10 rods)
    sector_5_positions = [
        [-66 * mm * scale_factor, -42.92 * mm * scale_factor, 0],
        [-89 * mm * scale_factor, -42.92 * mm * scale_factor, 0],
        [-43 * mm * scale_factor, -42.92 * mm * scale_factor, 0],
        [-66 * mm * scale_factor, -82.76 * mm * scale_factor, 0],
        [-54.5 * mm * scale_factor, -62.84 * mm * scale_factor, 0],
        [-77.5 * mm * scale_factor, -62.84 * mm * scale_factor, 0],
        [-54.5 * mm * scale_factor, -23 * mm * scale_factor, 0],
        [-77.5 * mm * scale_factor, -23 * mm * scale_factor, 0],
        [-31.5 * mm * scale_factor, -23 * mm * scale_factor, 0],
        [-100.5 * mm * scale_factor, -23 * mm * scale_factor, 0],
    ]

    for i, pos in enumerate(sector_5_positions):
        rod = sim.add_volume("Tubs", f"{phantom_body.name}_tubs_5_rod_{i}")
        rod.material = "G4_WATER"
        rod.mother = phantom_body.name
        rod.rmin = 0 * mm
        rod.rmax = 5.75 * mm * scale_factor
        rod.dz = ROD_LENGTH * scale_factor
        rod.color = yellow
        rod.translation = pos

    # Sector 6: 5.0 mm diameter rods (15 rods)
    sector_6_positions = [
        [-66 * mm * scale_factor, 17 * mm * scale_factor, 0],
        [-46 * mm * scale_factor, 17 * mm * scale_factor, 0],
        [-26 * mm * scale_factor, 17 * mm * scale_factor, 0],
        [-86 * mm * scale_factor, 17 * mm * scale_factor, 0],
        [-106 * mm * scale_factor, 17 * mm * scale_factor, 0],
        [-56 * mm * scale_factor, 34.32 * mm * scale_factor, 0],
        [-36 * mm * scale_factor, 34.32 * mm * scale_factor, 0],
        [-76 * mm * scale_factor, 34.32 * mm * scale_factor, 0],
        [-96 * mm * scale_factor, 34.32 * mm * scale_factor, 0],
        [-66 * mm * scale_factor, 51.64 * mm * scale_factor, 0],
        [-46 * mm * scale_factor, 51.64 * mm * scale_factor, 0],
        [-86 * mm * scale_factor, 51.64 * mm * scale_factor, 0],
        [-76 * mm * scale_factor, 68.96 * mm * scale_factor, 0],
        [-56 * mm * scale_factor, 68.96 * mm * scale_factor, 0],
        [-66 * mm * scale_factor, 86.28 * mm * scale_factor, 0],
    ]

    for i, pos in enumerate(sector_6_positions):
        rod = sim.add_volume("Tubs", f"{phantom_body.name}_tubs_6_rod_{i}")
        rod.material = "G4_WATER"
        rod.mother = phantom_body.name
        rod.rmin = 0 * mm
        rod.rmax = 5.0 * mm * scale_factor
        rod.dz = ROD_LENGTH * scale_factor
        rod.color = yellow
        rod.translation = pos

    return phantom_body


def add_sources(sim, derenzo_phantom, activity_Bq_mL, particle="e+", energy=511):
    """
    Add radioactive sources to the Derenzo phantom rods.

    This version works with explicitly created volumes (no repetitions).

    Args:
        sim: OpenGATE simulation object
        derenzo_phantom: Phantom body volume returned by add_derenzo_phantom()
        activity_Bq_mL: List of 6 activity concentrations (Bq/mL) for each sector
        particle: Particle type (default "e+")
        energy: Particle energy in keV (default 511)

    Returns:
        List of source objects
    """
    sources = []

    # Number of rods in each sector
    rods_per_sector = [3, 6, 10, 10, 10, 15]

    for sector_idx in range(6):
        sector_num = sector_idx + 1
        num_rods = rods_per_sector[sector_idx]

        for rod_idx in range(num_rods):
            # Get the rod volume
            rod_name = f"{derenzo_phantom.name}_tubs_{sector_num}_rod_{rod_idx}"
            rod = sim.volume_manager.volumes[rod_name]

            # Create source for this rod
            source = sim.add_source("GenericSource", f"{rod_name}_source")
            source.attached_to = rod_name
            source.particle = particle
            source.energy.mono = energy * keV

            # Calculate activity based on rod volume
            source.activity = activity_Bq_mL[sector_idx] * rod.solid_info.cubic_volume

            # Configure source geometry
            source.position.type = "cylinder"
            source.position.radius = rod.rmax
            source.position.dz = rod.dz
            source.direction.type = "iso"

            sources.append(source)

    return sources
