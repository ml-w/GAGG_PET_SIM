"""
Derenzo Phantom for PET Spatial Resolution Testing

This module implements a Derenzo (Hot Rod) phantom with 6 sectors containing
different rod diameters for spatial resolution evaluation in PET imaging.

Reference: https://doi.org/10.3390/diagnostics15111387

Default phantom dimensions (scale_factor=1.0):
- Phantom body: 300 × 300 × 50 mm (Box)
- Rod diameters: 14.5, 9.3, 7.85, 6.5, 5.75, 5.0 mm (6 sectors)
- Rod height: 24 mm

Recommended Scale Factors:
------------------------
Clinical PET (human):
  scale_factor = 1.0 (default)
  - Rod sizes: 5.0 - 14.5 mm
  - Suitable for clinical scanners with ~4-5 mm spatial resolution

Small Animal PET (mouse/rat):
  scale_factor = 0.3 - 0.5
  - scale_factor = 0.3: Rod sizes 1.5 - 4.4 mm (micro-PET, <1 mm resolution)
  - scale_factor = 0.4: Rod sizes 2.0 - 5.8 mm (small animal, ~1-2 mm resolution)
  - scale_factor = 0.5: Rod sizes 2.5 - 7.3 mm (high-res small animal)

Preclinical PET (primate/large animal):
  scale_factor = 0.7 - 0.8
  - Rod sizes: 3.5 - 11.6 mm (scale=0.7)
  - Intermediate between clinical and small animal

Example Usage:
--------------
# Small animal PET (mouse)
derenzo = add_derenzo_phantom(sim, name="derenzo_mouse", scale_factor=0.3)

# Clinical PET (human)
derenzo = add_derenzo_phantom(sim, name="derenzo_clinical", scale_factor=1.0)

# Add radioactive sources
activity_Bq_mL = [1e6, 1e6, 1e6, 1e6, 1e6, 1e6]  # Equal activity in all sectors
sources = add_sources(sim, derenzo, activity_Bq_mL)
"""

import opengate as gate
from scipy.spatial.transform import Rotation as R

# Define the units used in the simulation set-up
cm = gate.g4_units.cm
keV = gate.g4_units.keV
mm = gate.g4_units.mm


def add_derenzo_phantom(sim, name="derenzo", scale_factor=1.0):
    """
    Add a Derenzo phantom with 6 sets of cylinders of different size for each set.
    The phantom is described in https://doi.org/10.3390/diagnostics15111387

    Args:
        sim: OpenGATE simulation object
        name: Name for the phantom volume (default "derenzo")
        scale_factor: Scaling factor for all dimensions (default 1.0)
                      Recommended values:
                      - 1.0: Clinical PET (5.0-14.5 mm rods)
                      - 0.5: Small animal PET (2.5-7.3 mm rods)
                      - 0.3: Micro-PET (1.5-4.4 mm rods)

    Returns:
        phantom_body: The main phantom volume object

    Example:
        # Create small animal phantom
        derenzo = add_derenzo_phantom(sim, name="derenzo_mouse", scale_factor=0.4)
    """
    phantom_body = sim.add_volume("Box", name)
    phantom_body.size = [30 * cm * scale_factor, 30 * cm * scale_factor, 5 * cm * scale_factor]
    phantom_body.mother = "world"
    phantom_body.material = "G4_AIR"
    rot = R.from_euler("x", [-90], degrees=True)
    phantom_body.rotation = rot.as_matrix()
    yellow = [1, 1, 0, 0.5]

    tubs_1 = sim.add_volume("Tubs", f"{phantom_body.name}_tubs_1")
    tubs_1.material = "G4_WATER"
    tubs_1.mother = phantom_body.name
    tubs_1.rmin = 0 * mm
    tubs_1.rmax = 14.5 * mm * scale_factor
    tubs_1.dz = 12 * mm * scale_factor
    tubs_1.color = yellow
    tubs_1.translation = [29 * mm, 1.5 * mm, 105 * mm]
    m = R.identity().as_matrix()
    tubs_1.translation = [
        [29 * mm * scale_factor, 105 * mm * scale_factor, 0],
        [-29 * mm * scale_factor, 105 * mm * scale_factor, 0],
        [0 * mm * scale_factor, 54.77 * mm * scale_factor, 0],
    ]

    tubs_2 = sim.add_volume("Tubs", f"{phantom_body.name}_tubs_2")
    tubs_2.material = "G4_WATER"
    tubs_2.mother = phantom_body.name
    tubs_2.rmin = 0 * mm
    tubs_2.rmax = 9.3 * mm * scale_factor
    tubs_2.dz = 12 * mm * scale_factor
    tubs_2.color = yellow
    tubs_2.translation = [29 * mm, 1.5 * mm, 105 * mm]
    m = R.identity().as_matrix()
    tubs_2.translation = [
        [66 * mm * scale_factor, 25 * mm * scale_factor, 0],
        [66 * mm * scale_factor, 89.44 * mm * scale_factor, 0],
        [28.8 * mm * scale_factor, 25 * mm * scale_factor, 0],
        [103.2 * mm * scale_factor, 25 * mm * scale_factor, 0],
        [84.5 * mm * scale_factor, 57.21 * mm * scale_factor, 0],
        [47.5 * mm * scale_factor, 57.21 * mm * scale_factor, 0],
    ]

    tubs_3 = sim.add_volume("Tubs", f"{phantom_body.name}_tubs_3")
    tubs_3.material = "G4_WATER"
    tubs_3.mother = phantom_body.name
    tubs_3.rmin = 0 * mm
    tubs_3.rmax = 7.85 * mm * scale_factor
    tubs_3.dz = 12 * mm * scale_factor
    tubs_3.color = yellow
    tubs_3.translation = [29 * mm, 1.5 * mm, 105 * mm]
    m = R.identity().as_matrix()
    tubs_3.translation = [
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

    tubs_4 = sim.add_volume("Tubs", f"{phantom_body.name}_tubs_4")
    tubs_4.material = "G4_WATER"
    tubs_4.mother = phantom_body.name
    tubs_4.rmin = 0 * mm
    tubs_4.rmax = 6.5 * mm * scale_factor
    tubs_4.dz = 12 * mm * scale_factor
    tubs_4.color = yellow
    tubs_4.translation = [29 * mm, 1.5 * mm, 105 * mm]
    m = R.identity().as_matrix()
    tubs_4.translation = [
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

    tubs_5 = sim.add_volume("Tubs", f"{phantom_body.name}_tubs_5")
    tubs_5.material = "G4_WATER"
    tubs_5.mother = phantom_body.name
    tubs_5.rmin = 0 * mm
    tubs_5.rmax = 5.75 * mm * scale_factor
    tubs_5.dz = 12 * mm * scale_factor
    tubs_5.color = yellow
    tubs_5.translation = [29 * mm, 1.5 * mm, 105 * mm]
    m = R.identity().as_matrix()
    tubs_5.translation = [
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

    tubs_6 = sim.add_volume("Tubs", f"{phantom_body.name}_tubs_6")
    tubs_6.material = "G4_WATER"
    tubs_6.mother = phantom_body.name
    tubs_6.rmin = 0 * mm
    tubs_6.rmax = 5 * mm * scale_factor
    tubs_6.dz = 12 * mm * scale_factor
    tubs_6.color = yellow
    tubs_6.translation = [29 * mm, 1.5 * mm, 105 * mm]
    m = R.identity().as_matrix()
    tubs_6.translation = [
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
    return phantom_body


def add_sources(sim, derenzo_phantom, activity_Bq_mL, particle="e+", energy=511):
    """
    The source is attached to the tubs volumes of the derenzo,
    it means its coordinate system is the same
    activity_Bq_mL should contain the activity concentration for each set of tubs (6)
    """

    sources = []
    for nb_tub in range(1, 7):
        tubs = sim.volume_manager.volumes[f"{derenzo_phantom.name}_tubs_{nb_tub}"]
        for i in range(len(tubs.translation)):
            tub_name = tubs.get_repetition_name_from_index(i)
            s = tubs.solid_info
            source = sim.add_source(
                "GenericSource", f"{derenzo_phantom.name}_tubs_{nb_tub}_source_{i}"
            )
            source.attached_to = tub_name
            source.particle = particle
            source.energy.mono = energy * keV
            source.activity = activity_Bq_mL[nb_tub - 1] * s.cubic_volume
            source.position.type = "cylinder"
            source.position.radius = tubs.rmax
            source.position.dz = tubs.dz
            source.direction.type = "iso"
            sources.append(source)

    return sources