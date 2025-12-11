import opengate as gate
from opengate.contrib.pet import *

import click
import math, scipy
import numpy as np
from opengate.sources.base import SourceBase
from opengate.geometry.volumes import VolumeBase
from opengate.actors.digitizers import *
from opengate.geometry.utility import get_grid_repetition, get_circular_repetition
from opengate.sources.utility import get_spectrum
from scipy.spatial.transform import Rotation
from phantom_derenzo import add_derenzo_phantom, add_sources as derenzo_add_source
import pathlib


# ==============================================================================
# CONFIGURATION
# ==============================================================================

sec = gate.g4_units.second
cm = gate.g4_units.cm
mm = gate.g4_units.mm
deg = gate.g4_units.deg
Bq = gate.g4_units.Bq
MBq = Bq * 1E6
keV = gate.g4_units.keV

# WORLD
WORLD_RADIUS = 20 * cm
WORLD_DEPTH = 15 * cm

# FOV
FOV_RADIUS = 10 * cm

# DETECTOR
CRYSTAL_SIZE_X = 2 * mm
CRYSTAL_SIZE_Y = 2 * mm
CRYSTAL_THICKNESS = 19 * mm
DETECTOR_SPACING = 0.1 * mm



def add_materials(sim):
    # Use local GateMaterials.db if available, otherwise standard
    f = pathlib.Path(__file__).parent.resolve()
    fdb = f / "../GateMaterials.db"
    if fdb.exists():
        if str(fdb) not in sim.volume_manager.material_database.filenames:
            sim.volume_manager.add_material_database(str(fdb))
    else:
        # Fallback or add standard database
        pass

class PETGeometry():
    def __init__(self, sim: gate.Simulation, debug=False):
        self._sim = sim
        # pixelized crystal
        if debug:
            self._nx = 5
            self._ny = 5
        else:
            self._nx = 50
            self._ny = 50
        self._housing_size = [
            CRYSTAL_THICKNESS, 
            self._nx * CRYSTAL_SIZE_X + (self._nx - 1) * DETECTOR_SPACING,
            self._ny * CRYSTAL_SIZE_Y + (self._ny - 1) * DETECTOR_SPACING,
        ]
        print(f"Calculated housing size: {self._housing_size}")

        # output config
        self._output_filename = "output/events.root"

        # References to created objects
        self._phantom = None
        self._sources = []
        
    def _build_detector(self):
        sim = self._sim

        # Housing
        panel  = sim.add_volume("Box", name="CrystalHousing")
        panel.mother = sim.world.name
        panel.material = "Air"
        panel.size = self._housing_size
        panel.color = [0.5, 0.5, 0.5, 1]  # grey
        trans, rot = get_circular_repetition(
            8, [FOV_RADIUS * 1.4, 0, 0], start_angle_deg = 0, axis=[0, 0, 1]
        )
        panel.translation = trans
        panel.rotation = rot
        
        # Build the panel with GAGG
        pixelized_crystals = sim.add_volume("Box", "PixelizedCrystals")
        pixelized_crystals.mother = panel
        pixelized_crystals.size = [CRYSTAL_THICKNESS, CRYSTAL_SIZE_X, CRYSTAL_SIZE_Y]
        trans = get_grid_repetition(
            [1, self._nx, self._ny], [0, CRYSTAL_SIZE_X + DETECTOR_SPACING, CRYSTAL_SIZE_Y + DETECTOR_SPACING]
        )
        pixelized_crystals.material = "LYSO"
        pixelized_crystals.color = [0, 1, 0, 0.3]  # green
        pixelized_crystals.translation = trans
        
        self._panel = panel
        self._crystals = pixelized_crystals
        
    def save_geom(self, output_path: str = "../output/detector_geometry.json") -> None:
        """
        Save detector geometry configuration to JSON file.

        This file can be loaded by visualization and reconstruction tools to
        automatically determine the correct bin sizes and spatial parameters.

        Args:
            output_path: Path to save geometry JSON file
        """
        import json
        from pathlib import Path

        # Calculate detector face dimensions
        detector_width = self._nx * CRYSTAL_SIZE_X + (self._nx - 1) * DETECTOR_SPACING
        detector_height = self._ny * CRYSTAL_SIZE_Y + (self._ny - 1) * DETECTOR_SPACING

        config = {
            'detector_type': 'dual_panel_pet',
            'crystal_array': {
                'nx': self._nx,
                'ny': self._ny,
                'total_crystals': self._nx * self._ny
            },
            'crystal_dimensions': {
                'size_x_mm': float(CRYSTAL_SIZE_X),
                'size_y_mm': float(CRYSTAL_SIZE_Y),
                'thickness_mm': float(CRYSTAL_THICKNESS),
                'spacing_mm': float(DETECTOR_SPACING)
            },
            'detector_face': {
                'width_mm': float(detector_width),
                'height_mm': float(detector_height)
            },
            'housing': {
                'size_mm': [float(s) for s in self._housing_size]
            },
            'field_of_view': {
                'radius_mm': float(FOV_RADIUS)
            },
            'output_file': self._output_filename
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Saved detector geometry to: {output_file}")
        return str(output_file)
        
    def add_digitizer(self) -> None:
        """
        Build digitizer chain for PET detector.

        Creates hits collection and readout actors to process detector events.
        """
        sim = self._sim

        # Hits collection actor
        hc = sim.add_actor("DigitizerHitsCollectionActor", "Hits")
        hc.attached_to = self._crystals.name
        hc.authorize_repeated_volumes = True
        hc.output_filename = self._output_filename
        hc.attributes = [
            "PostPosition",
            "TotalEnergyDeposit",
            "PreStepUniqueVolumeID",
            "GlobalTime",
        ]

        # Readout actor
        sc = sim.add_actor("DigitizerReadoutActor", "Reads")
        sc.authorize_repeated_volumes = True
        sc.attached_to = self._crystals.name
        sc.input_digi_collection = hc.name
        sc.group_volume = self._panel.name
        sc.discretize_volume = self._crystals.name
        sc.policy = "EnergyWeightedCentroidPosition"
        sc.output_filename = hc.output_filename

        # Energy filter
        # 在 add_digitizer 中加入
        energy_filter = sim.add_actor("DigitizerEnergyWindowsActor", "EnergyFilter")
        energy_filter.input_digi_collection = hc.name
        energy_filter.channels = [
            {"min": 400 * keV, "max": 650 * keV, "name": "scatter"}  # 只接受 511 keV 附近
        ]
        energy_filter.output_filename = hc.output_filename

        return hc, sc
        
    def add_pet(self):
        self._build_detector()

    # ==========================================================================
    # PHANTOM METHODS
    # ==========================================================================

    def add_nema_iq_phantom(self, activity_concentration=1.0 * MBq / cm**3):
        """
        Add NEMA NU 2-2012 / IEC 61675-1 Image Quality phantom.

        The phantom contains:
        - Elliptical body filled with water
        - 6 fillable spheres (10, 13, 17, 22, 28, 37 mm diameter)
        - Cylindrical lung insert (51 mm diameter)

        Args:
            activity_concentration: Activity concentration for hot spheres (default 1 MBq/cm³)

        Returns:
            Dictionary with phantom volumes
        """
        sim = self._sim

        # Main phantom body (simplified as cylinder for this implementation)
        # Actual NEMA phantom is elliptical, but cylinder is easier to implement
        phantom_body: VolumeBase = sim.add_volume("Cylinder", "NEMA_Body")
        phantom_body.rmax = 9.5 * cm  # Approximate radius
        phantom_body.dz = 18 * cm  # 180 mm length
        phantom_body.material = "G4_WATER"
        phantom_body.color = [0, 0, 1, 0.1]  # Blue transparent

        # Sphere diameters in mm (inner diameters per NEMA spec)
        sphere_diameters = [10, 13, 17, 22, 28, 37]  # mm
        sphere_volumes = []

        # Arrange spheres in hexagonal pattern
        # Distance from center for hexagonal arrangement
        pattern_radius = 5.5 * cm

        for i, diameter in enumerate(sphere_diameters):
            # Calculate position in hexagonal pattern
            angle = i * 60 * deg  # 6 spheres at 60 degree intervals
            x = pattern_radius * np.cos(angle)
            y = pattern_radius * np.sin(angle)
            z = 0  # All spheres in same plane

            # Create sphere
            sphere = sim.add_volume("Sphere", f"NEMA_Sphere_{diameter}mm")
            sphere.rmax = diameter * mm / 2
            sphere.material = "G4_WATER"
            sphere.mother = phantom_body.name
            sphere.translation = [x, y, z]
            sphere.color = [1, 0, 0, 0.5]  # Red semi-transparent

            sphere_volumes.append(sphere)

        # Add lung insert (low-density cylinder)
        lung_insert = sim.add_volume("Cylinder", "NEMA_Lung_Insert")
        lung_insert.rmax = 2.55 * cm  # 51 mm diameter
        lung_insert.dz = 18 * cm  # Full phantom length
        lung_insert.material = "G4_LUNG_ICRP"
        lung_insert.mother = phantom_body.name
        lung_insert.translation = [0, 0, 0]  # Centered
        lung_insert.color = [0.7, 0.7, 0.7, 0.3]  # Gray transparent

        self._phantom = {
            "type": "NEMA_IQ",
            "body": phantom_body,
            "spheres": sphere_volumes,
            "lung": lung_insert
        }

        return self._phantom

    def add_derenzo_phantom(self):
        """
        Add Derenzo (Hot Rod) phantom for spatial resolution testing.

        Uses the external phantom_derenzo module for accurate geometry.

        Args:
            scale_factor: Scaling factor for phantom dimensions (default 1.0)
                         - 1.0: Clinical PET (5.0-14.5 mm rods)
                         - 0.5: Small animal PET (2.5-7.3 mm rods)
                         - 0.3: Micro-PET (1.5-4.4 mm rods)

        Returns:
            Phantom body volume object
        """
        from phantom_derenzo import add_derenzo_phantom

        sim = self._sim

        # Create phantom using external module
        phantom_body = add_derenzo_phantom(sim, name="derenzo", scale_factor=0.33333)

        # Store phantom reference
        self._phantom = {
            "type": "Derenzo",
            "body": phantom_body,
            "name": phantom_body.name
        }

        return phantom_body

    # ==========================================================================
    # SOURCE METHODS
    # ==========================================================================

    def add_phantom_source(self, activity=10 * MBq, isotope="F18", activity_Bq_mL=None):
        """
        Add radioactive source to the phantom (for NEMA or Derenzo).

        Args:
            activity: Total activity in phantom (default 10 MBq) - used for NEMA phantom
            isotope: Isotope name - "F18", "C11", "Ga68", "Tc99m"
            activity_Bq_mL: List of activity concentrations (Bq/mL) for each Derenzo sector
                           If None, equal activity is assigned to all sectors
                           Example: [1e6, 1e6, 1e6, 1e6, 1e6, 1e6]

        Returns:
            Source object (NEMA) or list of source objects (Derenzo)
        """
        sim = self._sim

        if self._phantom is None:
            raise ValueError("No phantom created. Call add_nema_iq_phantom() or add_derenzo_phantom() first.")

        # Handle Derenzo phantom separately using external module
        sources = []
        if self._phantom['type'] == "Derenzo":
            from phantom_derenzo import add_sources

            # Default equal activity for all 6 sectors if not specified
            if activity_Bq_mL is None:
                activity_Bq_mL = [1e6 * Bq, 1e6 * Bq, 1e6 * Bq,  1e6 * Bq, 1e6 * Bq, 1e6 * Bq]

            # Use external add_sources function
            phantom_body = self._phantom['body']
            sources.extend(derenzo_add_source(sim, phantom_body, activity_Bq_mL))
        else:
            # NEMA phantom - original implementation
            source = sim.add_source("GenericSource", f"{self._phantom['type']}_Source")
            for source in sources:
                # Set isotope-specific properties
                if isotope == "F18":
                    source.energy.type = "F18"
                    source.half_life = 109.77 * 60  # seconds (109.77 min)
                elif isotope == "C11":
                    source.energy.type = "C11"
                    source.half_life = 20.38 * 60  # seconds
                elif isotope == "Ga68":
                    source.energy.type = "Ga68"
                    source.half_life = 67.71 * 60  # seconds
                else:
                    # Default to mono-energetic positron
                    source.energy.type = "mono"
                    source.energy.mono = 511 * keV
                    
                source.particle = "e+"  # Positron
                source.activity = activity
                self._sources.append(source)
            return source

    def add_ring_source(self, 
                        pos: list[float] = [0., 0., 0.], 
                        ori_vec: list[float] = [1., 1., 1.], 
                        radius: float = 8 * cm, activity: float = 10 * MBq, isotope: str = "F18") -> SourceBase:
        """
        Add ring source configuration for rotating detector simulations.

        Args:
            pos: Position of the ring center [x, y, z] in world coordinates (default [0, 0, 0])
            ori_vec: Orientation vector for ring normal direction (default [1, 1, 1]), normalized automatically
            radius: Radius of the ring source (default 8 cm)
            activity: Total activity distributed in ring (default 10 MBq)
            isotope: Isotope name - "F18", "C11", "Ga68", etc.

        Returns:
            Source object
        """
        sim = self._sim

        # Fill FOV with water to promote enihlation events
        FOV = sim.add_volume("Tubs", "FOV")
        FOV.mother = sim.world.name
        FOV.material = "Water"
        FOV.rmax = FOV_RADIUS
        FOV.rmin = 0
        FOV.dz = WORLD_DEPTH * 0.5
        FOV.color = [0.8, 1, 0.2, 0.1]

        # Create ring source using tube geometry
        source = sim.add_source("GenericSource", "Ring_Source")
        source.particle = "e+"  # Positron

        # Set isotope-specific properties
        if isotope == "F18":
            source.energy.type = "F18"
            source.half_life = 109.77 * 60 * sec # seconds
        elif isotope == "C11":
            source.energy.type = "C11"
            source.half_life = 20.38 * 60 * sec # seconds
        elif isotope == "Ga68":
            source.energy.type = "Ga68"
            source.half_life = 67.71 * 60 * sec # seconds
        else:
            source.energy.type = "mono"
            source.energy.mono = 511 * keV

        source.activity = activity

        # Ring geometry - thin hollow cylinder
        source.position.type = "cylinder"
        source.position.radius = radius
        source.position.dz = 1 * cm  # Axial extent of ring
        source.position.translation = pos
        
        # calculate rot
        init_vec = [0, 0, 1]
        rot, rss = Rotation.align_vectors([ori_vec], [init_vec])
        source.position.rotation = rot.as_matrix()
        
        # visualization
        source_vis = sim.add_volume("Tubs", "Ring Source")
        source_vis.mother = "FOV"
        source_vis.rmax = radius
        source_vis.rmin = 0
        source_vis.dz = 1 * mm
        source_vis.color = [1, 0, 0, 1]
        source_vis.translation = pos
        source_vis.rotation = rot.as_matrix()

        # Isotropic emission
        source.direction.type = "iso"

        self._sources.append(source)
        return source

    def add_point_source(self, position=[0, 0, 0], activity=1 * MBq, isotope="F18"):
        """
        Add point source for calibration or testing.

        Args:
            position: [x, y, z] position in world coordinates
            activity: Source activity (default 1 MBq)
            isotope: Isotope name

        Returns:
            Source object
        """
        sim = self._sim

        source = sim.add_source("GenericSource", "Point_Source")
        source.particle = "e+"

        if isotope == "F18":
            source.energy.type = "F18"
            source.half_life = 109.77 * 60
        elif isotope == "C11":
            source.energy.type = "C11"
            source.half_life = 20.38 * 60
        elif isotope == "Ga68":
            source.energy.type = "Ga68"
            source.half_life = 67.71 * 60
        else:
            source.energy.type = "mono"
            source.energy.mono = 511 * keV

        source.activity = activity
        source.position.type = "point"
        source.position.translation = position
        source.direction.type = "iso"

        self._sources.append(source)
        return source

    # ==========================================================================
    # PHYSICS CONFIGURATION
    # ==========================================================================

    def setup_physics(self, physics_list="QGSP_BERT_EMV"):
        """
        Configure physics for PET simulation.

        Args:
            physics_list: Geant4 physics list name (default: QGSP_BERT_EMV)
        """
        sim = self._sim

        # Set physics list
        sim.physics_manager.physics_list_name = physics_list

        # Production cuts for different regions
        sim.physics_manager.set_production_cut("world", "all", 10 * mm)
        sim.physics_manager.set_production_cut(
            self._crystals.name, "all", 1 * mm  # 只在晶體處用細緻的 cut
        )

        # Enable specific processes if needed
        sim.physics_manager.enable_decay = True

        return sim.physics_manager
        

@click.command()
@click.option('--visu', is_flag=True, help="Turn on visualization. This turns off multi-thread and reduce sim time.", 
              default=False)
@click.option('--sim-time', type=float, help="Seconds to simulate. Default to 0.0001 sec", 
              default=0.0001)
@click.option('--output', type=str, help="Output file directory.", default="./output")
@click.option('--scenario', type=click.Choice(['NEMA', 'Derenzo', 'Ring', 'Point'], case_sensitive=False), 
              help="Simulation scenario to run. Default to 'Ring'.", default="Ring")
def main(visu, sim_time, output, scenario):
    """
    Example usage scenarios for PET simulation.

    Uncomment the desired scenario to run:
    1. NEMA IQ Phantom with F-18 source
    2. Derenzo Phantom for spatial resolution testing
    3. Ring source configuration (for rotating detector simulation)
    4. Point source for calibration+
    """

    # ==========================================================================
    # SCENARIO 1: NEMA IQ Phantom (Image Quality Assessment)
    # ==========================================================================
    sim = gate.Simulation()

    # Basic simulation configuration
    add_materials(sim)
    sim.check_volumes_overlap = True
    sim.world.size = [WORLD_RADIUS * 2, WORLD_RADIUS * 2, WORLD_DEPTH]
    sim.world.material = "Air"
    sim.world.color = [1, 0, 1, 1]  # invisible

    # Create PET geometry
    pet = PETGeometry(sim, debug=False)  # Set debug=True for faster 5x5 array
    pet.add_pet()
    pet.add_digitizer()
    pet._output_filename = output

    # Save geometry configuration for visualization tools
    pet.save_geom()

    # Configure physics
    pet.setup_physics("G4EmStandardPhysics_option4")

    if scenario == "NEMA":
        # Add NEMA IQ phantom
        print("Creating NEMA IQ phantom...")
        phantom = pet.add_nema_iq_phantom()
        print(f"  - Created {len(phantom['spheres'])} spheres")
        print(f"  - Sphere diameters: 10, 13, 17, 22, 28, 37 mm")

        # Add F-18 source to phantom
        source = pet.add_phantom_source(activity=10 * MBq, isotope="F18")
        print(f"  - Added F-18 source with 10 MBq activity")

    elif scenario == "Derenzo":
        # Add Derenzo phantom
        print("Creating Derenzo (Hot Rod) phantom...")
        phantom = pet.add_derenzo_phantom()  # or "clinical"
        phantom.rotation=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        print(f"  - Created Derenzo phantom")

        # Add source to Derenzo phantom
        source = pet.add_phantom_source(activity=20 * MBq, isotope="F18")
        print(f"  - Added F-18 source with 20 MBq activity")

    elif scenario == "Ring":
        # Ring source configuration (no phantom)
        source = pet.add_ring_source(radius=0.5 * cm, activity=150E6 * Bq, 
                                     pos=[5 * cm, 7 * cm, -5 * cm], isotope="F18")
        pass
    elif scenario == "Point":
        # Point source for calibration
        print("Creating point source...")
        source = pet.add_point_source(position=[0, 0, 0], activity=1 * MBq, isotope="F18")
        print(f"  - Position: center")
        print(f"  - Activity: 1 MBq F-18")

    # Simulation parameters
    sim.number_of_threads = 32
    sim.random_seed = 123456

    # Visualization (disable for production runs)
    sim.visu = visu

    # Run parameters (for actual simulation, not just visualization)
    # Uncomment and adjust for production:
    sim.run_timing_intervals = [(0, sim_time * sec)]  # 1 second acquisition
    # sim.number_of_events = 1e4  # Number of primary particles

    if sim.visu:
        # Program will run into a deadlock if multi-threaded.
        sim.number_of_threads = 1
        # sim.run_timing_intervals = [(0, .000001 * sec)]
        sim.run_timing_intervals = [(0, .00000001 * sec)]

    print("\nStarting simulation...")
    print("Close visualization window to end.")
    sim.run()
    print("Simulation complete!")
    
    
if __name__ == "__main__":
    main()