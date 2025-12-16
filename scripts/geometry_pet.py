import opengate as gate
from opengate.contrib.pet import *

import click, rich
import uproot
import math, scipy
import numpy as np
import os
from opengate.sources.base import SourceBase
from opengate.geometry.volumes import VolumeBase
from opengate.actors.digitizers import *
from opengate.geometry.utility import get_grid_repetition, get_circular_repetition
from opengate.sources.utility import get_spectrum
from opengate.actors.coincidences import coincidences_sorter
from scipy.spatial.transform import Rotation
from phantom_derenzo import add_derenzo_phantom, add_sources as derenzo_add_source
import pathlib

# Import detector ID parsing utilities
from utils import (
    parse_volume_ids,
    crystal_id_to_xy,
    add_detector_ids_to_coincidences,
    add_detector_ids_to_reads
)


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
    def __init__(self, sim: gate.Simulation, output_file: str, debug: bool=False):
        self._sim = sim
        # pixelized crystal
        if debug:
            self._nx = 5
            self._ny = 5
        else:
            self._nx = 50
            self._ny = 50
        
        # Number of panels (rsectors) arranged circularly. This needs to consider FOV
        self._n_rsectors = 8
        
        self._housing_size = [
            CRYSTAL_THICKNESS,
            self._nx * CRYSTAL_SIZE_X + (self._nx - 1) * DETECTOR_SPACING,
            self._ny * CRYSTAL_SIZE_Y + (self._ny - 1) * DETECTOR_SPACING,
        ]
        print(f"Calculated housing size: {self._housing_size}")

        # output config
        self._output_filename = output_file

        # References to created objects
        self._phantom = None
        self._sources = []
        
        self._scanner_info = {
            # Crystal configuration
            'crystalTransNr': self._nx,                           # 50 crystals transaxially per panel
            'crystalAxialNr': self._ny,                           # 50 crystals axially per panel
            'crystalTransSpacing': float(CRYSTAL_SIZE_X + DETECTOR_SPACING),  # 2.1 mm
            'crystalAxialSpacing': float(CRYSTAL_SIZE_Y + DETECTOR_SPACING),  # 2.1 mm
            
            # Submodule configuration (no subdivision in your case)
            'submoduleTransNr': 1,
            'submoduleAxialNr': 1,
            'submoduleTransSpacing': 0.0,
            'submoduleAxialSpacing': 0.0,
            
            # Module configuration (no subdivision in your case)
            'moduleTransNr': 1,
            'moduleAxialNr': 1,
            'moduleTransSpacing': 0.0,
            'moduleAxialSpacing': 0.0,
            
            # Rsector configuration (8 panels arranged in octagon)
            'rsectorTransNr': self._n_rsectors,                   # 8 panels
            'rsectorAxialNr': 1,                                  # Single ring
            
            # Scanner geometry
            'radius': float(FOV_RADIUS * 1.4),                    # Distance from center to panel face
            'firstCrystalAxis': 0,                                # First crystal along X axis
            
            # Derived values
            'NrCrystalsPerRing': self._nx * self._n_rsectors,     # 50 * 8 = 400 crystals per ring
            'NrRings': self._ny,                                  # 50 rings (axial crystals)
            
            # Event filtering
            'min_rsector_difference': 0,                          # Accept all coincidences
            
            # Crystal dimensions (for reference, not used by pytomography functions)
            'crystal_length': float(CRYSTAL_THICKNESS),           # 19 mm
            
            # Tell that TOF is there
            'TOF': 1
        }
        
    @property
    def scanner_info(self):
        return self._scanner_info
        
    def _build_detector(self):
        sim = self._sim

        # Housing
        panel  = sim.add_volume("Box", name="Panel")
        panel.mother = sim.world.name
        panel.material = "Aluminium"
        panel.size = self._housing_size
        panel.color = [0.5, 0.5, 0.5, 1]  # grey
        trans, rot = get_circular_repetition(
            self._n_rsectors, [FOV_RADIUS * 1.4, 0, 0], start_angle_deg = 0, axis=[0, 0, 1]
        )
        panel.translation = trans
        panel.rotation = rot
        for i in range(panel.number_of_repetitions):
            print(f"Panel name: {panel.get_repetition_name_from_index(i)}")
            
        
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
        print(f"Crystal configuration: {self._nx} x {self._ny} y, totally repeated for {pixelized_crystals.number_of_repetitions}")
        
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
            'output_file': self._output_filename, 
            'scanner_info': self._scanner_info
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file.with_suffix('.json'), 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Saved detector geometry to: {output_file}")
        return str(output_file)

    def generate_lut(self):
        """
        Post-process the calibration ROOT file to generate a Scanner LUT.
        
        Outputs:
            - output/scanner_lut.npy: Numpy array for PyTomography
            - output/scanner_lut.root: ROOT file with unique crystal coordinates
        """
        import pandas as pd
        import time
        from pathlib import Path
        print(f"\n[LUT GENERATION] Processing {self._output_filename}...")
               
        # 1. Read ROOT file
        with uproot.open(self._output_filename) as f:
            print(f"Opened tree file keys: {list(f.keys())}")
            # We need the 'Reads' tree because it contains the readout (pixelated) positions
            # Note: We need to parse IDs first using your utility
            tree_data = {
                key: f['Reads'][key].array(library="np")
                for key in f['Reads'].keys()
            }
            print(f"Tree data keys: {list(tree_data.keys())}")
            
        # 2. Parse IDs (using your existing utility)
        print("  Parsing volume IDs...")
        tree_data = add_detector_ids_to_reads(tree_data, verbose=False)
        
        # 3. Create DataFrame
        df = pd.DataFrame({
            'rsectorID': tree_data['rsectorID'],
            'moduleID': tree_data['moduleID'],
            'submoduleID': tree_data['submoduleID'],
            'crystalID': tree_data['crystalID'],
            'volumeID': tree_data['TrackVolumeInstanceID'],
            'x': tree_data['PostPosition_X'],
            'y': tree_data['PostPosition_Y'],
            'z': tree_data['PostPosition_Z']
        })
        
        # 4. Drop duplicates to get unique crystals
        # Since we used Readout, all hits in the same crystal should have identical positions
        # (or very close centroids). We take the first occurrence.
        unique_crystals = df.drop_duplicates(subset=['rsectorID', 'moduleID', 'submoduleID', 'crystalID'])
        
        expected_crystals = self._scanner_info['NrCrystalsPerRing'] * self._scanner_info['NrRings']
        print(f"  Found {len(unique_crystals)} unique crystals (Expected: {expected_crystals})")
        
        if len(unique_crystals) != expected_crystals:
            print(f"  [WARNING] Mismatch in crystal count! Simulation time might be too short.")
        
        # 5. Calculate Global ID for sorting
        # Formula must match PyTomography's expectation
        # ID = rsector * (crystals_per_rsector) + ...
        # Based on your config: 
        # rsectorTransNr=8, module=1, submodule=1, crystalTrans=50, crystalAxial=50
        crystals_per_rsector = self._nx * self._ny # 2500
        
        # 注意：這裡的計算邏輯必須與您在 PyTomography 中使用的 gate.py 邏輯完全一致
        # 假設 gate.py 是先排 module, 再排 submodule, 再排 crystal
        unique_crystals['GlobalID'] = (
            unique_crystals['rsectorID'] * crystals_per_rsector + 
            unique_crystals['crystalID'] # 假設 crystalID 已經是 0-2499 的線性索引
        )
        
        # 6. Sort by Global ID
        lut_sorted = unique_crystals.sort_values('GlobalID')
        
        # 7. Save as .npy (for PyTomography)
        lut_numpy = lut_sorted[['x', 'y', 'z']].values.astype(np.float32)
        npy_path = os.path.join(os.path.dirname(self._output_filename), "scanner_lut.npy")
        np.save(npy_path, lut_numpy)
        print(f"  Saved PyTomography LUT: {npy_path}")
        
        # 8. Save as ROOT (as requested)
        root_path = os.path.join(os.path.dirname(self._output_filename), "scanner_lut.root")
        with uproot.recreate(root_path) as f:
            f["LUT"] = {
                "GlobalID": lut_sorted['GlobalID'].values,
                "rsectorID": lut_sorted['rsectorID'].values,
                "crystalID": lut_sorted['crystalID'].values,
                "x": lut_sorted['x'].values,
                "y": lut_sorted['y'].values,
                "z": lut_sorted['z'].values
            }
        print(f"  Saved ROOT LUT: {root_path}")
        
    def add_digitizer(self,calibration_mode: bool = False) -> None:
        """
        Build digitizer chain for PET detector.

        Creates hits collection and readout actors to process detector events.
        Uses volume repetition methods to properly track panel indices.
        """
        sim = self._sim

        # Print panel information for debugging
        print(f"\nDigitizer Setup:")
        print(f"  Number of panels: {self._panel.number_of_repetitions}")
        print(f"  Number of crystals per panel: {self._crystals.number_of_repetitions}")
        print(f"  Panel repetition names:")
        for i in range(self._panel.number_of_repetitions):
            panel_name = self._panel.get_repetition_name_from_index(i)
            print(f"    Panel {i}: {panel_name}")

        # Hits collection actor - attached to crystals with authorization for repeated volumes
        hc = sim.add_actor("DigitizerHitsCollectionActor", "Hits")
        hc.attached_to = self._crystals.name
        hc.authorize_repeated_volumes = True
        hc.output_filename = self._output_filename
        hc.attributes = [
            "PostPosition",              # Crystal impact position
            "TotalEnergyDeposit",        # Energy deposited in crystal
            "PreStepUniqueVolumeID",     # Machine-readable volume encoding
            "GlobalTime",                # Geant4 simulation time
            "EventID",                   # Primary particle ID
            "TrackVolumeName",           # Full volume path (CRITICAL for rsectorID extraction)
            "TrackVertexPosition",       # Source position for tracking
            "TrackVolumeInstanceID"
        ]

        # Readout actor - groups hits by panel, discretizes by crystal
        sc = sim.add_actor("DigitizerReadoutActor", "Reads")
        sc.authorize_repeated_volumes = True
        sc.attached_to = self._crystals.name
        sc.input_digi_collection = hc.name
        sc.group_volume = self._panel.name  # Group by panel repetitions
        sc.discretize_volume = self._crystals.name  # Discretize to crystal level
        sc.policy = "EnergyWeightedCentroidPosition"
        sc.output_filename = hc.output_filename

        # * No need for energy resolution for calibration
        if calibration_mode:
            print("\n[INFO] Digitizer configured in CALIBRATION MODE")
            print("  - Blurring: DISABLED")
            print("  - Energy Window: DISABLED")
            print("  - Output: Raw geometric centers from Readout")            
            return hc, sc

        # * Energy Blurring Actor (Simulation of Energy Resolution)
        blur = sim.add_actor("DigitizerBlurringActor", "Blurring")
        blur.input_digi_collection = sc.name  # Input comes from Readout (sc)
        blur.blur_attribute = "TotalEnergyDeposit"
        blur.blur_method = "Gaussian"
        blur.blur_sigma = 1
        if 'GAGG': # NEED CHANGE:
            blur.blur_resolution = 0.06
        elif 'LYSO':
            blur.blur_resolution = 0.12 
        blur.blur_reference_value = 511 * keV
        blur.output_filename = hc.output_filename
        # -------------------------------------------------------------
        
        # Energy filter - accepts photopeak window
        energy_filter = sim.add_actor("DigitizerEnergyWindowsActor", "EnergyFilter")
        energy_filter.input_digi_collection = hc.name
        energy_filter.channels = [
            {"min": 400 * keV, "max": 650 * keV, "name": "scatter"}  # 511 keV photopeak
        ]
        energy_filter.output_filename = hc.output_filename

        print(f"\nDigitizer chain configured:")
        print(f"  Stage 1: Hits collection (attached to {self._crystals.name})")
        print(f"  Stage 2: Readout (group by {sc.group_volume} → rsectorID, discretize by {sc.discretize_volume} → crystalID)")
        print(f"  Stage 3: Energy window (400-650 keV photopeak)")
        print(f"  Output: {hc.output_filename}")

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

    def add_calibration_source(self):
        """
        Add a large cylindrical source to irradiate all crystals for geometry calibration.
        """
        sim = self._sim
        print("Creating Calibration Source (Cylinder covering FOV)...")
        
        source = sim.add_source("GenericSource", "Calibration_Source")
        source.particle = "gamma" 
        source.energy.type = "mono"
        source.energy.mono = 511 * keV
        source.activity = 150 * MBq # High activity to hit everything quickly
        
        # Geometry: Cylinder slightly larger than FOV to ensure coverage
        source.position.type = "cylinder"
        source.position.radius = 1.2 * mm
        source.position.dz = 5 * cm
        source.direction.type = "iso"
        
        self._sources.append(source)
        return source
    
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
            # Default equal activity for all 6 sectors if not specified
            if activity_Bq_mL is None:
                activity_Bq_mL = [activity] * 6

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
@click.option('--output', type=click.Path(dir_okay=False, writable=True), help="Output file directory.", default="./output/events.root")
@click.option('--scenario', type=click.Choice(['NEMA', 'Derenzo', 'Ring', 'Point', 'Calibration'], case_sensitive=False), 
              help="Simulation scenario to run. Default to 'Ring'.", default="Ring")
@click.option('-n', '--num-thread', type=int, help="Number of threads to use. Default to 1", default=32)
def main(visu, sim_time, output, scenario, num_thread):
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
    pet = PETGeometry(sim, str(output), debug=False)  # Set debug=True for faster 5x5 array
    pet.add_pet()
    pet.add_digitizer(calibration_mode=scenario == 'Calibration')

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
        activity = 1 * MBq 
        source = pet.add_phantom_source(activity=activity / num_thread, isotope="F18")
        print(f"  - Added F-18 source with {activity} activity")

    elif scenario == "Ring":
        # Ring source configuration (no phantom)
        source = pet.add_ring_source(radius=0.5 * cm, activity=150E6 * Bq / num_thread, 
                                     pos=[5 * cm, 7 * cm, -5 * cm], isotope="F18")
        pass
    elif scenario == "Point":
        # Point source for calibration
        print("Creating point source...")
        source = pet.add_point_source(position=[0, 0, 0], activity=1 * MBq / num_thread, isotope="F18")
        print(f"  - Position: center")
        print(f"  - Activity: 1 MBq F-18")
    elif scenario == "Calibration":
        print("Creating Geometry Calibration setup...")
        pet.add_calibration_source()
        # Calibration needs enough time to hit all crystals
        # Override sim_time if it's too short (e.g. default 0.0001 is too short)
        if sim_time < 1.0 and not visu:
            print(f"  [NOTE] Increasing sim-time to 1.0s for calibration coverage")

    # Simulation parameters
    sim.number_of_threads = num_thread
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
        sim.run_timing_intervals = [(0, .000001 * sec)]

    print("\nStarting simulation...")
    print("Close visualization window to end.")
    sim.run()
    sim.close()
    print("Simulation complete!")
    
    # === CALIBRATION POST-PROCESSING ===
    if scenario == "Calibration":
        with uproot.open(pet._output_filename) as f:
            print(f"Reading ROOT file: {pet._output_filename}")

            # Get all tree names in the file
            tree_names = [key for key in f.keys() if f[key].classname.startswith('TTree')]
            print(f"Found {len(tree_names)} tree(s) in file: {tree_names}")
        pet.generate_lut()
        return # Skip coincidence processing for calibration
    # ===================================
    
    # Calculating the coincidences
    print("\nProcessing coincidences...")

    with uproot.open(pet._output_filename) as f:
        print(f"Reading ROOT file: {pet._output_filename}")

        # Get all tree names in the file
        tree_names = [key for key in f.keys() if f[key].classname.startswith('TTree')]
        print(f"Found {len(tree_names)} tree(s) in file: {tree_names}")

        # Read ALL existing trees into memory
        all_trees = {}
        for tree_name in tree_names:
            print(f"  Loading tree '{tree_name}'...")
            tree_name_clean = tree_name.split(';')[0]  # Remove cycle number if present
            all_trees[tree_name_clean] = {
                key: f[tree_name][key].array(library="np")
                for key in f[tree_name].keys()
            }

        # Parse and add rsectorID, moduleID, submoduleID, and crystalID to Reads tree
        print("  Parsing volume IDs to extract rsectorID and crystalID...")
        all_trees['Reads'] = add_detector_ids_to_reads(all_trees['Reads'], verbose=True)

        # Process coincidences from the Reads tree
        print("  Processing coincidences...")
        coincidences = coincidences_sorter(f['Reads'],
                                           3 * gate.g4_units.nanosecond,
                                           "takeAllGoods",
                                           0.5 * gate.g4_units.mm,
                                           "xy",
                                           0.5 * gate.g4_units.mm,
                                           chunk_size=100000,
                                           return_type="dict"
                                           )

        # Add rsectorID, moduleID, submoduleID, and crystalID to coincidences
        # This is needed as coincidences_sorter reads from f['Reads'] (file)
        # which hasn't been modified with the detector IDs yet (those are in all_trees dict)
        coincidences = add_detector_ids_to_coincidences(coincidences, verbose=True)


    print(f"Found {len(coincidences[list(coincidences.keys())[0]])} coincidence events")

    # Create temporary file with all trees + new Coincidences tree
    temp_filename = pet._output_filename.replace('.root', '_temp.root')

    print(f"\nCreating new ROOT file with all trees plus Coincidences...")
    with uproot.recreate(temp_filename) as f:
        # Write all original trees (now includes rsectorID, moduleID, submoduleID, crystalID)
        for tree_name, tree_data in all_trees.items():
            print(f"  Writing tree '{tree_name}' ({len(tree_data)} branches)")
            f[tree_name] = tree_data

        # Write the new Coincidences tree (already includes parsed IDs)
        print(f"  Writing new tree 'Coincidences' ({len(coincidences)} branches)")
        f["Coincidences"] = coincidences

    # Atomically replace the original file with the new one
    print(f"\nReplacing original file...")
    os.replace(temp_filename, pet._output_filename)

    print(f"\n✓ Successfully updated ROOT file: {pet._output_filename}")
    print(f"  Total trees in file: {len(all_trees) + 1}")
    print(f"  - Original trees: {', '.join(all_trees.keys())}")
    print(f"  - New tree: Coincidences")
    print(f"\nReads tree now includes:")
    print(f"  - rsectorID (panel 0-7)")
    print(f"  - moduleID (all = 0)")
    print(f"  - submoduleID (all = 0)")
    print(f"  - crystalID (crystal 0-2499)")
    print(f"\nCoincidence tree contains {len(coincidences)} branches:")
    coincidence_id_branches = [k for k in sorted(coincidences.keys()) if 'rsectorID' in k or 'moduleID' in k or 'submoduleID' in k or 'crystalID' in k]
    if coincidence_id_branches:
        print(f"  ID branches: {', '.join(coincidence_id_branches)}")
    for key in sorted(coincidences.keys())[:10]:  # Show first 10 branches
        print(f"  - {key}")
    if len(coincidences) > 10:
        print(f"  ... and {len(coincidences) - 10} more branches")

    
    # Saving the config
    pet.save_geom()
    
    
if __name__ == "__main__":
    main()