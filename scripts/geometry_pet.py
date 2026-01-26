from logging.handlers import WatchedFileHandler
from turtle import down
import opengate as gate
from opengate.contrib.pet import *

import click, rich
import uproot
import math, scipy
import numpy as np
import os, re, ast
from opengate.sources.base import SourceBase
from opengate.geometry.volumes import VolumeBase
from opengate.voxelize import voxelize_geometry, write_itk_image, write_voxelized_geometry
from opengate.actors.digitizers import *
from opengate.geometry.utility import get_grid_repetition, get_circular_repetition
from opengate.sources.utility import get_spectrum
from opengate.actors.coincidences import coincidences_sorter
from scipy.spatial.transform import Rotation
from phantom_derenzo import add_derenzo_phantom, add_sources as derenzo_add_source
import pathlib
from pathlib import Path

# Import detector ID parsing utilizties
from utils import (
    parse_volume_ids,
    crystal_id_to_xy,
    add_detector_ids_to_coincidences,
    add_detector_ids_to_reads,
    print_uproot_tree
)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

sec = gate.g4_units.second
ms = gate.g4_units.ms
us = gate.g4_units.us
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
CRYSTAL_THICKNESS = 20 * mm
DETECTOR_SPACING = 0.1 * mm

# COLIMATOR, Add by Xiaoyu
COLIMATOR_THICKNESS = 9.0 * mm  # THICKNESS of the COLIMATOR
COLIMATOR_SPACING = 0.1 * mm    # SPACE between crystal and COLIMATOR
COLIMATOR_MATERIAL = "G4_Pb"    # MATERIAL of the COLIMATOR

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
    def __init__(self, sim: gate.Simulation, output_file: str, debug: bool=False, gen_attenuation_img: bool=False, crystal="GAGG"):
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

        # start angle of the setup, Add by Xiaoyu
        self._start_angle = 40
        
        self._housing_size = [
            CRYSTAL_THICKNESS + 2 * (COLIMATOR_THICKNESS + COLIMATOR_SPACING), # to include COLIMATOR, Add by Xiaoyu
            self._nx * CRYSTAL_SIZE_X + (self._nx - 1) * DETECTOR_SPACING,
            self._ny * CRYSTAL_SIZE_Y + (self._ny - 1) * DETECTOR_SPACING,
        ]
        print(f"Calculated housing size: {self._housing_size}")

        # output config
        self._output_filename = output_file

        # References to created objects
        self._phantom = None
        self._sources = []
        self._crystal = crystal
        
        self._gen_attenuation_img = gen_attenuation_img
        
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
        panel.material = "G4_AIR"
        panel.size = self._housing_size
        panel.color = [0.5, 0.5, 0.5, 1]  # grey
        trans, rot = get_circular_repetition(
            self._n_rsectors, [FOV_RADIUS * 1.46, 0, 0], start_angle_deg = 0, axis=[0, 0, 1]
        )
        panel.translation = trans
        panel.rotation = rot
        for i in range(panel.number_of_repetitions):
            print(f"Panel name: {panel.get_repetition_name_from_index(i)}")
            
        
        # Build the panel with GAGG
        pixelized_crystals = sim.add_volume("Box", "PixelizedCrystals")
        pixelized_crystals.mother = panel.name
        pixelized_crystals.size = [CRYSTAL_THICKNESS, CRYSTAL_SIZE_X, CRYSTAL_SIZE_Y]
        trans = get_grid_repetition(
            [1, self._nx, self._ny], [0, CRYSTAL_SIZE_X + DETECTOR_SPACING, CRYSTAL_SIZE_Y + DETECTOR_SPACING]
        )
        pixelized_crystals.material = "GAGG"
        pixelized_crystals.color = [0, 1, 0, 0.3]  # green
        pixelized_crystals.translation = trans
        print(f"Crystal configuration: {self._nx} x {self._ny} y, totally repeated for {pixelized_crystals.number_of_repetitions}")
        print(f"Crystal: {self._crystal}")

        # Build lead shield with holds (colimator), Add by Xiaoyu
        lead_colimator = sim.add_volume("Box", "LeadColimator")
        lead_colimator.mother = panel.name
        lead_colimator.size = [COLIMATOR_THICKNESS, self._housing_size[1], self._housing_size[2]]
        lead_colimator.material = COLIMATOR_MATERIAL
        lead_colimator.translation = [- CRYSTAL_THICKNESS/2 - COLIMATOR_THICKNESS/2 - COLIMATOR_SPACING, 0, 0]
        lead_colimator.color = [1.0, 0, 0, 1.0]
        # Build lead shield holds
        filtered_trans = [
            [0, pos[1], pos[2]] 
            for pos in trans
            if abs(pos[1]) <= self._housing_size[1]/2 and abs(pos[2]) <= self._housing_size[2]/2
        ]
        for i, (x, y, z) in enumerate(filtered_trans):
            lead_hole = sim.add_volume("Tubs", f"LeadHole_{i:04d}")
            lead_hole.mother = lead_colimator.name
            lead_hole.material = "G4_AIR"
            lead_hole.rmax = (CRYSTAL_SIZE_X + CRYSTAL_SIZE_Y)/4      
            lead_hole.rmin = 0                
            lead_hole.dz = COLIMATOR_THICKNESS/2 
            lead_hole.sphi = 0                
            lead_hole.dphi = 360 * deg
            lead_hole.rotation = np.array([[0,0,-1],[0,1,0],[1,0,0]])
            lead_hole.translation = [x, y, z]
            lead_hole.color = [0, 0, 1, 0.5]
        
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

        # Following gate.py logic in pytomography        
        unique_crystals['GlobalID'] = (
            unique_crystals['rsectorID'] * crystals_per_rsector + 
            unique_crystals['crystalID'] 
        )
        
        # 6. Sort by Global ID
        lut_sorted = unique_crystals.sort_values('GlobalID')
        
        # 7. Save as .npy (for PyTomography)
        lut_numpy = lut_sorted[['x', 'y', 'z']].values.astype(np.float32)
        npy_path =  str(Path(self._output_filename).parent / "scanner_lut.npy")
        np.save(npy_path, lut_numpy)
        print(f"  Saved PyTomography LUT: {npy_path}")
        
        # 8. Save as ROOT (as requested)
        root_path = str(Path(self._output_filename).parent / "scanner_lut.root")
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

        # Handing filenames
        p = Path(self._output_filename)

        # STAGE 1: Hits collection actor - collect energy deposits in crystals
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
        ]

        # STAGE 2: Readout actor - group hits by panel, discretize by crystal
        sc = sim.add_actor("DigitizerReadoutActor", "Reads")
        sc.attached_to = self._crystals.name
        sc.authorize_repeated_volumes = True
        sc.input_digi_collection = hc.name  # Input from Hits
        sc.group_volume = self._panel.name  # Group by panel repetitions
        sc.discretize_volume = self._crystals.name  # Discretize to crystal level
        sc.policy = "EnergyWeightedCentroidPosition"
        sc.output_filename = self._output_filename

        # For calibration mode, skip blurring and energy windowing
        if calibration_mode:
            print("\n[INFO] Digitizer configured in CALIBRATION MODE")
            print("  - Chain: Hits → Readout")
            print("  - Blurring: DISABLED")
            print("  - Energy Window: DISABLED")
            print("  - Output: Raw geometric centers from Readout")
            print(f"  Output file: {sc.output_filename}")
            return hc, sc

        # STAGE 3: Energy Blurring Actor - simulate energy resolution
        blur = sim.add_actor("DigitizerBlurringActor", "Blurred")
        blur.authorize_repeated_volumes = True
        blur.attached_to = self._crystals.name
        blur.input_digi_collection = sc.name  # Input from Readout
        blur.blur_attribute = "TotalEnergyDeposit"
        blur.blur_method = "Gaussian"
        if self._crystal == 'GAGG':  
            blur.blur_fwhm = 0.06  # 6% at 511 keV
        elif self._crystal == 'LYSO':
            blur.blur_fwhm = 0.12  # 12% at 511 keV
        blur.blur_reference_value = 511 * keV
        blur.output_filename = hc.output_filename

        # STAGE 4: Energy window filter - accept 480-550 keV photopeak
        energy_filter = sim.add_actor("DigitizerEnergyWindowsActor", "EnergyFilter")
        energy_filter.attached_to = self._crystals.name
        energy_filter.authorize_repeated_volumes = True
        energy_filter.input_digi_collection = blur.name  # Input from Blurred
        energy_filter.channels = [
            {"min": 480 * keV, "max": 540 * keV, "name": "photopeak"}  # 511 keV ± 15 keV
        ]
        energy_filter.output_filename = hc.output_filename

        print(f"\nDigitizer chain configured:")
        print(f"  Stage 1: Hits collection (attached to {self._crystals.name})")
        print(f"  Stage 2: Readout (group by {sc.group_volume}, discretize by {sc.discretize_volume})")
        print(f"  Stage 3: Energy blurring (resolution: {blur.blur_resolution:.1%} at {blur.blur_reference_value/keV:.0f} keV)")
        print(f"  Stage 4: Energy window (480-550 keV photopeak)")
        print(f"  Output: {energy_filter.output_filename}")

        return hc, energy_filter  # Return hits and final output actor
        
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
        phantom_body: VolumeBase = sim.add_volume("TubsVolume", "NEMA_Body")
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
            sphere = sim.add_volume("SphereVolume", f"NEMA_Sphere_{diameter}mm")
            sphere.rmax = diameter * mm / 2
            sphere.material = "G4_WATER"
            sphere.mother = phantom_body.name
            sphere.translation = [x, y, z]
            sphere.color = [1, 0, 0, 0.5]  # Red semi-transparent

            sphere_volumes.append(sphere)

        # Add lung insert (low-density cylinder)
        lung_insert = sim.add_volume("TubsVolume", "NEMA_Lung_Insert")
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

        # * Not supported yet
        # if self._gen_attenuation_img:
        #     mumap = sim.add_actor("AttenuationImageActor", "mumap")
        #     mumap.image_volume = phantom_body # volume for the moment, not the name
        #     mumap.output_filename = str(Path(self._output_filename).with_suffix('.mhd'))
        #     mumap.energy = 510 * keV
        #     mumap.database = "NIST"
        #     mumap.attenuation_image.write_to_disk = True
        #     mumap.attenuation_image.active = False

        return phantom_body

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

    def add_point_source(self, position=[0, 0, 0], activity=1 * MBq, isotope="F18", name="Point_Source") -> SourceBase:
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

        source = sim.add_source("GenericSource", name)
        source.particle = "e+"

        if isotope == "F18":
            source.energy.type = "F18"
            source.half_life = 109.77 * 60 * sec
        elif isotope == "C11":
            source.energy.type = "C11"
            source.half_life = 20.38 * 60 * sec
        elif isotope == "Ga68":
            source.energy.type = "Ga68"
            source.half_life = 67.71 * 60 * sec
        else:
            source.energy.type = "mono"
            source.energy.mono = 511 * keV

        source.activity = activity
        source.position.type = "point"
        source.position.translation = position
        source.direction.type = "iso"

        self._sources.append(source)
        return source

    # point gamma source, Add by Xiaoyu
    def add_gamma_source(self, position=[0, 0, 0], energy=200.0 * keV, activity=1 * MBq, name="Gamma_Source") -> SourceBase:
        sim = self._sim
        source = sim.add_source("GenericSource", name)
        source.particle = "gamma"  
        source.energy.type = "mono"
        source.energy.mono = energy
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
@click.option('--visu', is_flag=True, default=False,
              help="Turn on visualization. This turns off multi-thread and reduce sim time.")
@click.option('--sim-time', type=float, default=0.0001, 
              help="Seconds to simulate. Default to 0.0001 sec", )
@click.option('-t', '--time-slices', type=str, default='[0]',
              help="Time slices as a comma-separated list, e.g., '[0,1,2,3]' or '0,1,2,3'")
@click.option('--output', type=click.Path(dir_okay=False, writable=True, path_type=Path), help="Output file directory.",
              default="./output/events.root")
@click.option('--scenario', type=click.Choice(['NEMA', 'Derenzo', 'Ring', 'Point', 'DualPoints','Calibration'], 
                                              case_sensitive=False), 
              help="Simulation scenario to run. Default to 'Ring'.", default="Ring")
@click.option('-n', '--num-thread', type=int, default=32, 
              help="Number of threads to use. Default to 1")
@click.option('--gen-attenuation-img', is_flag=True, 
              help="If true, save the source attention image to the same output specified with suffix .mhd")
@click.option('--scenario-params', type=str, default="",
              help="Added parameters for setting up sources.")
@click.option('--lyso', is_flag=True, 
              help="If true, use LYSO crystals instead of GAGG.")
def main(visu, sim_time, time_slices, output, scenario, num_thread, gen_attenuation_img, scenario_params, lyso):
    """
    Example usage scenarios for PET simulation.

    Uncomment the desired scenario to run:
    1. NEMA IQ Phantom with F-18 source
    2. Derenzo Phantom for spatial resolution testing
    3. Ring source configuration (for rotating detector simulation)
    4. Point source for calibration+
    
    Usage:
    
    Scenario Parameters
    -------------------
        Dualpoints:
            [dx, dy, dz] (list of float) - Provide this to control the symmetrical distance between the two sources.
            They are provided as mm. 
        
    """

    # * Verify input
    # Time
    if not re.match(r'[\[\(][0-9, ]*[\]\)]', time_slices):
        raise ValueError("Cannot parse time slices, please input a list of int or tuple or int.")
    time_slices = ast.literal_eval(f"{time_slices}")
    
    # Automatic rename
    _sim_time = sim_time * sec
    simtime_unit = 's' if _sim_time >= sec else 'ms' if _sim_time >= ms else 'us' if _sim_time >= us else 'ps'
    simtime_suffix = f"{_sim_time / sec:.0f}{simtime_unit}" if simtime_unit == 's' \
        else f"{_sim_time / us:.0f}{simtime_unit}" if simtime_unit == 'us' \
            else f"{_sim_time / ms:.0f}{simtime_unit}" if simtime_unit == 'ms' \
                else f"{_sim_time:.0f}ps"
    output = output.with_stem(output.stem + f'_{"-".join([str(s) for s in time_slices])}' + f'_{simtime_suffix}')
    print(f"Automatically renamed output: {str(output)}")

    if visu:
        num_thread = 1

    # ==========================================================================
    # SCENARIO 1: NEMA IQ Phantom (Image Quality Assessment)
    # ==========================================================================
    sim = gate.Simulation()

    # Basic simulation configuration
    add_materials(sim)
    sim.check_volumes_overlap = True
    sim.world.size = [WORLD_RADIUS * 2, WORLD_RADIUS * 2, WORLD_DEPTH]
    sim.world.material = "G4_AIR"
    sim.world.color = [1, 0, 1, 1]  # invisible

    # Create PET geometry
    pet = PETGeometry(sim, str(output), debug=False, gen_attenuation_img=gen_attenuation_img, crystal="LYSO" if lyso else "GAGG")  # Set debug=True for faster 5x5 array
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
        print(f"  - Created NEMA spheres")
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
                                     pos=[3 * cm, 0 * cm, -3 * cm], isotope="F18")
        pass
    elif scenario == "Point":
        # Point source for calibration
        print("Creating point source...")
        source = pet.add_point_source(position=[0, 0, 0], activity=240 * MBq / num_thread, isotope="F18")
        print(f"  - Position: center")
        print(f"  - Activity: 1 MBq F-18")
    elif scenario == "DualPoints":
        # Point source for calibration
        print("Creating dual point source...")
        if re.match(r'\[[\d\.]+,[\d\.]+,[\d\.]+\]', scenario_params):
            dx, dy, dz = [float(x) * mm for x in re.findall(r'\d+\.?\d*', scenario_params)]
        else:
            dx, dy, dz = 1.0 * cm, 0 * cm, 0 * cm
        # Water sphere
        water_sphere = sim.add_volume("SphereVolume", "Water_Sphere")
        water_sphere.mother = sim.world.name
        water_sphere.material = "G4_WATER"
        water_sphere.rmax = (np.sqrt((dx/2)**2 + (dy/2)**2 + (dz/2)**2) * 1.3) # covering both source
        water_sphere.color = [0, 0, 1, 1]  # blue

        source1 = pet.add_point_source(position=[-dx/2, -dy/2, -dz/2], activity=240 * MBq / num_thread, isotope="F18", name="Source1")
        source2 = pet.add_point_source(position=[+dx/2, +dy/2, +dz/2], activity=240 * MBq / num_thread, isotope="F18", name="Source2")
        print(f"  - Positions: [{-dx/2}, {-dy/2}, {-dz/2}] and [{+dx/2}, {+dy/2}, {+dz/2}]")
        print(f"  - Activity: 1 MBq F-18 each")

        # gamma source, Add by Xiaoyu
        #source1 = pet.add_gamma_source(position=[-dx/2, -dy/2, -dz/2], energy=200 * keV, activity=240 * MBq / num_thread, name="Source1")
        #source2 = pet.add_gamma_source(position=[+dx/2, +dy/2, +dz/2], energy=200 * keV, activity=240 * MBq / num_thread, name="Source2")
        
    elif scenario == "Calibration":
        print("Creating Geometry Calibration setup...")
        pet.add_calibration_source()
        # Calibration needs enough time to hit all crystals
        # Override sim_time if it's too short (e.g. default 0.0001 is too short)
        if sim_time < 1.0 and not visu:
            print(f"  [NOTE] Increasing sim-time to 1.0s for calibration coverage")

    # ==================================
    # Save phantom to attenuation images
    # ==================================
    if gen_attenuation_img:
        print(f"\n[ATTENUATION IMAGE GENERATION]")
        print(f"  Voxelizing phantom geometry...")
        fname = Path(output).with_suffix(".mhd")
        fname_imfile = fname.with_stem(fname.stem + "_image")

        # Voxelize the phantom geometry
        volume_labels, image = sim.voxelize_geometry(
            extent=phantom,
            spacing=(1, 1, 1),  # 1mm voxel spacing
            filename=str(fname)
        )

        # Verify voxelized geometry was saved
        if fname_imfile.is_file():
            print(f"  ✓ Voxelized phantom saved to: {str(fname)}")
        else:
            raise FileNotFoundError(f"  ✗ Failed to save voxelized phantom")

        # Clean up existing simulation configuration
        print(f"  Resetting simulation for attenuation calculation...")
        sim.close()
        del sim
        sim = gate.Simulation() # Creates a new sim instance

        # Basic simulation configuration
        add_materials(sim)
        sim.check_volumes_overlap = True
        sim.world.size = [WORLD_RADIUS * 2, WORLD_RADIUS * 2, WORLD_DEPTH]
        sim.world.material = "G4_AIR"
        sim.world.color = [1, 0, 1, 1]  # invisible


        # Prepare new voxelized phantom volume for attenuation calculation
        print(f"  Creating voxelized phantom volume...")
        vox_phantom = sim.add_volume("Image", "VoxelizedPhantom")
        vox_phantom.image = str(fname_imfile)  # Load the voxelized geometry
        vox_phantom.material = "G4_AIR"  # Default material (will be overridden by voxel data)
        vox_phantom.mother = sim.world.name

        # Load labels
        print(f"  Loading volume labels for attenuation mapping...")
        import json

        def labels_to_ranges(labels_dict):
            """Convert label dict to material ranges [[label, label+1, material], ...]"""
            ranges = [[info["label"], info["label"] + 1, info["material"]]
                      for info in labels_dict.values()]
            return sorted(ranges, key=lambda x: x[0])

        label_file = fname.with_stem(fname.stem + "_labels").with_suffix(".json")
        with open(label_file, 'r') as f:
            label_dict = json.load(f)

        vox_phantom.voxel_materials = labels_to_ranges(label_dict)


        print(f"  Adding attenuation calculation actor...")
        mumap = sim.add_actor("AttenuationImageActor", "AttenuationMap")
        mumap.image_volume = vox_phantom
        mumap.output_filename = str(output.with_stem(output.stem + '_mumap').with_suffix('.nii.gz'))
        mumap.energy = 511.0 * keV  # 511 keV for PET photons
        mumap.database = "NIST"
        mumap.attenuation_image.write_to_disk = True
        mumap.attenuation_image.active = True

        print(f"  Running attenuation calculation...")
        print(f"  Energy: 511 keV")
        print(f"  Output: {mumap.output_filename}")

        # Run simulation (no particles needed, just calculate attenuation from geometry)
        sim.run()
        sim.close()

        print(f"\n✓ Attenuation image generation complete!")
        print(f"  Voxelized phantom: {fname}")
        print(f"  Attenuation map: {mumap.output_filename}")
        return 

    # Simulation parameters
    sim.number_of_threads = num_thread
    sim.random_seed = 123456

    # Visualization (disable for production runs)
    sim.visu = visu

    # Run parameters (for actual simulation, not just visualization)
    # Uncomment and adjust for production:
    sim.run_timing_intervals = [(t * sec, (t + sim_time) * sec) for t in time_slices] # 1 second acquisition
    print(f"  - 🕰️ time slices {sim.run_timing_intervals}")
    # sim.number_of_events = 1e4  # Number of primary particles

    if sim.visu:
        # Program will run into a deadlock if multi-threaded.
        sim.number_of_threads = 1
        # sim.run_timing_intervals = [(0, .000001 * sec)]
        if scenario == "Derenzo":
            sim.run_timing_intervals = [(0, .00000001 * sec)]
        elif scenario == "DualPoints":
            sim.run_timing_intervals = [(0, 1E-6 * sec)]

    print("\nStarting simulation...")
    print("Close visualization window to end.")
    sim.run()
    sim.close()
    print("Simulation complete!")
    print("↪ Run process_coincidences next to obtain LORs.")
    
    # Saving the config
    pet.save_geom()
    
    
@click.command()
@click.argument('input_files', nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option('--time-window', type=float, default=3.0,
              help="Coincidence time window in nanoseconds (default: 3.0 ns)")
@click.option('--policy', type=str, default='takeWinnerOfGoods',
              help="Coincidence policy: takeAllGoods, takeWinnerOfGoods, etc. See https://opengate.readthedocs."
                   "io/en/latest/digitizer_and_detector_modeling.html#id44 for more details. (default: "
                   "takeWinnerOfGoods)")
@click.option('--min-sector-diff', type=int, default=0,
              help="Minimum sector difference for valid coincidences (default: 0). FIXME: Not implemented yet.")
@click.option('--min-transaxial-distance', type=float, default=None,
              help="Minimum transaxial distance for valid coincidences in mm (default: FOV diameter = 200.0)")
@click.option('--max-axial-distance', type=float, default=100,
              help="Maximum axial distance for valid coincidences in mm (default: 100 mm ~ panel width/height)")
@click.option('--chunk-size', type=int, default=100000,
              help="Processing chunk size for memory efficiency (default: 100000)")
@click.option('--lyso', is_flag=True, default=False,
              help="If set, use LYSO crystal properties for coincidence sorting.")
@click.option('--tree-name', '-s', type=str, default='Reads',
              help="Name of the tree in the ROOT file to process (default: 'Reads')")
def process_coincidences(input_files, time_window, policy, min_sector_diff, min_transaxial_distance, max_axial_distance, chunk_size, lyso, tree_name):
    """
    Process coincidences from one or more GATE simulation ROOT files.

    Reads the 'Reads' tree from ROOT file(s), applies coincidence sorting,
    adds detector IDs, and saves to a new ROOT file with '_coincidence' suffix.

    INPUT_FILES: One or more paths to ROOT files containing simulation data

    Example:
        python geometry_pet.py process-coincidences output/events.root
        python geometry_pet.py process-coincidences output/events_*.root --time-window 5.0
        python geometry_pet.py process-coincidences file1.root file2.root file3.root
    """
    # Set default for min_transaxial_distance if not provided
    if min_transaxial_distance is None:
        min_transaxial_distance = FOV_RADIUS * 2.0  # FOV diameter (200 mm for 10 cm FOV radius)

    # Print global settings
    print(f"\n{'='*60}")
    print(f"COINCIDENCE PROCESSING - BATCH MODE")
    print(f"{'='*60}")
    print(f"Number of files: {len(input_files)}")
    print(f"Time window: {time_window} ns")
    print(f"Policy: {policy}")
    print(f"Min sector difference: {min_sector_diff}")
    print(f"Min transaxial distance: {min_transaxial_distance} mm")
    print(f"Max axial distance: {max_axial_distance} mm")
    print(f"Chunk size: {chunk_size}")
    print(f"{'='*60}\n")

    # Accumulate all coincidences from all files
    all_coincidences = None
    total_events = 0
    processed_files = 0

    # Process each file
    for file_idx, input_file in enumerate(input_files, 1):
        print(f"\n{'='*60}")
        print(f"PROCESSING FILE {file_idx}/{len(input_files)}: {input_file.name}")
        print(f"{'='*60}")

        # Verify file exists and contains expected tree
        print_uproot_tree(input_file)
        with uproot.open(input_file) as f:
            tree_names = [key for key in f.keys() if f[key].classname.startswith('TTree')]
            print(f"Found {len(tree_names)} tree(s) in file: {tree_names}")

            if tree_name not in f:
                print(f"⚠️  WARNING: Input file does not contain '{tree_name}' tree. Skipping.")
                print(f"   Available trees: {tree_names}")
                continue

            # for LYSO, we need extra steps to simulate lower yield
            if lyso:
                from rootfiles_handlers import downsample_root_tree
                rich.print("  ⬇️ Downsampling Reads tree for LYSO crystal properties...")     
                downsample_root_tree(input_file, tree_name, "_tempfile.root", fraction=1/3.)
                input_file = "_tempfile.root"

        with uproot.open(input_file) as f: # open again in case it changes
            # Process coincidences from the Reads tree
            print("\nProcessing coincidences...")
            coincidences = coincidences_sorter(
                f[tree_name],
                time_window * gate.g4_units.nanosecond,
                policy,
                min_transaxial_distance, # min transaxial distance (in mm)
                "xy",
                max_axial_distance * gate.g4_units.mm, # max axial differences
                chunk_size=chunk_size,
                return_type="dict"
            )

            # Add rsectorID, moduleID, submoduleID, and crystalID to coincidences
            print("  Adding detector IDs to coincidences...")
            coincidences = add_detector_ids_to_coincidences(coincidences, verbose=True)

        num_events = len(coincidences[list(coincidences.keys())[0]])
        print(f"\nFound {num_events} coincidence events from this file")
        total_events += num_events
        processed_files += 1

        # Accumulate coincidences
        if all_coincidences is None:
            # First file - initialize the combined dictionary
            all_coincidences = coincidences
        else:
            # Subsequent files - concatenate arrays
            for key in coincidences.keys():
                all_coincidences[key] = np.concatenate([all_coincidences[key], coincidences[key]])

        print(f"✓ File processed. Total accumulated events: {total_events}")

    # Check if any files were processed
    if processed_files == 0:
        print("\n⚠️  No valid files were processed!")
        return

    # Determine output filename based on input files
    if len(input_files) == 1:
        # Single file - use same naming convention
        output_file = input_files[0].with_stem(input_files[0].stem + "_coincidence")
    else:
        # Multiple files - create a combined output name
        # Use the directory of the first file and create a combined name
        first_file = input_files[0]
        output_file = first_file.parent / "combined_coincidence.root"

    # Save all coincidences to a single file
    print(f"\n{'='*60}")
    print(f"SAVING COMBINED RESULTS")
    print(f"{'='*60}")
    print(f"Total coincidence events: {total_events}")
    print(f"Output file: {output_file}")

    try:
        with uproot.create(str(output_file)) as f:
            f["Coincidences"] = all_coincidences
        print(f"✓ Created new file: {output_file}")
    except FileExistsError:
        print(f"⚠️  Output file exists, overwriting: {output_file}")
        with uproot.recreate(str(output_file)) as f:
            f["Coincidences"] = all_coincidences

    # Display summary
    coincidence_id_branches = [k for k in sorted(all_coincidences.keys())
                               if 'rsectorID' in k or 'moduleID' in k or
                                  'submoduleID' in k or 'crystalID' in k]
    if coincidence_id_branches:
        print(f"ID branches: {', '.join(coincidence_id_branches)}")

    print(f"\nFirst 10 branches:")
    for key in sorted(all_coincidences.keys())[:10]:
        print(f"  - {key}")
    if len(all_coincidences) > 10:
        print(f"  ... and {len(all_coincidences) - 10} more branches")

    # Final summary
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Processed {processed_files}/{len(input_files)} file(s)")
    print(f"Total coincidence events: {total_events}")
    print(f"Output file: {output_file}")
    print(f"{'='*60}\n")


@click.group()
def cli():
    """4
    GATE PET Simulation and Analysis Tools

    Available commands:
    - simulate: Run GATE PET simulation with various scenarios
    - process-coincidences: Process coincidences from existing ROOT files
    """
    pass


# Register commands
cli.add_command(main, name='simulate')
cli.add_command(process_coincidences, name='process_coincidences')


if __name__ == "__main__":
    cli()
