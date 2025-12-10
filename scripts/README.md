# GAGG_PET Scripts

This directory contains Python scripts for PET simulations using OpenGATE and reconstruction using PyTomography.

## Core Module

### geometry_pet.py

Main module containing the `PETGeometry` class for building PET scanner simulations.

**Features:**
- Two-panel coincidence PET detector with pixelized scintillator arrays
- NEMA NU 2-2012 Image Quality phantom
- Derenzo (Hot Rod) phantom for spatial resolution testing
- Multiple source configurations (phantom, ring, point)
- Physics configuration for PET imaging
- Digitizer chain with hits collection and readout

**Class: PETGeometry**

```python
pet = PETGeometry(sim, debug=False)
```

**Methods:**

*Detector:*
- `add_pet()` - Build detector geometry

*Phantoms:*
- `add_nema_iq_phantom(activity_concentration)` - NEMA IQ phantom with 6 spheres
- `add_derenzo_phantom(rod_pattern, activity_concentration)` - Derenzo hot rod phantom

*Sources:*
- `add_phantom_source(activity, isotope)` - Add source to existing phantom
- `add_ring_source(radius, activity, isotope)` - Ring source configuration
- `add_point_source(position, activity, isotope)` - Point source for calibration

*Physics:*
- `setup_physics(physics_list)` - Configure Geant4 physics

## Example Scripts

### example_nema_phantom.py

NEMA NU 2-2012 Image Quality phantom simulation.

**Purpose:** Assess image quality metrics including:
- Spatial resolution
- Contrast recovery
- Image uniformity
- Scatter correction accuracy

**Usage:**
```bash
cd /home/lwong/Source/Repos/gagg_nm_project/Projects/GAGG_PET
python scripts/example_nema_phantom.py
```

**Configuration:**
- Phantom: 6 spheres (10, 13, 17, 22, 28, 37 mm) + lung insert
- Source: F-18, 10 MBq
- Detector: Two 50×50 panel arrays

### example_derenzo_phantom.py

Derenzo (Hot Rod) phantom for spatial resolution testing.

**Purpose:** Evaluate scanner's ability to resolve closely-spaced objects.

**Usage:**
```bash
python scripts/example_derenzo_phantom.py
```

**Configuration:**
- Rod patterns: "micro" (1.0-4.0 mm) or "clinical" (3.5-6.0 mm)
- 6 wedge sectors with increasing rod sizes
- Source: F-18, 20 MBq

**Interpretation:**
- Smallest resolvable rod diameter = spatial resolution limit
- Larger rods should be clearly separated
- Smaller rods may merge at resolution limit

### example_ring_source.py

Ring source configuration for limited-angle PET systems.

**Purpose:** Simulate rotating two-panel detectors or calibration measurements.

**Usage:**
```bash
python scripts/example_ring_source.py
```

**Configuration:**
- Ring radius: 8 cm (adjustable)
- Activity: 50 MBq F-18
- Axial extent: 5 cm

**Applications:**
- Limited-angle PET with rotation
- Sensitivity measurements
- Calibration studies

### debug.py

Quick visualization using pre-built Philips Vereos PET scanner model.

**Usage:**
```bash
python scripts/debug.py
```

**Features:**
- Pre-built clinical PET scanner geometry
- Patient table included
- Fast prototyping and geometry checking

### process_events.py

**NEW!** Process ROOT files from GATE simulations for reconstruction.

**Purpose:** Extract and process coincidence events from GATE output files.

**Usage:**
```bash
python scripts/process_events.py
```

**Features:**
- Load hits from ROOT files (using uproot)
- Apply energy window filtering (400-600 keV default)
- Detect coincidence events (10 ns time window)
- Create list-mode data structure
- Generate statistical plots
- Export processed data for reconstruction

**Outputs:**
- `output/processed/hits_data.npz` - Raw hit data
- `output/processed/coincidences.npy` - Coincidence pairs
- `output/processed/listmode_data.npz` - List-mode format for reconstruction
- `output/processed/event_statistics.png` - Quality metrics plots

**Requirements:**
```bash
pip install uproot awkward numpy matplotlib
```

### reconstruct_pet.py

**NEW!** PET image reconstruction using PyTomography.

**Purpose:** Reconstruct PET images from processed list-mode data.

**Usage:**
```bash
python scripts/reconstruct_pet.py
```

**Features:**
- Load processed list-mode data
- Create sinogram representation
- OSEM reconstruction template (PyTomography)
- Visualization of sinograms and images
- Support for iterative algorithms

**Outputs:**
- `output/reconstruction/sinogram.npy` - Sinogram data
- `output/reconstruction/reconstruction_results.png` - Visualizations

**Requirements:**
```bash
pip install pytomography torch numpy matplotlib
```

**Note:** Full OSEM reconstruction requires scanner calibration and system matrix setup. See PyTomography documentation for complete workflow.

### cli_process_events.py

**NEW!** Command-line interface for event processing.

**Purpose:** CLI tool with configurable parameters for processing ROOT files.

**Usage:**
```bash
# Basic usage
cli_process_events.py output/events.root

# Custom parameters
cli_process_events.py --time-window 5.0 --energy-min 350 --energy-max 650 output/events.root

# Specify output location
cli_process_events.py -o results/processed output/events.root

# Get help
cli_process_events.py --help
```

**Options:**
- `--output-dir, -o`: Output directory (default: ../output/processed)
- `--time-window, -t`: Coincidence window in ns (default: 10.0)
- `--energy-min`: Lower energy threshold in keV (default: 400)
- `--energy-max`: Upper energy threshold in keV (default: 600)
- `--tree-name`: ROOT tree name (default: Hits)
- `--plot/--no-plot`: Generate plots (default: yes)
- `--verbose/--quiet, -v/-q`: Verbose output (default: yes)

**Requirements:**
```bash
pip install click uproot awkward numpy matplotlib
```

### cli_reconstruct_pet.py

**NEW!** Command-line interface for reconstruction.

**Purpose:** CLI tool for sinogram creation and reconstruction preparation.

**Usage:**
```bash
# Basic usage
cli_reconstruct_pet.py output/processed/listmode_data.npz

# Custom image dimensions
cli_reconstruct_pet.py --image-size 256,256,128 --voxel-size 1.0,1.0,1.0 data.npz

# High-resolution sinogram
cli_reconstruct_pet.py --num-bins-radial 256 --num-bins-angular 360 data.npz

# Get help
cli_reconstruct_pet.py --help
```

**Options:**
- `--output-dir, -o`: Output directory (default: ../output/reconstruction)
- `--image-size`: Image dimensions as x,y,z (default: 128,128,64)
- `--voxel-size`: Voxel size in mm as dx,dy,dz (default: 2.0,2.0,2.0)
- `--num-bins-radial`: Radial bins (default: 128)
- `--num-bins-angular`: Angular bins (default: 180)
- `--plot/--no-plot`: Generate plots (default: yes)
- `--show-template/--no-template`: Show OSEM template (default: yes)
- `--verbose/--quiet, -v/-q`: Verbose output (default: yes)

**Requirements:**
```bash
pip install click numpy matplotlib
# Optional for full reconstruction:
pip install pytomography torch
```

## Complete Workflow

### Option A: CLI Workflow (Recommended)

```bash
# Step 1: Run GATE simulation
python scripts/example_ring_source.py

# Step 2: Process ROOT files (CLI)
cli_process_events.py output/events.root

# Step 3: Reconstruct images (CLI)
cli_reconstruct_pet.py output/processed/listmode_data.npz
```

### Option B: Python Script Workflow

```bash
# Step 1: Run GATE simulation
python scripts/example_ring_source.py

# Step 2: Process ROOT files
python scripts/process_events.py

# Step 3: Reconstruct images
python scripts/reconstruct_pet.py
```

## Quick Start

### Basic Simulation

```python
import opengate as gate
from geometry_pet import PETGeometry, add_materials, MBq

# Create simulation
sim = gate.Simulation()
add_materials(sim)
sim.world.material = "Air"

# Build PET detector
pet = PETGeometry(sim, debug=False)
pet.add_pet()
pet.add_digitizer()  # Add digitizer for event output
pet.setup_physics()

# Add NEMA phantom with source
pet.add_nema_iq_phantom()
pet.add_phantom_source(activity=10 * MBq, isotope="F18")

# Run production simulation
sim.visu = False
sim.run_timing_intervals = [[0, 60]]  # 60 seconds
sim.run()
```

### Production Simulation

```python
# Disable visualization for production
sim.visu = False

# Set acquisition parameters
sim.run_timing_intervals = [[0, 60]]  # 60 second scan
sim.number_of_events = 1e7  # 10 million primaries
sim.number_of_threads = 4
sim.random_seed = 123456

# Run simulation
sim.run()

# Output: output/events.root
```

## Available Isotopes

- **F-18**: Half-life 109.77 min (most common PET isotope)
- **C-11**: Half-life 20.38 min
- **Ga-68**: Half-life 67.71 min
- **Custom**: Mono-energetic 511 keV positrons

## Phantom Specifications

### NEMA IQ Phantom

Based on NEMA NU 2-2012 / IEC 61675-1 standards:
- Body: Cylindrical, ~19 cm diameter, 18 cm length
- Spheres: 6 fillable (10, 13, 17, 22, 28, 37 mm inner diameter)
- Lung insert: 51 mm diameter cylinder (low-density material)
- Material: Water-equivalent
- Typical activity: 10-50 MBq F-18

### Derenzo Phantom

Hot rod design for resolution testing:
- Body: Cylindrical, 12 cm diameter, 4 cm height
- Patterns:
  - **Micro**: 1.0, 1.5, 2.0, 2.5, 3.0, 4.0 mm rods
  - **Clinical**: 3.5, 4.0, 4.5, 5.0, 5.5, 6.0 mm rods
- Spacing: Center-to-center = 2× rod diameter
- 6 wedge sectors (60° each)
- Typical activity: 20-100 MBq F-18

## Customization

### Change Crystal Material

Edit `geometry_pet.py` line 83:

```python
pixelized_crystals.material = "LYSO"  # or "GAGG", "BGO", etc.
```

### Adjust Detector Dimensions

Edit constants in `geometry_pet.py`:

```python
CRYSTAL_SIZE_X = 2 * mm        # Crystal width
CRYSTAL_SIZE_Y = 2 * mm        # Crystal height
CRYSTAL_THICKNESS = 19 * mm    # Crystal depth
DETECTOR_SPACING = 0.1 * mm    # Gap between crystals
FOV_RADIUS = 10 * cm           # Distance from center
```

### Change Array Size

Modify `PETGeometry.__init__()`:

```python
self._nx = 50  # Crystals in X (default 50)
self._ny = 50  # Crystals in Y (default 50)
```

Use `debug=True` for 5×5 arrays (faster testing).

## Output Files

### Simulation Output

Simulations generate ROOT files in `../output/`:
- `events.root` - Default output file containing:
  - Hit positions
  - Energy deposits
  - Timing information
  - Volume IDs

### Processed Output

Processing scripts generate in `../output/processed/`:
- `hits_data.npz` - Extracted hit data from ROOT
- `coincidences.npy` - Detected coincidence pairs
- `listmode_data.npz` - List-mode format with LOR data
- `event_statistics.png` - Quality control plots

### Reconstruction Output

Reconstruction generates in `../output/reconstruction/`:
- `sinogram.npy` - Sinogram representation
- `reconstruction_results.png` - Visualization plots
- `reconstructed_images/` - OSEM reconstructed images (when fully configured)

## Performance Tips

**Fast iteration (geometry testing):**
```python
pet = PETGeometry(sim, debug=True)  # 5×5 arrays
sim.visu = True
sim.run()  # Visualization only
```

**Production simulations:**
```python
pet = PETGeometry(sim, debug=False)  # 50×50 arrays
sim.visu = False
sim.number_of_threads = 8
sim.number_of_events = 1e8  # High statistics
```

**Typical event counts:**
- Testing: 1e4 - 1e5 events
- Research: 1e6 - 1e7 events
- Publication: 1e8+ events

## Troubleshooting

**ImportError: No module named 'opengate'**
```bash
pip install opengate
```

**Visualization not working:**
- Check X11 forwarding: `echo $DISPLAY`
- Use alternative: `sim.visu_type = "vrml"`
- Try headless mode: `sim.visu = False`

**Geometry overlaps:**
```python
sim.check_volumes_overlap = True  # Enable checking
```

**Materials not found:**
- Ensure `GateMaterials.db` exists in `../` directory
- Check `add_materials()` function

## References

- OpenGATE Documentation: https://opengate-python.readthedocs.io/
- NEMA NU 2-2012 Standards: https://www.nema.org/
- Derenzo Phantom Design: Standard resolution phantom for medical imaging

## Event Processing Parameters

### Coincidence Detection

Default parameters in `process_events.py`:
- **Time Window**: 10 ns (COINCIDENCE_WINDOW)
- **Energy Window**: 400-600 keV (ENERGY_WINDOW_LOW/HIGH)
- **Coincidence Logic**: Two photons within time window

Adjust these values based on your scanner characteristics:
```python
# For better time resolution scanners
COINCIDENCE_WINDOW = 5.0  # ns

# For wider energy acceptance
ENERGY_WINDOW_LOW = 350  # keV
ENERGY_WINDOW_HIGH = 650  # keV
```

### Reconstruction Parameters

Default in `reconstruct_pet.py`:
- **Image Size**: 128×128×64 voxels
- **Voxel Size**: 2.0×2.0×2.0 mm
- **OSEM**: 8 subsets, 10 iterations

## Installation

### Minimal (Simulation Only)
```bash
pip install opengate numpy
```

### Standard (Simulation + CLI Analysis)
```bash
pip install opengate click uproot awkward numpy matplotlib
```

### Full (Simulation + CLI + Reconstruction)
```bash
pip install opengate click uproot awkward numpy matplotlib pytomography torch
```

## Complete Example Workflows

### CLI Workflow (Recommended)

```bash
# 1. Install dependencies
pip install opengate click uproot awkward matplotlib numpy

# 2. Run simulation (generates ROOT file)
cd /path/to/GAGG_PET
python scripts/example_ring_source.py

# 3. Process events with CLI
cli_process_events.py output/events.root
# Output: output/processed/listmode_data.npz, event_statistics.png

# 4. Reconstruct with CLI
cli_reconstruct_pet.py output/processed/listmode_data.npz
# Output: output/reconstruction/sinogram.npy, sinogram_analysis.png

# 5. View results
# Check output/processed/event_statistics.png
# Check output/reconstruction/sinogram_analysis.png
```

### Advanced CLI Usage

```bash
# Custom coincidence window and energy window
cli_process_events.py \
  --time-window 5.0 \
  --energy-min 350 \
  --energy-max 650 \
  output/events.root

# High-resolution reconstruction
cli_reconstruct_pet.py \
  --image-size 256,256,128 \
  --voxel-size 1.0,1.0,1.0 \
  --num-bins-radial 256 \
  --num-bins-angular 360 \
  output/processed/listmode_data.npz

# Quiet mode (minimal output)
cli_process_events.py -q output/events.root
cli_reconstruct_pet.py -q output/processed/listmode_data.npz
```

### Python Script Workflow

```bash
# 1. Install dependencies
pip install opengate uproot awkward matplotlib numpy

# 2. Run simulation
python scripts/example_ring_source.py

# 3. Process events
python scripts/process_events.py

# 4. Reconstruct
python scripts/reconstruct_pet.py
```

## Related Files

- `../CLAUDE.md` - Sub-project documentation
- `../../CLAUDE.md` - Repository-wide documentation
- `../../Notebooks/Lab.ipynb` - PyTomography reconstruction examples

## References

**PyTomography:**
- Paper: "PyTomography: A Python Library for Quantitative Medical Image Reconstruction"
- DOI: 10.1016/j.softx.2024.101909
- GitHub: https://github.com/PyTomography/PyTomography
- Docs: https://pytomography.readthedocs.io/

**GATE/OpenGATE:**
- OpenGATE: https://opengate-python.readthedocs.io/
- GATE: https://opengate.readthedocs.io/

**ROOT Analysis:**
- uproot: https://uproot.readthedocs.io/
