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

**Requirements:**

You will need two conda envrionment, one for opengate 10, another for pytomography. For pytomography, if you need cuda support, you need to first install cupy before anything:

```bash
conda install -c conda-forge cupy cuda-version>=12
```

Install `torch` next, following online instructions. Doesn't matter that your cuda versions does not match. Then you are up for installing pytomography. See their official website for more.

```bash
pip install pytomography torch numpy matplotlib tqdm
```

Then you will need to have opengate as another environment. I used v10.0.2 for fast installation. 


## Complete Workflow

### Option A: CLI Workflow (Recommended)

```bash
# Step 1: Run GATE simulation
python geometry_pet.py simulate --scenario dualpoints --sim-time 1E-3 -n 32 --output ./output/dualpoints.root 

# Step 2: Process ROOT files (CLI) to get coincidnece, use --LYSO for generating LOR with LYSO properties
python geometry_pet.py process_coincidence ./output/dualpoints_0_50ms.root

# Step 3: Reconstruct images (CLI)
# See juptyernotebook
```

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

* Check if your overlapping volumes are `childed`

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
- **Time Window**: 3 ns (COINCIDENCE_WINDOW)
- **Energy Window**: 480-540 keV (ENERGY_WINDOW_LOW/HIGH)
- **Coincidence Logic**: Two photons within time window

### Reconstruction Parameters

See jupyter notebook for the parameters. 

### ⚠️ Necessary modification to `opengate`

Some of the functions are cooked for image voxelization and generation of attenuation map. I am fixing them to try more complicated phantoms, but I endup not being able to msuter enough resources for the simuation. There's always some problem occur if I stretch the scan time. 

```python
def create_image_with_extent(extent, spacing=(1, 1, 1), margin=0):
    # define the new size and spacing  
    extent = np.asarray(extent) <<<<Add this line>>>>
    spacing = np.array(spacing).astype(float)
    size = np.ceil((extent[1] - extent[0]) / spacing).astype(int) + 2 * margin

    # create image
    image = create_3d_image(size, spacing)

    # The origin is considered to be at the center of first pixel.
    # It is set such that the image is at the exact extent (bounding volume).
    # The volume contour thus goes through the center of the first pixel.
    origin = extent[0] + spacing / 2.0 - margin
    image.SetOrigin(origin)
    return image

```


### ⚠️ Necessary modifications to `pytomography`

#### /parallelproj/backend.py:551 & :660

Because cupy tensor is not mixing well with torch tensor used by pytomography, this modification is a manual fix

```python
# This is needed to prevent a mix of cupy and pytorch cuda tensor
if is_cuda_array(img):
    return xp.from_dlpack(img_fwd)
else:
    return xp.asarray(img_fwd, device=array_api_compat.device(img))

```

A few more places need to be fixed, just search for `xp.asarray` and replace all those clauses with the above snippet, but remember to *change the variables*. 
