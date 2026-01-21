import uproot
import re
import numpy as np
from typing import Tuple


# ==============================================================================
# DETECTOR ID PARSING FUNCTIONS
# ==============================================================================

def parse_volume_ids(volume_ids):
    """
    Parse rsectorID and crystalID from OpenGATE PreStepUniqueVolumeID.

    OpenGATE encodes repeated volume hierarchy in the format:
    'PixelizedCrystals_rep_{crystal_id}-0_{panel_id}_{crystal_id}'

    Args:
        volume_ids: Array of PreStepUniqueVolumeID strings from ROOT file

    Returns:
        tuple: (rsector_ids, crystal_ids) as numpy arrays

    Example:
        >>> parse_volume_ids(['PixelizedCrystals_rep_1355-0_6_1355'])
        (array([6]), array([1355]))
    """
    pattern = re.compile(r'PixelizedCrystals_rep_(\d+)-0_(\d+)_(\d+)')

    rsector_ids = []
    crystal_ids = []

    for vid in volume_ids:
        if isinstance(vid, bytes):
            vid = vid.decode('utf-8')

        match = pattern.match(vid)
        if match:
            crystal_id = int(match.group(1))
            panel_id = int(match.group(2))

            rsector_ids.append(panel_id)
            crystal_ids.append(crystal_id)
        else:
            # Fallback if pattern doesn't match
            rsector_ids.append(-1)
            crystal_ids.append(-1)

    return np.array(rsector_ids), np.array(crystal_ids)


def crystal_id_to_xy(crystal_id, nx=50, ny=50):
    """
    Convert flat crystal index to (x, y) coordinates in detector array.

    Args:
        crystal_id: Flat index from 0 to nx*ny-1
        nx: Number of crystals in X direction (default 50)
        ny: Number of crystals in Y direction (default 50)

    Returns:
        tuple: (x, y) coordinates in the crystal array

    Example:
        >>> crystal_id_to_xy(1355, 50, 50)
        (5, 27)
    """
    x = crystal_id % nx
    y = crystal_id // nx
    return x, y


def add_detector_ids_to_coincidences(coincidences, verbose=True):
    """
    Add rsectorID, moduleID, submoduleID, and crystalID to coincidence events.

    This function parses the PreStepUniqueVolumeID fields in coincidence data
    and adds explicit detector hierarchy IDs for both sides of each coincidence.

    Args:
        coincidences (dict): Dictionary of coincidence data from coincidences_sorter()
                            Must contain 'PreStepUniqueVolumeID1' and 'PreStepUniqueVolumeID2'
        verbose (bool): If True, print processing information (default: True)

    Returns:
        dict: Modified coincidences dictionary with added branches:
              - rsectorID1, rsectorID2: Panel IDs (0-7 for 8-panel geometry)
              - moduleID1, moduleID2: Module IDs (all 0 for single-module panels)
              - submoduleID1, submoduleID2: Submodule IDs (all 0)
              - crystalID1, crystalID2: Crystal indices in panel array

    Example:
        >>> coincidences = coincidences_sorter(...)
        >>> coincidences = add_detector_ids_to_coincidences(coincidences)
        >>> print(coincidences['rsectorID1'])  # Panel IDs for first detection
    """
    if len(coincidences) == 0:
        if verbose:
            print("    No coincidences to process")
        return coincidences

    # Check if required fields exist
    if 'PreStepUniqueVolumeID1' not in coincidences:
        if verbose:
            print("    Warning: PreStepUniqueVolumeID1 not found in coincidences")
        return coincidences

    # Parse volume IDs for both sides of coincidence
    rsector_ids_1, crystal_ids_1 = parse_volume_ids(coincidences['PreStepUniqueVolumeID1'])
    rsector_ids_2, crystal_ids_2 = parse_volume_ids(coincidences['PreStepUniqueVolumeID2'])

    # Add all IDs to coincidences dictionary
    # Side 1
    coincidences['rsectorID1'] = rsector_ids_1
    coincidences['moduleID1'] = np.zeros_like(rsector_ids_1, dtype=np.int32)
    coincidences['submoduleID1'] = np.zeros_like(rsector_ids_1, dtype=np.int32)
    coincidences['crystalID1'] = crystal_ids_1

    # Side 2
    coincidences['rsectorID2'] = rsector_ids_2
    coincidences['moduleID2'] = np.zeros_like(rsector_ids_2, dtype=np.int32)
    coincidences['submoduleID2'] = np.zeros_like(rsector_ids_2, dtype=np.int32)
    coincidences['crystalID2'] = crystal_ids_2

    if verbose:
        print(f"    Added rsectorID, moduleID, submoduleID, and crystalID to {len(rsector_ids_1)} coincidence events")
        print(f"      Panel range: {rsector_ids_1.min()}-{rsector_ids_1.max()} ↔ {rsector_ids_2.min()}-{rsector_ids_2.max()}")
        print(f"      Crystal range: {crystal_ids_1.min()}-{crystal_ids_1.max()} ↔ {crystal_ids_2.min()}-{crystal_ids_2.max()}")

    return coincidences


def add_detector_ids_to_reads(reads_tree, verbose=True):
    """
    Add rsectorID, moduleID, submoduleID, and crystalID to Reads tree data.

    This function parses the PreStepUniqueVolumeID field in Reads tree data
    and adds explicit detector hierarchy IDs.

    Args:
        reads_tree (dict): Dictionary of Reads tree data loaded from ROOT file
                          Must contain 'PreStepUniqueVolumeID' key
        verbose (bool): If True, print processing information (default: True)

    Returns:
        dict: Modified reads_tree dictionary with added branches:
              - rsectorID: Panel IDs (0-7 for 8-panel geometry)
              - moduleID: Module IDs (all 0 for single-module panels)
              - submoduleID: Submodule IDs (all 0)
              - crystalID: Crystal indices in panel array

    Example:
        >>> with uproot.open('events.root') as f:
        ...     reads = {key: f['Reads'][key].array(library='np') for key in f['Reads'].keys()}
        ...     reads = add_detector_ids_to_reads(reads)
        >>> print(reads['rsectorID'])  # Panel IDs
    """
    if 'PreStepUniqueVolumeID' not in reads_tree:
        if verbose:
            print("    Warning: PreStepUniqueVolumeID not found in Reads tree")
        return reads_tree

    # Parse volume IDs
    rsector_ids, crystal_ids = parse_volume_ids(reads_tree['PreStepUniqueVolumeID'])

    # Add parsed IDs to the tree data
    reads_tree['rsectorID'] = rsector_ids
    reads_tree['moduleID'] = np.zeros_like(rsector_ids, dtype=np.int32)      # All modules = 0
    reads_tree['submoduleID'] = np.zeros_like(rsector_ids, dtype=np.int32)   # All submodules = 0
    reads_tree['crystalID'] = crystal_ids

    if verbose:
        print(f"    Extracted rsectorID range: {rsector_ids.min()} to {rsector_ids.max()}")
        print(f"    Extracted crystalID range: {crystal_ids.min()} to {crystal_ids.max()}")
        print(f"    Unique panels detected: {len(np.unique(rsector_ids))}")
        print(f"    Set moduleID = 0, submoduleID = 0 for all readouts")

    return reads_tree


# ==============================================================================
# ROOT FILE MANIPULATION FUNCTIONS
# ==============================================================================

def change_branch_name(input_file, branch_name_mapping, output=None, target_tree=None):
    """Walk ALL trees in a ROOT file and rename branches based on a dictionary mapping.
    
    Preserves all trees and only renames branches that match the mapping.

    Args:
        input_file (str): File path to ROOT file (must be a string path, not TTree object).
        branch_name_mapping (dict): Dictionary where key is the current branch name
            and value is the new branch name.
        output (str, optional): Output file path to save all trees with renamed branches.
            If None, returns a dictionary of all processed trees without saving.

    Returns:
        dict: Dictionary where keys are tree names and values are dictionaries
            with branch names as keys and branch data as values.
            Format: {tree_name: {branch_name: array_data, ...}, ...}

    Raises:
        TypeError: If input_file is not a string path.
        ValueError: If no TTree found in the file.

    Examples:
        Process all trees and return data without saving::
        
            mapping = {"oldName1": "newName1", "oldName2": "newName2"}
            all_trees = change_branch_name("data.root", mapping)

        Process all trees and save to new file::
        
            all_trees = change_branch_name("input.root", mapping, output="renamed.root")
    """
    if not isinstance(input_file, str):
        raise TypeError("input_file must be a file path string, not a TTree object")

    all_processed_trees = {}

    # Open the input file and process all trees
    with uproot.open(input_file) as file:
        print(f"Reading ROOT file: {input_file}")

        # Get all tree names in the file
        tree_keys = [k for k in file.keys() if file[k].classname.startswith('TTree')]
        if not tree_keys:
            raise ValueError(f"No TTree found in file: {input_file}")

        print(f"Found {len(tree_keys)} tree(s): {[k.split(';')[0] for k in tree_keys]}")
        
        # Process each tree
        for tree_key in tree_keys:
            tree_name = tree_key.split(';')[0]  # Remove cycle number
            # Skip if not in target
            if not tree_name in target_tree or target_tree is None:
                print(f"Skipping tree as it's not in target_tree: {tree_key}...")
                continue
            tree = file[tree_key]

            print(f"  Processing tree '{tree_name}'...")
            renamed_data = _process_branches(tree, branch_name_mapping)
            all_processed_trees[tree_name] = renamed_data

    # Save to file if output path is provided
    if output is not None:
        print(f"\nSaving all trees to: {output}")
        with uproot.recreate(output) as out_file:
            for tree_name, tree_data in all_processed_trees.items():
                print(f"  Writing tree '{tree_name}' ({len(tree_data)} branches)")
                out_file[tree_name] = tree_data

        print(f"✓ Successfully saved {len(all_processed_trees)} tree(s) to {output}")

    return all_processed_trees


def _process_branches(tree, branch_name_mapping):
    """Helper function to process branches and apply name mapping.

    Args:
        tree (uproot.TTree): uproot TTree object.
        branch_name_mapping (dict): Dictionary mapping old names to new names.

    Returns:
        dict: Dictionary with renamed branches.
    """
    # Get all branch names
    all_branches = tree.keys()

    # Create result dictionary with renamed branches
    renamed_data = {}

    for branch_name in all_branches:
        # Remove any ROOT type suffix (e.g., "/D", "/I") from branch name
        clean_branch_name = branch_name.split('/')[0]

        # Check if this branch should be renamed
        if clean_branch_name in branch_name_mapping:
            new_name = branch_name_mapping[clean_branch_name]
            renamed_data[new_name] = tree[branch_name].array(library="np")
        else:
            # Keep original name if not in mapping
            renamed_data[clean_branch_name] = tree[branch_name].array(library="np")

    return renamed_data


def print_uproot_tree(file_path):
    """Opens a ROOT file and recursively walks through Directories, Trees, and Branches.
    
    Prints an ASCII tree illustration including branch interpretations (types).
    
    Args:
        file_path (str): Path to the ROOT file.
    """
    
    def _get_node_info(obj):
        """Returns (icon, type_label, extra_info) for a given object.
        
        Args:
            obj: Object to get information from.
            
        Returns:
            tuple: (icon, type_label, extra_info) for the object.
        """
        icon = "❓"
        type_label = type(obj).__name__
        extra_info = ""

        # 1. TTree
        if hasattr(obj, "branches") and hasattr(obj, "numentries"):
            icon = "🌲"
            type_label = "TTree"
            extra_info = f"[Entries: {obj.numentries}]"

        # 2. TBranch
        elif hasattr(obj, "interpretation"):
            icon = "🌿"
            type_label = "Branch"
            # The interpretation tells us the data structure (e.g., int32, jagged array)
            try:
                # Clean up the interpretation string for cleaner output
                interp = str(obj.interpretation)
                interp = interp.replace("AsDtype('", "").replace("')", "")
                interp = interp.replace("AsJagged", "Jagged")
                interp = interp.replace("AsStrings()", "String")
                extra_info = f": {interp}"

                # Add branch length information with comma formatting
                if hasattr(obj, "num_entries"):
                    extra_info += f" [Len: {obj.num_entries:,}]"
            except:
                extra_info = ""

        # 3. Directory
        elif hasattr(obj, "keys"):
            icon = "📁"
            type_label = "Dir"
        
        # 4. Histogram / Other
        elif hasattr(obj, "values"): 
            icon = "📊"
            type_label = "Hist"

        return icon, type_label, extra_info

    def _get_children(obj):
        """Helper to extract children based on object type.
        
        Args:
            obj: Object to extract children from.
            
        Returns:
            list: List of tuples (name, child_object).
        """
        children = []
        
        # Case A: Directories (File or TDirectory)
        # We check for 'keys' method which implies it's a directory-like object
        # Note: TTree also has keys, but we prefer treating it via branches below
        is_directory = hasattr(obj, "keys") and not hasattr(obj, "branches")
        
        if is_directory:
            try:
                # cycle=False hides multiple versions (tree;1, tree;2)
                keys = obj.keys(cycle=False) 
                for key in keys:
                    try:
                        child_obj = obj[key]
                        children.append((key, child_obj))
                    except Exception as e:
                        children.append((f"{key} <Error>", None))
            except Exception:
                pass

        # Case B: Trees or Branches
        # Both TTree and TBranch have a 'branches' attribute.
        # If a branch has sub-branches (split mode), this will find them.
        if hasattr(obj, "branches"):
            for branch in obj.branches:
                children.append((branch.name, branch))
                
        return children

    def _print_node(obj, name, prefix, is_last):
        """Recursive function to print a single node and its children.
        
        Args:
            obj: Object to print.
            name (str): Name of the object.
            prefix (str): Prefix for tree formatting.
            is_last (bool): Whether this is the last child.
        """
        
        # 1. Get visual details
        icon, type_label, extra_info = _get_node_info(obj)
        
        # 2. Determine Connector
        connector = "└── " if is_last else "├── "
        
        # 3. Print the current node
        print(f"{prefix}{connector}{icon} {name} \033[90m({type_label}{extra_info})\033[0m")

        # 4. Prepare prefix for the children
        child_prefix = prefix + ("    " if is_last else "│   ")

        # 5. Get children (Sub-branches, directory contents, etc.)
        children = _get_children(obj)
        count = len(children)
        
        for i, (child_name, child_obj) in enumerate(children):
            is_last_child = (i == count - 1)
            if child_obj is not None:
                _print_node(child_obj, child_name, child_prefix, is_last_child)

    # --- Main Execution ---
    try:
        with uproot.open(file_path) as file:
            print(f"📂 {file_path}")
            
            # Start recursion from the file level
            top_level_items = _get_children(file)
            count = len(top_level_items)
            
            if count == 0:
                print("    (Empty file)")
            
            for i, (name, obj) in enumerate(top_level_items):
                is_last = (i == count - 1)
                _print_node(obj, name, "", is_last)
                
    except Exception as e:
        print(f"Error reading file: {e}")
        
        
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def reconstruct_volume_from_vertex_positions(
    vertex_positions: np.ndarray,
    volume_origin: np.ndarray,
    volume_size: Tuple[int, int, int],
    voxel_spacing: Tuple[float, float, float],
    weights: Optional[np.ndarray] = None,
    gaussian_sigma: Optional[float] = None
) -> np.ndarray:
    """Reconstruct PET volume by histogramming TrackVertexPosition (annihilation points).
    
    This directly bins the true source positions into a 3D volume.
    
    Args:
        vertex_positions (np.ndarray): Array of TrackVertexPosition coordinates [x, y, z] in mm.
            Shape: (n_events, 3). These represent the actual positron annihilation locations.
        volume_origin (np.ndarray): Origin of the reconstruction volume [x, y, z] in mm.
            Shape: (3,).
        volume_size (Tuple[int, int, int]): Number of voxels in each dimension (nx, ny, nz).
        voxel_spacing (Tuple[float, float, float]): Voxel size in mm (dx, dy, dz).
        weights (np.ndarray, optional): Weight for each event (e.g., based on energy, timing, etc.).
            If None, all events weighted equally.
        gaussian_sigma (float, optional): Apply Gaussian smoothing with this sigma (in voxels).
            Useful to account for positron range.
        
    Returns:
        np.ndarray: Reconstructed activity distribution with shape volume_size.
    """
    
    nx, ny, nz = volume_size
    dx, dy, dz = voxel_spacing
    
    print(f"Reconstructing volume from {len(vertex_positions)} vertex positions")
    print(f"Volume: {volume_size}, spacing: {voxel_spacing} mm")
    print(f"Origin: {volume_origin} mm")
    
    # Initialize volume
    volume = np.zeros(volume_size, dtype=np.float32)
    
    # Set default weights
    if weights is None:
        weights = np.ones(len(vertex_positions))
    
    # Convert vertex positions to voxel indices
    voxel_coords = (vertex_positions - volume_origin) / np.array([dx, dy, dz])
    voxel_indices = voxel_coords.astype(int)
    
    # Count events in each voxel
    valid_events = 0
    out_of_bounds = 0
    
    for idx, (ix, iy, iz) in enumerate(voxel_indices):
        # Check if within volume bounds
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            volume[ix, iy, iz] += weights[idx]
            valid_events += 1
        else:
            out_of_bounds += 1
    
    print(f"Valid events: {valid_events}")
    print(f"Out of bounds: {out_of_bounds}")
    
    # Apply Gaussian smoothing if requested
    if gaussian_sigma is not None and gaussian_sigma > 0:
        print(f"Applying Gaussian smoothing (sigma={gaussian_sigma} voxels)...")
        volume = gaussian_filter(volume, sigma=gaussian_sigma)
    
    return volume


def reconstruct_with_histogramdd(
    vertex_positions: np.ndarray,
    volume_origin: np.ndarray,
    volume_size: Tuple[int, int, int],
    voxel_spacing: Tuple[float, float, float],
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """Fast reconstruction using numpy's histogramdd.
    
    This is typically faster than the loop-based approach for large datasets.
    
    Args:
        vertex_positions (np.ndarray): Array of vertex coordinates [x, y, z] in mm.
            Shape: (n_events, 3).
        volume_origin (np.ndarray): Origin of the reconstruction volume [x, y, z] in mm.
            Shape: (3,).
        volume_size (Tuple[int, int, int]): Number of voxels in each dimension (nx, ny, nz).
        voxel_spacing (Tuple[float, float, float]): Voxel size in mm (dx, dy, dz).
        weights (np.ndarray, optional): Weight for each event. If None, all events weighted equally.
        
    Returns:
        np.ndarray: Reconstructed volume with shape volume_size.
    """
    
    nx, ny, nz = volume_size
    dx, dy, dz = voxel_spacing
    
    print(f"Fast reconstruction using histogramdd...")
    print(f"Processing {len(vertex_positions)} events")
    
    # Define bin edges
    x_edges = np.linspace(volume_origin[0], 
                         volume_origin[0] + nx * dx, 
                         nx + 1)
    y_edges = np.linspace(volume_origin[1], 
                         volume_origin[1] + ny * dy, 
                         ny + 1)
    z_edges = np.linspace(volume_origin[2], 
                         volume_origin[2] + nz * dz, 
                         nz + 1)
    
    # Create 3D histogram
    volume, edges = np.histogramdd(
        vertex_positions,
        bins=[x_edges, y_edges, z_edges],
        weights=weights
    )
    
    return volume.astype(np.float32)


def reconstruct_with_positron_range(
    vertex_positions: np.ndarray,
    volume_origin: np.ndarray,
    volume_size: Tuple[int, int, int],
    voxel_spacing: Tuple[float, float, float],
    isotope: str = 'F18',
    tissue: str = 'water'
) -> np.ndarray:
    """Reconstruct with positron range modeling.
    
    Args:
        vertex_positions (np.ndarray): Array of vertex coordinates [x, y, z] in mm.
            Shape: (n_events, 3).
        volume_origin (np.ndarray): Origin of the reconstruction volume [x, y, z] in mm.
            Shape: (3,).
        volume_size (Tuple[int, int, int]): Number of voxels in each dimension (nx, ny, nz).
        voxel_spacing (Tuple[float, float, float]): Voxel size in mm (dx, dy, dz).
        isotope (str, optional): Isotope name ('F18', 'C11', 'O15', 'Ga68', etc.).
            Defaults to 'F18'.
        tissue (str, optional): Tissue type ('water', 'bone', 'lung', etc.).
            Defaults to 'water'. Currently not used in implementation.
        
    Returns:
        np.ndarray: Reconstructed volume with positron range modeling applied.
    """
    
    # Approximate FWHM of positron range in mm (in water)
    positron_range_fwhm = {
        'F18': 0.6,   # 18F
        'C11': 1.1,   # 11C
        'N13': 1.5,   # 13N
        'O15': 2.5,   # 15O
        'Rb82': 5.9,  # 82Rb
        'Ga68': 2.9,  # 68Ga
    }
    
    fwhm_mm = positron_range_fwhm.get(isotope, 0.6)
    
    # Convert FWHM to sigma in voxels
    # FWHM = 2.355 * sigma
    dx, dy, dz = voxel_spacing
    avg_spacing = np.mean([dx, dy, dz])
    sigma_voxels = (fwhm_mm / 2.355) / avg_spacing
    
    print(f"Modeling positron range for {isotope}")
    print(f"FWHM: {fwhm_mm} mm, Gaussian sigma: {sigma_voxels:.2f} voxels")
    
    # Reconstruct with Gaussian smoothing
    volume = reconstruct_with_histogramdd(
        vertex_positions,
        volume_origin,
        volume_size,
        voxel_spacing
    )
    
    volume = gaussian_filter(volume, sigma=sigma_voxels)
    
    return volume


def analyze_vertex_distribution(
    vertex_positions: np.ndarray,
    volume_origin: np.ndarray,
    volume_size: Tuple[int, int, int],
    voxel_spacing: Tuple[float, float, float]
):
    """Analyze the distribution of vertex positions relative to reconstruction volume.
    
    Prints statistics about vertex position distribution and coverage within the
    reconstruction volume.
    
    Args:
        vertex_positions (np.ndarray): Array of vertex coordinates [x, y, z] in mm.
            Shape: (n_events, 3).
        volume_origin (np.ndarray): Origin of the reconstruction volume [x, y, z] in mm.
            Shape: (3,).
        volume_size (Tuple[int, int, int]): Number of voxels in each dimension (nx, ny, nz).
        voxel_spacing (Tuple[float, float, float]): Voxel size in mm (dx, dy, dz).
    """
    
    nx, ny, nz = volume_size
    dx, dy, dz = voxel_spacing
    
    volume_extent = volume_origin + np.array([nx*dx, ny*dy, nz*dz])
    
    print("\n" + "="*60)
    print("Vertex Position Analysis")
    print("="*60)
    
    print(f"\nTotal events: {len(vertex_positions)}")
    
    print(f"\nVolume bounds:")
    print(f"  X: [{volume_origin[0]:.1f}, {volume_extent[0]:.1f}] mm")
    print(f"  Y: [{volume_origin[1]:.1f}, {volume_extent[1]:.1f}] mm")
    print(f"  Z: [{volume_origin[2]:.1f}, {volume_extent[2]:.1f}] mm")
    
    print(f"\nVertex position range:")
    print(f"  X: [{vertex_positions[:, 0].min():.1f}, {vertex_positions[:, 0].max():.1f}] mm")
    print(f"  Y: [{vertex_positions[:, 1].min():.1f}, {vertex_positions[:, 1].max():.1f}] mm")
    print(f"  Z: [{vertex_positions[:, 2].min():.1f}, {vertex_positions[:, 2].max():.1f}] mm")
    
    # Check coverage
    in_bounds_x = np.logical_and(
        vertex_positions[:, 0] >= volume_origin[0],
        vertex_positions[:, 0] <= volume_extent[0]
    )
    in_bounds_y = np.logical_and(
        vertex_positions[:, 1] >= volume_origin[1],
        vertex_positions[:, 1] <= volume_extent[1]
    )
    in_bounds_z = np.logical_and(
        vertex_positions[:, 2] >= volume_origin[2],
        vertex_positions[:, 2] <= volume_extent[2]
    )
    
    in_bounds = np.logical_and(in_bounds_x, np.logical_and(in_bounds_y, in_bounds_z))
    
    print(f"\nCoverage:")
    print(f"  Events within volume: {in_bounds.sum()} ({100*in_bounds.sum()/len(vertex_positions):.1f}%)")
    print(f"  Events outside volume: {(~in_bounds).sum()} ({100*(~in_bounds).sum()/len(vertex_positions):.1f}%)")


def visualize_slices(
    volume: np.ndarray,
    voxel_spacing: Tuple[float, float, float],
    title: str = "Reconstructed Volume",
    vmax: float = 0.1,
    slice_coords: Tuple[int, int, int] = None,
    slice_coords_mm: Tuple[float, float, float] = None,
    interpolation: str = 'nearest'
) -> Tuple[plt.Figure, np.ndarray]:
    """Visualize orthogonal slices of the reconstructed volume.

    Creates a figure with three orthogonal slice views: transaxial (XY),
    coronal (XZ), and sagittal (YZ).

    Args:
        volume (np.ndarray): 3D volume array to visualize. Shape: (nx, ny, nz).
        voxel_spacing (Tuple[float, float, float]): Voxel size in mm (dx, dy, dz).
        title (str, optional): Title for the figure. Defaults to "Reconstructed Volume".
        vmax (float, optional): Maximum value for colorbar clipping. Defaults to 0.1.
        slice_coords (Tuple[int, int, int], optional): Slice indices to display (slice_x, slice_y, slice_z).
            If None, defaults to center slices (nx//2, ny//2, nz//2).
        slice_coords_mm (Tuple[float, float, float], optional): Slice positions in physical coordinates (x_mm, y_mm, z_mm).
            Coordinates are relative to the volume center (origin). Cannot be used together with slice_coords.
        interpolation (str, optional): Interpolation method for image display. Options include:
            'nearest' (default, no smoothing, shows pixels clearly),
            'bilinear' (smooth, good for presentation),
            'bicubic' (smoother, may introduce artifacts),
            'gaussian' (Gaussian blur),
            'none' (same as 'nearest').
            See matplotlib.pyplot.imshow documentation for all options.

    Raises:
        ValueError: If both slice_coords and slice_coords_mm are specified.
    """

    # Validate that only one coordinate system is used
    if slice_coords is not None and slice_coords_mm is not None:
        raise ValueError("Cannot specify both slice_coords and slice_coords_mm. Use only one coordinate system.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    nx, ny, nz = volume.shape
    dx, dy, dz = voxel_spacing

    if vmax != None:
        volume = np.clip(volume, 0, vmax)

    # Determine slice indices
    if slice_coords_mm is not None:
        # Convert physical coordinates to indices
        x_mm, y_mm, z_mm = slice_coords_mm
        slice_x = int(round(x_mm / dx + nx / 2))
        slice_y = int(round(y_mm / dy + ny / 2))
        slice_z = int(round(z_mm / dz + nz / 2))
        # Validate indices
        slice_x = max(0, min(slice_x, nx - 1))
        slice_y = max(0, min(slice_y, ny - 1))
        slice_z = max(0, min(slice_z, nz - 1))
    elif slice_coords is not None:
        slice_x, slice_y, slice_z = slice_coords
        # Validate indices
        slice_x = max(0, min(slice_x, nx - 1))
        slice_y = max(0, min(slice_y, ny - 1))
        slice_z = max(0, min(slice_z, nz - 1))
    else:
        # Default to center slices
        slice_x = nx // 2
        slice_y = ny // 2
        slice_z = nz // 2

    # Convert slice indices to physical coordinates (centered at origin)
    x_pos = (slice_x - nx/2) * dx
    y_pos = (slice_y - ny/2) * dy
    z_pos = (slice_z - nz/2) * dz

    # Transaxial (XY) - centered at origin
    img_xy = volume[:, :, slice_z]
    extent_xy = [-nx*dx/2, nx*dx/2, -ny*dy/2, ny*dy/2]
    im0 = axes[0].imshow(img_xy.T, origin='lower', extent=extent_xy, cmap='hot', interpolation=interpolation)
    axes[0].set_xlabel('X (mm)')
    axes[0].set_ylabel('Y (mm)')
    axes[0].set_title(f'Transaxial (Z={z_pos:.1f} mm)')
    axes[0].axhline(y=y_pos, color='white', linestyle='--', alpha=0.3, linewidth=0.5)
    axes[0].axvline(x=x_pos, color='white', linestyle='--', alpha=0.3, linewidth=0.5)
    plt.colorbar(im0, ax=axes[0])

    # Coronal (XZ) - centered at origin
    img_xz = volume[:, slice_y, :]
    extent_xz = [-nx*dx/2, nx*dx/2, -nz*dz/2, nz*dz/2]
    im1 = axes[1].imshow(img_xz.T, origin='lower', extent=extent_xz, cmap='hot', interpolation=interpolation)
    axes[1].set_xlabel('X (mm)')
    axes[1].set_ylabel('Z (mm)')
    axes[1].set_title(f'Coronal (Y={y_pos:.1f} mm)')
    axes[1].axhline(y=z_pos, color='white', linestyle='--', alpha=0.3, linewidth=0.5)
    axes[1].axvline(x=x_pos, color='white', linestyle='--', alpha=0.3, linewidth=0.5)
    plt.colorbar(im1, ax=axes[1])

    # Sagittal (YZ) - centered at origin
    img_yz = volume[slice_x, :, :]
    extent_yz = [-ny*dy/2, ny*dy/2, -nz*dz/2, nz*dz/2]
    im2 = axes[2].imshow(img_yz.T, origin='lower', extent=extent_yz, cmap='hot', interpolation=interpolation)
    axes[2].set_xlabel('Y (mm)')
    axes[2].set_ylabel('Z (mm)')
    axes[2].set_title(f'Sagittal (X={x_pos:.1f} mm)')
    axes[2].axhline(y=z_pos, color='white', linestyle='--', alpha=0.3, linewidth=0.5)
    axes[2].axvline(x=y_pos, color='white', linestyle='--', alpha=0.3, linewidth=0.5)
    plt.colorbar(im2, ax=axes[2])

    # Set max value for colorbar consistency across views
    if vmax is None:
        vmax = volume.max()

    # Apply vmax clipping to all images
    for im in [im0, im1, im2]:
        im.set_clim(vmin=0, vmax=vmax)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    return fig, axes    
    


def visualize_slices_interactive(
    volume: np.ndarray,
    voxel_spacing: Tuple[float, float, float],
    volume_origin: np.ndarray = None,
    axis: int = 2,
    title: str = "Volume Slice Viewer",
    cmap: str = "hot",
    vmin: float = None,
    vmax: float = None
):
    """Interactive slice-by-slice visualization of a 3D volume along a specified axis.

    Creates an interactive matplotlib figure with a slider to scroll through slices
    along the specified axis. Use this for detailed inspection of the entire volume.

    Args:
        volume (np.ndarray): 3D volume array to visualize. Shape: (nx, ny, nz).
        voxel_spacing (Tuple[float, float, float]): Voxel size in mm (dx, dy, dz).
        volume_origin (np.ndarray, optional): Origin of the volume [x, y, z] in mm.
            Shape: (3,). If None, assumes origin at (0, 0, 0). Defaults to None.
        axis (int, optional): Axis along which to slice:
            0 = sagittal (YZ planes, varying X)
            1 = coronal (XZ planes, varying Y)
            2 = transaxial (XY planes, varying Z). Defaults to 2.
        title (str, optional): Title for the figure. Defaults to "Volume Slice Viewer".
        cmap (str, optional): Colormap for displaying slices. Defaults to "hot".
        vmin (float, optional): Minimum value for colormap. If None, uses volume min.
        vmax (float, optional): Maximum value for colormap. If None, uses volume max.

    Returns:
        tuple: (fig, ax, slider) matplotlib figure, axes, and slider objects.

    Example:
        >>> fig, ax, slider = visualize_slices_interactive(
        ...     reconstructed_volume,
        ...     voxel_spacing=(2.0, 2.0, 2.0),
        ...     volume_origin=np.array([-100, -100, -100]),
        ...     axis=2,  # Transaxial slices
        ...     title="OSEM Reconstruction"
        ... )
        >>> plt.show()
    """
    from matplotlib.widgets import Slider

    nx, ny, nz = volume.shape
    dx, dy, dz = voxel_spacing

    # Set default origin if not provided
    if volume_origin is None:
        volume_origin = np.array([0.0, 0.0, 0.0])

    ox, oy, oz = volume_origin

    # Set colormap limits
    if vmin is None:
        vmin = volume.min()
    if vmax is None:
        vmax = volume.max()

    # Determine axis configuration with proper origin offset
    axis_config = {
        0: {  # Sagittal (YZ planes)
            'n_slices': nx,
            'axis_name': 'X',
            'xlabel': 'Y (mm)',
            'ylabel': 'Z (mm)',
            'extent': [oy, oy + ny*dy, oz, oz + nz*dz],
            'view_name': 'Sagittal',
            'slice_coord_offset': ox,
            'slice_coord_spacing': dx
        },
        1: {  # Coronal (XZ planes)
            'n_slices': ny,
            'axis_name': 'Y',
            'xlabel': 'X (mm)',
            'ylabel': 'Z (mm)',
            'extent': [ox, ox + nx*dx, oz, oz + nz*dz],
            'view_name': 'Coronal',
            'slice_coord_offset': oy,
            'slice_coord_spacing': dy
        },
        2: {  # Transaxial (XY planes)
            'n_slices': nz,
            'axis_name': 'Z',
            'xlabel': 'X (mm)',
            'ylabel': 'Y (mm)',
            'extent': [ox, ox + nx*dx, oy, oy + ny*dy],
            'view_name': 'Transaxial',
            'slice_coord_offset': oz,
            'slice_coord_spacing': dz
        }
    }

    if axis not in axis_config:
        raise ValueError(f"Invalid axis {axis}. Must be 0 (sagittal), 1 (coronal), or 2 (transaxial).")

    config = axis_config[axis]
    n_slices = config['n_slices']

    # Create figure with space for slider
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.15)

    # Get initial slice
    initial_slice = n_slices // 2

    def get_slice(slice_idx):
        """Extract slice from volume along specified axis."""
        if axis == 0:  # Sagittal
            return volume[slice_idx, :, :].T
        elif axis == 1:  # Coronal
            return volume[:, slice_idx, :].T
        else:  # Transaxial
            return volume[:, :, slice_idx].T

    def get_slice_position(slice_idx):
        """Get physical position of slice in mm."""
        return config['slice_coord_offset'] + slice_idx * config['slice_coord_spacing']

    # Display initial slice
    img_data = get_slice(initial_slice)
    im = ax.imshow(img_data, origin='lower', extent=config['extent'],
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel(config['xlabel'], fontsize=12)
    ax.set_ylabel(config['ylabel'], fontsize=12)

    slice_pos = get_slice_position(initial_slice)
    ax.set_title(f"{title}\n{config['view_name']} View - {config['axis_name']}={slice_pos:.1f} mm (slice {initial_slice})",
                 fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Intensity', fontsize=10)

    # Create slider
    slider_ax = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(
        slider_ax,
        f'{config["axis_name"]} Slice',
        0,
        n_slices - 1,
        valinit=initial_slice,
        valstep=1
    )

    # Update function for slider
    def update(val):
        slice_idx = int(slider.val)
        img_data = get_slice(slice_idx)
        im.set_data(img_data)
        slice_pos = get_slice_position(slice_idx)
        ax.set_title(f"{title}\n{config['view_name']} View - {config['axis_name']}={slice_pos:.1f} mm (slice {slice_idx})",
                     fontsize=14, fontweight='bold')
        fig.canvas.draw_idle()

    slider.on_changed(update)

    return fig, ax, slider

def plot_line_profile_on_slice(
    volume: np.ndarray,
    voxel_spacing: Tuple[float, float, float],
    slice_axis: int,
    slice_position_mm: float,
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    volume_origin: Tuple[float, float, float] = None,
    title: str = "Line Profile",
    interpolation_order: int = 1,
    return_profile: bool = False
):
    """Draws a line and plots the intensity profile through a 2D slice.

    Coordinate System:
        - volume_origin specifies the corner (min x, min y, min z) of the volume in mm
        - If center of volume is at (0,0,0), then volume_origin = (-nx*dx/2, -ny*dy/2, -nz*dz/2)
        - Points are specified in physical world coordinates (mm)

    Args:
        volume: 3D volume array with shape (nx, ny, nz).
        voxel_spacing: Voxel size in mm as (dx, dy, dz).
        slice_axis: Axis to slice along. Options:
            - 0: Sagittal (YZ plane)
            - 1: Coronal (XZ plane)
            - 2: Transaxial (XY plane)
        slice_position_mm: Physical position of the slice in mm along slice_axis.
            For centered volume, use 0.0 for center slice.
        point1: First point in physical Cartesian coordinates (mm).
            Format depends on slice_axis:
            - axis=0 (Sagittal): (y, z)
            - axis=1 (Coronal): (x, z)
            - axis=2 (Transaxial): (x, y)
        point2: Second point in physical Cartesian coordinates (mm).
        volume_origin: Physical position of volume corner (ox, oy, oz) in mm.
            This is the minimum corner at voxel [0,0,0]. Defaults to None,
            which assumes volume is centered at origin.
            Example: For 100x100x100 volume with 2mm spacing centered at origin,
            use (-100,-100,-100).
        title: Title for the figure.
        interpolation_order: Order of spline interpolation (0-5).
            - 0: nearest neighbor
            - 1: linear (default)
            - 3: cubic
        return_profile: If True, returns profile data (positions, intensities)
            in addition to (fig, axes). Enables further analysis with
            analyze_intensity_profile.py functions.

    Returns:
        If return_profile=False (default):
            - fig: matplotlib Figure object
            - axes: matplotlib Axes array
        If return_profile=True:
            - fig: matplotlib Figure object
            - axes: matplotlib Axes array
            - positions: Position along profile in mm (np.ndarray)
            - intensities: Intensity values along profile (np.ndarray)

    Raises:
        ValueError: If slice_axis is invalid or points are identical.

    Example:
        >>> # Basic usage - visualization only
        >>> volume = np.load('recon.npy')  # shape (100, 100, 100)
        >>> fig, axes = plot_line_profile_on_slice(
        ...     volume=volume,
        ...     voxel_spacing=(2.0, 2.0, 2.0),
        ...     slice_axis=2,  # Transaxial
        ...     slice_position_mm=0.0,  # Center slice
        ...     point1=(-50.0, 0.0),  # Left of center
        ...     point2=(50.0, 0.0),   # Right of center
        ...     volume_origin=(-100.0, -100.0, -100.0)
        ... )
        >>>
        >>> # Advanced usage - with profile data for analysis
        >>> from analyze_intensity_profile import find_peak_fwhm
        >>> fig, axes, positions, intensities = plot_line_profile_on_slice(
        ...     volume=volume,
        ...     voxel_spacing=(2.0, 2.0, 2.0),
        ...     slice_axis=2,
        ...     slice_position_mm=0.0,
        ...     point1=(-50.0, 0.0),
        ...     point2=(50.0, 0.0),
        ...     volume_origin=(-100.0, -100.0, -100.0),
        ...     return_profile=True  # Get profile data
        ... )
        >>> fwhm_result = find_peak_fwhm(positions, intensities)
        >>> print(f"FWHM: {fwhm_result['fwhm']:.2f} mm")
    """
    from scipy.ndimage import map_coordinates

    dx, dy, dz = voxel_spacing
    nx, ny, nz = volume.shape

    # Set default origin if not provided - assume volume is centered at origin
    if volume_origin is None:
        ox = -nx * dx / 2.0
        oy = -ny * dy / 2.0
        oz = -nz * dz / 2.0
    else:
        ox, oy, oz = volume_origin

    # Define axis configuration (same as visualize_slice)
    axis_config = {
        0: {  # Sagittal (YZ planes)
            'axis_name': 'X',
            'xlabel': 'Y (mm)',
            'ylabel': 'Z (mm)',
            'view_name': 'Sagittal',
            'slice_coord_offset': ox,
            'slice_coord_spacing': dx,
            'extent': [oy, oy + ny*dy, oz, oz + nz*dz],
            'coord_names': ('Y', 'Z'),
            'coord_spacing': (dy, dz),
            'coord_offsets': (oy, oz),
            'array_shape': (ny, nz)
        },
        1: {  # Coronal (XZ planes)
            'axis_name': 'Y',
            'xlabel': 'X (mm)',
            'ylabel': 'Z (mm)',
            'view_name': 'Coronal',
            'slice_coord_offset': oy,
            'slice_coord_spacing': dy,
            'extent': [ox, ox + nx*dx, oz, oz + nz*dz],
            'coord_names': ('X', 'Z'),
            'coord_spacing': (dx, dz),
            'coord_offsets': (ox, oz),
            'array_shape': (nx, nz)
        },
        2: {  # Transaxial (XY planes)
            'axis_name': 'Z',
            'xlabel': 'X (mm)',
            'ylabel': 'Y (mm)',
            'view_name': 'Transaxial',
            'slice_coord_offset': oz,
            'slice_coord_spacing': dz,
            'extent': [ox, ox + nx*dx, oy, oy + ny*dy],
            'coord_names': ('X', 'Y'),
            'coord_spacing': (dx, dy),
            'coord_offsets': (ox, oy),
            'array_shape': (nx, ny)
        }
    }

    if slice_axis not in axis_config:
        raise ValueError(f"Invalid slice_axis {slice_axis}. Must be 0 (Sagittal), 1 (Coronal), or 2 (Transaxial).")

    config = axis_config[slice_axis]

    # Convert slice position to index
    slice_idx = int(np.round((slice_position_mm - config['slice_coord_offset']) / config['slice_coord_spacing']))

    # Extract 2D slice
    if slice_axis == 0:  # Sagittal
        img_2d = volume[slice_idx, :, :].T
    elif slice_axis == 1:  # Coronal
        img_2d = volume[:, slice_idx, :].T
    else:  # Transaxial
        img_2d = volume[:, :, slice_idx].T

    h, w = img_2d.shape  # Height (rows), Width (cols)

    # 2. Extract extent and convert points from mm to pixel coordinates
    extent = config['extent']  # [left, right, bottom, top] in mm
    left, right, bottom, top = extent

    # Points are in (x, y) format for this view where:
    # - x corresponds to horizontal axis (extent[0:2])
    # - y corresponds to vertical axis (extent[2:4])
    x1_mm, y1_mm = point1
    x2_mm, y2_mm = point2

    # Convert mm coordinates to pixel coordinates
    # For imshow, pixel coordinates go from 0 to w-1 (columns) and 0 to h-1 (rows)
    # The extent maps [left, right] to pixel x-range [0, w] and [bottom, top] to pixel y-range [0, h]
    x1_pix = (x1_mm - left) / (right - left) * w
    y1_pix = (y1_mm - bottom) / (top - bottom) * h
    x2_pix = (x2_mm - left) / (right - left) * w
    y2_pix = (y2_mm - bottom) / (top - bottom) * h

    if x1_pix == x2_pix and y1_pix == y2_pix:
        raise ValueError("Points must be distinct to define a line.")

    # 3. Calculate Line Extension (Edge to Edge) in pixel space
    # Parametric form: P(t) = P1 + t * (P2 - P1)
    # We find t values where the line intersects the bounding box
    vx = x2_pix - x1_pix
    vy = y2_pix - y1_pix

    candidates = []

    # Image bounds in pixel coordinates
    x_min, x_max = 0, w - 1
    y_min, y_max = 0, h - 1

    # Check intersection with vertical edges (x=0, x=w-1)
    if vx != 0:
        t_left = (x_min - x1_pix) / vx
        y_left = y1_pix + t_left * vy
        if y_min <= y_left <= y_max:
            candidates.append((x_min, y_left))

        t_right = (x_max - x1_pix) / vx
        y_right = y1_pix + t_right * vy
        if y_min <= y_right <= y_max:
            candidates.append((x_max, y_right))

    # Check intersection with horizontal edges (y=0, y=h-1)
    if vy != 0:
        t_bottom = (y_min - y1_pix) / vy
        x_bottom = x1_pix + t_bottom * vx
        if x_min <= x_bottom <= x_max:
            candidates.append((x_bottom, y_min))

        t_top = (y_max - y1_pix) / vy
        x_top = x1_pix + t_top * vx
        if x_min <= x_top <= x_max:
            candidates.append((x_top, y_max))

    # Sort candidates to ensure we draw from one side to the other
    # Remove duplicates and sort by x (or y if vertical)
    unique_candidates = sorted(list(set(candidates)))

    if len(unique_candidates) < 2:
        # Fallback: use clipped original points
        print("Warning: Could not extend line to edges properly. Using provided points clipped to image.")
        start_pt_pix = (np.clip(x1_pix, x_min, x_max), np.clip(y1_pix, y_min, y_max))
        end_pt_pix = (np.clip(x2_pix, x_min, x_max), np.clip(y2_pix, y_min, y_max))
    else:
        # Take the two most distant points found on the boundary
        start_pt_pix = unique_candidates[0]
        end_pt_pix = unique_candidates[-1]

    # 4. Sample Data along the line (in pixel space)
    # Calculate number of samples based on line length to ensure ~1 sample per pixel
    dist_pixels = np.sqrt((end_pt_pix[0] - start_pt_pix[0])**2 + (end_pt_pix[1] - start_pt_pix[1])**2)
    num_samples = int(np.ceil(dist_pixels * 2))  # 2x oversampling for smoothness

    # Generate pixel coordinates along the line
    x_coords_pix = np.linspace(start_pt_pix[0], end_pt_pix[0], num_samples)
    y_coords_pix = np.linspace(start_pt_pix[1], end_pt_pix[1], num_samples)

    # Map coordinates expects (row, col) -> (y, x)
    profile_values = map_coordinates(img_2d, [y_coords_pix, x_coords_pix], order=interpolation_order)

    # Calculate physical distance for x-axis of profile plot
    # Convert pixel deltas to mm
    dx_per_pixel = (right - left) / w
    dy_per_pixel = (top - bottom) / h
    step_dist = np.sqrt(np.diff(x_coords_pix)**2 * dx_per_pixel**2 +
                       np.diff(y_coords_pix)**2 * dy_per_pixel**2)
    # Start distance at 0
    dist_mm = np.concatenate(([0], np.cumsum(step_dist)))

    # Convert pixel coordinates back to mm for display
    start_pt_mm = (start_pt_pix[0] / w * (right - left) + left,
                   start_pt_pix[1] / h * (top - bottom) + bottom)
    end_pt_mm = (end_pt_pix[0] / w * (right - left) + left,
                 end_pt_pix[1] / h * (top - bottom) + bottom)

    # 5. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Debug info (optional - can be removed after verification)
    if False:  # Set to True to enable debug output
        print(f"Debug Info:")
        print(f"  Image shape (h, w): {h} x {w}")
        print(f"  Extent (left, right, bottom, top): {extent}")
        print(f"  Pixel spacing (dx, dy): ({dx_per_pixel:.3f}, {dy_per_pixel:.3f})")
        print(f"  Point1 (mm): {point1} -> pix: ({x1_pix:.2f}, {y1_pix:.2f})")
        print(f"  Point2 (mm): {point2} -> pix: ({x2_pix:.2f}, {y2_pix:.2f})")
        print(f"  Start (pix): {start_pt_pix} -> (mm): {start_pt_mm}")
        print(f"  End (pix): {end_pt_pix} -> (mm): {end_pt_mm}")

    # --- Subplot 1: Image with Line (in mm coordinates) ---
    # origin='lower' places index (0,0) at bottom-left, matching standard plot coordinates
    im = axes[0].imshow(img_2d, origin='lower', cmap='gray', interpolation='nearest',
                        extent=config['extent'])

    # Plot the line in mm coordinates
    axes[0].plot([start_pt_mm[0], end_pt_mm[0]], [start_pt_mm[1], end_pt_mm[1]],
                'r-', linewidth=1.5, label='Profile Line')
    axes[0].plot(start_pt_mm[0], start_pt_mm[1], 'ro', markersize=6, label='Start')
    axes[0].plot(end_pt_mm[0], end_pt_mm[1], 'rx', markersize=8, label='End')

    axes[0].set_title(f"{config['view_name']} View - {config['axis_name']}={slice_position_mm:.1f} mm (slice {slice_idx})")
    axes[0].set_xlabel(config['xlabel'], fontsize=12)
    axes[0].set_ylabel(config['ylabel'], fontsize=12)
    axes[0].legend(loc='upper right')

    # Add colorbar
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label='Intensity')

    # --- Subplot 2: Intensity Profile ---
    axes[1].plot(dist_mm, profile_values, 'b-', linewidth=2)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Highlight start and end points on the profile
    axes[1].plot(dist_mm[0], profile_values[0], 'ro', markersize=6, label='Start')
    axes[1].plot(dist_mm[-1], profile_values[-1], 'rx', markersize=8, label='End')

    axes[1].set_title(f"Intensity Profile")
    axes[1].set_xlabel("Distance (mm)")
    axes[1].set_ylabel("Voxel Value")
    axes[1].legend()

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    if return_profile:
        return fig, axes, dist_mm, profile_values
    else:
        return fig, axes


def save_volume(
    volume: np.ndarray,
    filename: str,
    voxel_spacing: Tuple[float, float, float],
    volume_origin: np.ndarray
):
    """Save reconstructed volume to file (supports .npy, .nii formats).
    
    Args:
        volume (np.ndarray): 3D volume array to save. Shape: (nx, ny, nz).
        filename (str): Output filename. Supported formats: .npy, .nii, .nii.gz.
        voxel_spacing (Tuple[float, float, float]): Voxel size in mm (dx, dy, dz).
        volume_origin (np.ndarray): Origin of the volume [x, y, z] in mm. Shape: (3,).
        
    Raises:
        ValueError: If unsupported file format is provided.
        
    Note:
        For .npy format, also saves metadata to a separate .txt file.
        For .nii format, requires nibabel package. Falls back to .npy if not available.
    """
    
    if filename.endswith('.npy'):
        # Save as NumPy array
        np.save(filename, volume)
        print(f"Saved volume to {filename}")
        
        # Save metadata
        metadata_file = filename.replace('.npy', '_metadata.txt')
        with open(metadata_file, 'w') as f:
            f.write(f"Volume shape: {volume.shape}\n")
            f.write(f"Voxel spacing (mm): {voxel_spacing}\n")
            f.write(f"Origin (mm): {volume_origin}\n")
        print(f"Saved metadata to {metadata_file}")
    
    elif filename.endswith('.nii') or filename.endswith('.nii.gz'):
        # Save as NIfTI
        try:
            import nibabel as nib
            
            # Create affine matrix
            affine = np.eye(4)
            affine[0, 0] = voxel_spacing[0]
            affine[1, 1] = voxel_spacing[1]
            affine[2, 2] = voxel_spacing[2]
            affine[:3, 3] = volume_origin
            
            # Create NIfTI image
            nifti_img = nib.Nifti1Image(volume, affine)
            nib.save(nifti_img, filename)
            print(f"Saved volume to {filename}")
            
        except ImportError:
            print("nibabel not installed. Install with: pip install nibabel")
            print("Falling back to .npy format")
            save_volume(volume, filename.replace('.nii', '.npy').replace('.gz', ''), 
                       voxel_spacing, volume_origin)
    
    else:
        raise ValueError(f"Unsupported file format: {filename}")


# Example usage
if __name__ == "__main__":
    
    # Example: Load vertex positions from GATE/opengate output
    # In practice, you would extract these from your simulation output
    
    np.random.seed(42)
    
    # Simulate a point source at (0, 0, 0) with Gaussian spread
    n_events = 100000
    source_center = np.array([0.0, 0.0, 0.0])
    source_sigma = 5.0  # mm
    
    vertex_positions = np.random.randn(n_events, 3) * source_sigma + source_center
    
    # Define reconstruction volume
    volume_origin = np.array([-50.0, -50.0, -50.0])  # mm
    volume_size = (100, 100, 100)  # voxels
    voxel_spacing = (1.0, 1.0, 1.0)  # mm
    
    # Analyze distribution
    analyze_vertex_distribution(
        vertex_positions, volume_origin, volume_size, voxel_spacing
    )
    
    # Method 1: Basic histogram
    print("\n" + "="*60)
    print("Method 1: Basic Histogram")
    print("="*60)
    volume_basic = reconstruct_with_histogramdd(
        vertex_positions, volume_origin, volume_size, voxel_spacing
    )
    visualize_slices(volume_basic, voxel_spacing, "Basic Histogram")
    
    # Method 2: With positron range modeling
    print("\n" + "="*60)
    print("Method 2: With Positron Range Modeling")
    print("="*60)
    volume_with_range = reconstruct_with_positron_range(
        vertex_positions, volume_origin, volume_size, voxel_spacing,
        isotope='F18'
    )
    visualize_slices(volume_with_range, voxel_spacing, "With Positron Range (F-18)")
    
    # Save results
    save_volume(volume_with_range, "reconstructed_volume.npy", 
                voxel_spacing, volume_origin)