import uproot
import pandas as pd
import numpy as np
import os
import awkward as ak
from typing import List, Dict, Callable, Optional
from tqdm.auto import tqdm


class ROOTFileHandler:
    """
    Unified handler for filtering ROOT files with configurable column schemas.

    Supports both single detector ID and dual detector ID (coincidence) modes.
    """

    def __init__(self, columns: List[str], energy_col: str = "energy", time_col: str = "time"):
        """
        Initialize the handler with column configuration.

        Args:
            columns: List of column names to read from ROOT file
            energy_col: Name of the energy column
            time_col: Name of the time column
        """
        self.columns = columns
        self.energy_col = energy_col
        self.time_col = time_col

        # Determine detector ID columns (all columns except energy and time)
        self.detector_id_cols = [col for col in columns if col not in [energy_col, time_col]]
        self.is_dual_id = len(self.detector_id_cols) == 2

    def filter_hits_chunked(self, input_file: str, tree_name: str, output_file: str,
                           time_window: float = 10.0, chunk_size: int = 100000):
        """
        Filters hits using a moving window approach, processing data in chunks.

        Args:
            input_file: Path to input ROOT file
            tree_name: Name of the tree in ROOT file
            output_file: Path to output ROOT file
            time_window: Time window for filtering (ns)
            chunk_size: Number of events per chunk
        """
        if os.path.exists(output_file):
            os.remove(output_file)

        outfile = uproot.recreate(output_file)
        buffer_df = pd.DataFrame()
        first_write = True

        mode = "Dual ID" if self.is_dual_id else "Single ID"

        # Get total number of entries for progress bar
        with uproot.open(f"{input_file}:{tree_name}") as tree:
            total_entries = tree.num_entries

        print(f"Processing {input_file} ({mode})")
        print(f"Total entries: {total_entries:,}, chunk size: {chunk_size:,}")

        # Progress bar for overall processing
        total_processed = 0
        total_accepted = 0

        for batch_idx, chunk_ak in enumerate(uproot.iterate(f"{input_file}:{tree_name}",
                                               expressions=self.columns,
                                               step_size=chunk_size,
                                               library="ak")):

            # 1. Convert Awkward Array to Pandas
            chunk_df = self._ak_to_pandas(chunk_ak)

            # 2. Concatenate with buffer from previous iteration
            if not buffer_df.empty:
                current_batch = pd.concat([buffer_df, chunk_df], ignore_index=True)
            else:
                current_batch = chunk_df

            # 3. Sort by time
            current_batch = current_batch.sort_values(self.time_col).reset_index(drop=True)

            # 4. Determine Safe Time Cutoff
            if len(current_batch) > 0:
                max_time = current_batch[self.time_col].iloc[-1]
                min_time = current_batch[self.time_col].iloc[0]

                # Handle infinite or very large time window
                if np.isinf(time_window) or time_window >= (max_time - min_time):
                    # Time window covers entire batch - process all but keep last event in buffer
                    # to handle potential conflicts with next chunk
                    if len(current_batch) > 1:
                        safe_time_cutoff = max_time - 1e-9  # Process all except very last moment
                    else:
                        safe_time_cutoff = min_time - 1  # Process nothing, keep in buffer
                else:
                    safe_time_cutoff = max_time - time_window
            else:
                safe_time_cutoff = 0

            # 5. Split data into "Processable" and "Next Buffer"
            df_to_process = current_batch[current_batch[self.time_col] < safe_time_cutoff].copy()
            buffer_df = current_batch[current_batch[self.time_col] >= safe_time_cutoff].copy()

            if len(df_to_process) == 0:
                print(f"Batch {batch_idx}: Buffering {len(buffer_df)} events")
                continue

            # 6. Run filtering logic with progress bar
            print(f"\nBatch {batch_idx}: Filtering {len(df_to_process):,} events...")
            filtered_df = self._apply_filter_logic(df_to_process, time_window, show_progress=True)

            # 7. Write to ROOT file
            self._write_to_root(outfile, tree_name, filtered_df, first_write)
            if first_write and len(filtered_df) > 0:
                first_write = False

            total_processed += len(df_to_process)
            total_accepted += len(filtered_df)

            print(f"  → Accepted: {len(filtered_df):,}/{len(df_to_process):,} ({100*len(filtered_df)/len(df_to_process):.1f}%), Buffer: {len(buffer_df):,}")

        # Final Step: Process remaining buffer
        if not buffer_df.empty:
            print(f"\n{'='*60}")
            print(f"Processing final buffer ({len(buffer_df):,} events)...")
            filtered_final = self._apply_filter_logic(buffer_df, time_window, show_progress=True)
            self._write_to_root(outfile, tree_name, filtered_final, first_write)
            total_processed += len(buffer_df)
            total_accepted += len(filtered_final)
            print(f"  → Final buffer accepted: {len(filtered_final):,}/{len(buffer_df):,}")

        outfile.close()

        # Summary statistics
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Input events:  {total_entries:,}")
        print(f"Output events: {total_accepted:,}")
        print(f"Rejection rate: {100*(1-total_accepted/total_entries):.2f}%")
        print(f"{'='*60}")

    def _ak_to_pandas(self, ak_array) -> pd.DataFrame:
        """Convert awkward array to pandas DataFrame."""
        data_dict = {col: ak_array[col].to_numpy() for col in self.columns}
        return pd.DataFrame(data_dict)

    def _write_to_root(self, outfile, tree_name: str, df: pd.DataFrame, first_write: bool):
        """Write DataFrame to ROOT file."""
        if len(df) == 0:
            return

        data_dict = {col: df[col].to_numpy() for col in self.columns}

        if first_write and tree_name not in outfile:
            outfile[tree_name] = data_dict
        else:
            outfile[tree_name].extend(data_dict)

    def _apply_filter_logic(self, df: pd.DataFrame, time_window: float, show_progress: bool = False) -> pd.DataFrame:
        """
        Core filtering logic that handles both single and dual ID modes.

        For each event, checks if there's a conflict in the time window:
        - Dual ID mode: conflict if detector IDs overlap (set intersection)
        - Single ID mode: conflict if detector ID matches

        Higher energy event wins in case of conflict.
        """
        df = df.copy()
        df['temp_idx'] = range(len(df))

        # Build column list for extraction
        extract_cols = self.detector_id_cols + [self.energy_col, self.time_col, 'temp_idx']
        vals = df[extract_cols].values
        n_rows = len(vals)

        accepted_indices = []
        active_window = []  # [time, detector_ids, energy, temp_idx]

        # Create progress bar for filtering loop
        iterator = tqdm(range(n_rows), desc="  Filtering", unit="events", disable=not show_progress) if show_progress else range(n_rows)

        for i in iterator:
            current_row = vals[i]

            # Parse based on mode
            if self.is_dual_id:
                id1, id2 = current_row[0], current_row[1]
                detector_ids = {id1, id2}
                e_curr = current_row[2]
                t_curr = current_row[3]
                idx_curr = current_row[4]
            else:
                detector_ids = current_row[0]  # Single value
                e_curr = current_row[1]
                t_curr = current_row[2]
                idx_curr = current_row[3]

            # Clean expired events from active window
            while active_window and active_window[0][0] < (t_curr - time_window):
                active_window.pop(0)

            # Check for conflicts
            conflict_found = False
            conflict_idx = -1

            for k, item in enumerate(active_window):
                if self.is_dual_id:
                    # Set intersection check
                    if not detector_ids.isdisjoint(item[1]):
                        conflict_found = True
                        conflict_idx = k
                        break
                else:
                    # Direct equality check
                    if detector_ids == item[1]:
                        conflict_found = True
                        conflict_idx = k
                        break

            if not conflict_found:
                # No conflict - add to window and accept
                active_window.append([t_curr, detector_ids, e_curr, idx_curr])
                accepted_indices.append(idx_curr)
            else:
                # Conflict found - higher energy wins
                prev_event = active_window[conflict_idx]
                if e_curr > prev_event[2]:
                    # Current event wins
                    prev_idx_val = prev_event[3]
                    if prev_idx_val in accepted_indices:
                        accepted_indices.remove(prev_idx_val)

                    active_window[conflict_idx] = [t_curr, detector_ids, e_curr, idx_curr]
                    accepted_indices.append(idx_curr)
                # else: Previous event wins, current discarded

            # Update progress bar postfix every 1000 events
            if show_progress and isinstance(iterator, tqdm) and i % 1000 == 0:
                iterator.set_postfix({
                    "accepted": len(accepted_indices),
                    "window": len(active_window),
                    "rate": f"{100*len(accepted_indices)/(i+1):.1f}%"
                })

        return df[df['temp_idx'].isin(accepted_indices)].drop(columns=['temp_idx'])


# ---------------------------------------------------------
# Convenience functions for common use cases
# ---------------------------------------------------------

def filter_hits_chunked(input_file: str, tree_name: str, output_file: str,
                       time_window: float = 10.0, chunk_size: int = 100000):
    """
    Filters hits with dual detector IDs (coincidence mode).

    Args:
        input_file: Path to input ROOT file
        tree_name: Name of the tree in ROOT file
        output_file: Path to output ROOT file
        time_window: Time window for filtering (ns)
        chunk_size: Number of events per chunk
    """
    columns = ["PreStepUniqueVolumeID", "detectorID2", "TotalEnergyDeposit", "GlobalTime"]
    handler = ROOTFileHandler(columns, energy_col="TotalEnergyDeposit", time_col="GlobalTime")
    handler.filter_hits_chunked(input_file, tree_name, output_file, time_window, chunk_size)


def filter_hits_chunked_single_id(input_file: str, tree_name: str, output_file: str,
                                  time_window: float = 10.0, chunk_size: int = 100000):
    """
    Filters hits with single detector ID.

    Args:
        input_file: Path to input ROOT file
        tree_name: Name of the tree in ROOT file
        output_file: Path to output ROOT file
        time_window: Time window for filtering (ns)
        chunk_size: Number of events per chunk
    """
    columns = ["PreStepUniqueVolumeID", "TotalEnergyDeposit", "GlobalTime"]
    handler = ROOTFileHandler(columns, energy_col="TotalEnergyDeposit", time_col="GlobalTime")
    handler.filter_hits_chunked(input_file, tree_name, output_file, time_window, chunk_size)


def downsample_root_tree(input_file, tree_name, output_file, keep_columns=None, fraction=0.1, chunk_size=100000, seed=42):
    """
    Downsamples a ROOT tree by selecting specific columns and keeping a random fraction of events.
    
    Args:
        input_file (str): Path to the input ROOT file.
        tree_name (str): Name of the tree to process.
        output_file (str): Path to the output ROOT file.
        keep_columns (list): List of column names (branches) to keep. All others are discarded.
        fraction (float): Fraction of events to keep (0.0 to 1.0). e.g., 0.1 means keep 10%.
        chunk_size (int): Number of entries to process at a time.
        seed (int): Random seed for reproducibility.
    """
    
    # Validation
    if not (0.0 < fraction <= 1.0):
        raise ValueError("Fraction must be between 0.0 and 1.0")
    
    # Clean up existing output file
    if os.path.exists(output_file):
        os.remove(output_file)

    print(f"Downsampling {input_file}...")
    print(f"  - Keeping columns: {keep_columns}")
    print(f"  - Target fraction: {fraction*100}%")
    
    # Initialize random number generator
    rng = np.random.default_rng(seed)
    
    # Create output file
    outfile = uproot.recreate(output_file)
    
    total_original = 0
    total_kept = 0
    first_write = True
    
    if keep_columns is None:
        # If no columns specified, keep all
        with uproot.open(f"{input_file}") as f:
            keep_columns = f[tree_name].keys()
    
    # Iterate through the file in chunks
    # We only read the columns we intend to keep to save memory/IO
    for batch in uproot.iterate(f"{input_file}:{tree_name}", 
                                expressions=keep_columns, 
                                step_size=chunk_size, 
                                library="np"): # Using numpy library for speed
        
        n_events = len(batch[keep_columns[0]]) # Get number of events in this batch
        total_original += n_events
        
        # Generate a boolean mask for random selection
        # This creates an array of True/False where True appears with probability 'fraction'
        mask = rng.random(n_events) < fraction
        
        # Apply mask to all columns in the batch
        filtered_batch = {}
        for col in keep_columns:
            filtered_batch[col] = batch[col][mask]
            
        n_kept_in_batch = len(filtered_batch[keep_columns[0]])
        total_kept += n_kept_in_batch
        
        # Write to output file
        if n_kept_in_batch > 0:
            if first_write:
                outfile[tree_name] = filtered_batch
                first_write = False
            else:
                outfile[tree_name].extend(filtered_batch)
                
        print(f"  Processed chunk: {n_events} -> {n_kept_in_batch} entries.")

    outfile.close()
    
    print("-" * 30)
    print(f"Done.")
    print(f"Original entries: {total_original}")
    print(f"Final entries:    {total_kept} ({total_kept/total_original:.2%})")

