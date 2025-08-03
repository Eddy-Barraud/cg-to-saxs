# Script to compute structure factors from molecular dynamics trajectories
# Usage: python intensity_profile_gen.py trj.lammpsdump

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os
# Set matplotlib plotting parameters
plt.rcParams['font.size'] = 14
plt.rcParams["font.family"] = "sans-serif"
import sys
from workers import *
from workers import rc  # Import conversion factor

# Note: type_weight, type_radius, and rho_0 are now defined in workers.py


# ===================================================================
# INPUT ARGUMENTS AND PATHS
# ===================================================================
# Physical constants and simulation parameters
# Note: rc is now defined in workers.py
restart = True  # Whether to restart from saved trajectory data
skip_every_ts = 1  # Process every nth timestep to reduce computation
if len(sys.argv) < 2:
    print("Usage: python intensity_profile_gen.py <trajectory_file>")
    sys.exit(1)
traj_path = f"{sys.argv[1]}"  # Path to trajectory file from command line argument

# Set the q-max value for scattering calculation
# Warning: The higher the q_max, the more computationally intensive the calculation
q_max = 8e-1  # Maximum q value (1/Angstrom)
# q_min is automatically set based on box size (see below)
dq=0.007                    # controls resolution of q values of the 1D output histogram
density_method="dummy_in_cell"  # Method for density assignment: "default", "cic", "voxelization", "dummy_in_cell"

# Define which atom types to include in scattering calculation
typestodo = np.array([1, 2, 3, 4, 5, 6])


# ===================================================================
# MAIN SCRIPT EXECUTION
# ===================================================================

if __name__ == '__main__':
    
    print(f"\nProcessing trajectory: {traj_path}\n")

    # ===================================================================
    # TRAJECTORY LOADING AND PREPROCESSING
    # ===================================================================
    
    # Generate output filenames based on input trajectory
    basename = traj_path.rsplit('.', 2)[0]  # Remove double extension (.tar.gz, .dump.zst, etc.)
    save_trj = basename + ".npz"

    # Attempt to load preprocessed trajectory data
    if restart and os.path.exists(save_trj):
        print("Restoring saved trajectory data")
        saved_arrays = np.load(save_trj)
        data = saved_arrays["data"]
        boxes = saved_arrays["boxes"]
        ts_in_file = len(data)
    else:
        # Parse trajectory from scratch
        ts_in_file = count_pattern_simple_chunked(traj_path, "ITEM: TIMESTEP")
        print(f"\nFound {ts_in_file} timesteps in file\n")
        number_atoms = count_atoms(traj_path)
        print(f"\nBead count per frame: {number_atoms}\n")
        boxes, data = read_lammps_dump_npt(traj_path, ts_in_file, skip_every_ts, number_atoms)
        ts_in_file = len(data)  # Update count after frame skipping
        np.savez_compressed(save_trj, data=data, boxes=boxes, allow_pickle=False)

    # Clean up any frames with zero box dimensions
    if np.any(np.all(boxes == 0, axis=1)):
        print("Warning: Found frames with zero box dimensions, removing...")
        rows_different_from_zero = np.any(boxes != 0, axis=1)
        boxes = boxes[rows_different_from_zero]
        data = data[rows_different_from_zero]
        ts_in_file = len(data)

    print(f"\nFinal timestep count: {ts_in_file}\n")


    # ===================================================================
    # COORDINATE TRANSFORMATION AND UNIT CONVERSION
    # ===================================================================
    
    # Apply periodic boundary conditions (wrap coordinates into [0, L_i])
    data[..., 3:6] %= boxes[:, np.newaxis, :]
    # Convert from reduced units to real units (Angstroms)
    data[..., 3:6] *= rc
    boxes *= rc

    N = len(data[0])  # Number of beads per frame
    
    # Set q-range for scattering calculation
    q_min = 2*np.pi/(np.max(boxes))  # Minimum q limited by box size

    # ===================================================================
    # STRUCTURE FACTOR CALCULATION LOOP
    # ===================================================================
    
    # Initialize storage for results from all frames
    q_values_agg = []
    s_q_agg = []

    # Process each trajectory frame
    for framenb in range(ts_in_file):
        print(f"S(Q) progress: frame {framenb+1} / {ts_in_file}", end="\r")
        
        # Compute structure factor for this frame
        q_values_manual, avg_structure_factor = compute_one_structure_factor(
            typestodo, framenb, data, boxes, q_max, q_min, 
            dq=dq,                    # q-spacing 
            density_method=density_method,  # Use dummy particle method
            method="fft",                # Use FFT-based calculation
            accelerator="default"        # Use CPU implementation
        )
        
        # Store results
        q_values_agg.append(q_values_manual)
        s_q_agg.append(avg_structure_factor)

    # Save computed intensity profiles to file
    save_sq = basename + ".i_q.npz"
    np.savez_compressed(save_sq, q_values_agg=q_values_agg, s_q_agg=s_q_agg, allow_pickle=False)
    print(f"\nSaved intensity profiles to {save_sq}")

    # ===================================================================
    # VISUALIZATION AND OUTPUT
    # ===================================================================

    # Create intensity profile plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(q_values_agg[-1], s_q_agg[-1], 'b-', linewidth=2)
    ax.set_xlabel(r'$Q \, (\AA^{-1})$')
    ax.set_ylabel(r'$I(Q)$ (cm$^{-1}$)')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Add secondary x-axis showing length scale (2π/Q in nm)
    def forward(x):
        """Convert Q (1/Å) to length scale (nm)"""
        return np.divide(2 * np.pi, x*10, where=x!=0)

    def inverse(x):
        """Inverse transformation for secondary axis"""
        return forward(x)

    # Create secondary axis
    secax = ax.secondary_xaxis('top', functions=(forward, inverse))
    secax.set_xlabel(r"Length scale: $2\pi / Q$ (nm)")
    secax.xaxis.set_major_formatter(ScalarFormatter())

    # Final plot formatting and save
    plt.tight_layout()
    
    # Save plot
    plot_filename = basename + ".i_q.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_filename}")

    print("\nIntensity profile calculation completed successfully!")

