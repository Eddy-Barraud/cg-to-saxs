# Script to compute structure factors from molecular dynamics trajectories
# Usage: python structure_factor.py trj.lammpsdump

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os
# Set matplotlib plotting parameters
plt.rcParams['font.size'] = 14
plt.rcParams["font.family"] = "sans-serif"
import zstandard as zstd  # For compressed file reading
import gzip                # For gzip compressed files
import sys
import math


# Physical constants and simulation parameters
rc=6.589  # Angstroms, conversion factor from reduced units to Angstroms
restart = True  # Whether to restart from saved trajectory data
skip_every_ts = 40  # Process every nth timestep to reduce computation
traj_path = f"{sys.argv[1]}"  # Path to trajectory file from command line argument

def cloud_in_cell_vectorized(density_beads, density_weights, num_points_x, num_points_y, num_points_z, Lx, Ly, Lz):
    """
    Vectorized implementation of the Cloud-in-Cell (CIC) method for density interpolation.
    
    This method distributes particle densities onto a regular grid by interpolating
    each particle's contribution to the 8 nearest grid points using trilinear interpolation.
    
    Parameters:
    -----------
    density_beads : np.ndarray
        Particle positions (N, 3)
    density_weights : np.ndarray
        Particle weights/electron counts (N,)
    num_points_x, num_points_y, num_points_z : int
        Grid dimensions in each direction
    Lx, Ly, Lz : float
        Box dimensions in Angstroms
        
    Returns:
    --------
    electron_density : np.ndarray
        3D density grid (num_points_x, num_points_y, num_points_z)
    voxel_edges : list
        Grid edge positions for each dimension
    """
    # Grid spacing
    dx = Lx / num_points_x
    dy = Ly / num_points_y
    dz = Lz / num_points_z

    # Normalized positions (fractional grid coordinates)
    gx = density_beads[:, 0] / dx
    gy = density_beads[:, 1] / dy
    gz = density_beads[:, 2] / dz

    # Lower grid indices (floor of fractional coordinates)
    ix = np.floor(gx).astype(int)
    iy = np.floor(gy).astype(int)
    iz = np.floor(gz).astype(int)

    # Fractional distances from lower grid point
    fx = gx - ix
    fy = gy - iy
    fz = gz - iz

    # Initialize electron density grid
    electron_density = np.zeros((num_points_x, num_points_y, num_points_z))

    # Each bead contributes to 8 neighboring grid points (trilinear interpolation)
    # Loop over the 8 corners of the grid cell
    for dx_ in [0, 1]:
        # Weight in x-direction: (1-fx) for lower corner, fx for upper corner
        wx = np.where(dx_ == 0, 1 - fx, fx)
        x_idx = ix + dx_

        # Check if x-indices are within grid bounds
        valid_x = (x_idx >= 0) & (x_idx < num_points_x)

        for dy_ in [0, 1]:
            # Weight in y-direction
            wy = np.where(dy_ == 0, 1 - fy, fy)
            y_idx = iy + dy_

            # Check if y-indices are within grid bounds
            valid_y = (y_idx >= 0) & (y_idx < num_points_y)

            for dz_ in [0, 1]:
                # Weight in z-direction
                wz = np.where(dz_ == 0, 1 - fz, fz)
                z_idx = iz + dz_

                # Check if z-indices are within grid bounds
                valid_z = (z_idx >= 0) & (z_idx < num_points_z)

                # Only keep indices within bounds
                mask = valid_x & valid_y & valid_z

                # Extract valid indices and compute total weights
                xi = x_idx[mask]
                yi = y_idx[mask]
                zi = z_idx[mask]
                w = density_weights[mask] * wx[mask] * wy[mask] * wz[mask]

                # Add contributions using np.add.at (handles repeated indices correctly)
                np.add.at(electron_density, (xi, yi, zi), w)

    # Create voxel edge arrays for each dimension
    voxel_edges = [
        np.linspace(0, Lx, num_points_x + 1),
        np.linspace(0, Ly, num_points_y + 1),
        np.linspace(0, Lz, num_points_z + 1)
    ]

    return electron_density, voxel_edges

def generate_points_in_sphere(center, radius, num_points):
    """
    Generate approximately num_points inside a sphere using a uniform grid in spherical coordinates.
    Includes the center point explicitly to ensure good sampling near the origin.

    This function creates a quasi-uniform distribution of points within a sphere by:
    1. Using a cubic root scaling for radial positions to achieve uniform volume density
    2. Creating a uniform angular grid in spherical coordinates
    3. Explicitly including the center point

    Args:
        center (array-like): Center of the sphere (length-3)
        radius (float): Sphere radius
        num_points (int): Desired total number of points

    Returns:
        np.ndarray: Points inside the sphere, shape (N, 3)
    """
    # Distribute points among r, theta, phi for approximately cubic grid
    n_r = int(np.round(num_points ** (1/3)))
    if n_r < 2:
        n_r = 2
    # Reserve one point for center, distribute rest among shells
    points_per_shell = int(np.ceil((num_points-1) / (n_r-1)))  
    n_theta = int(np.sqrt(points_per_shell))
    n_phi = int(np.ceil(points_per_shell / n_theta))

    # Uniform grid in r (cube root for uniform volume distribution)
    r_lin = np.linspace(0, 1, n_r, endpoint=False)
    r = r_lin ** (1/3)  # Cube root scaling for uniform volume density

    # Uniform grid in angles (skip theta=0 to avoid duplicate pole points)
    theta = np.linspace(0, np.pi, n_theta, endpoint=False)[1:]  
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)

    # Start with center point
    points = [np.array(center)]  
    
    # Generate points on each spherical shell
    for ri in r[1:]:  # skip r=0 (center already included)
        for ti in theta:
            for pi in phi:
                # Convert spherical to Cartesian coordinates
                x = ri * np.sin(ti) * np.cos(pi)
                y = ri * np.sin(ti) * np.sin(pi)
                z = ri * np.cos(ti)
                # Scale by radius and translate to center
                xyz = np.array([x, y, z]) * radius + np.array(center)
                points.append(xyz)

    # Trim to exact number of points if we generated too many
    points = np.array(points)
    if len(points) > num_points:
        indices = np.linspace(0, len(points)-1, num_points).astype(int)
        points = points[indices]

    return points

def voxels_in_cell(density_beads, density_weights, beads_radius, num_points_per_sphere, num_points_x, num_points_y, num_points_z, Lx, Ly, Lz):
    """
    Voxelization method for density assignment using spherical volume sampling.
    
    This method represents each bead as a sphere and samples points within that sphere
    to create a more accurate representation of the bead's finite size and shape.
    
    Parameters:
    -----------
    density_beads : np.ndarray
        Bead center positions (N, 3)
    density_weights : np.ndarray  
        Bead electron weights (N,)
    beads_radius : np.ndarray
        Radius of each bead (N,)
    num_points_per_sphere : int
        Number of sample points per sphere
    num_points_x, num_points_y, num_points_z : int
        Grid dimensions
    Lx, Ly, Lz : float
        Box dimensions
        
    Returns:
    --------
    electron_density : np.ndarray
        3D density grid
    voxel_edges : list
        Grid edge positions
    """
    # Generate unit sphere points (radius = 1.0)
    sphere_points = generate_points_in_sphere(center=[0, 0, 0], radius=1.0, num_points=num_points_per_sphere)
    num_points_per_sphere = sphere_points.shape[0]  # Update count in case of padding
    
    # Scale sphere points by each bead's radius
    # Broadcasting: (1, num_points_per_sphere, 3) * (n_beads, 1, 1) -> (n_beads, num_points_per_sphere, 3)
    sphere_points_scaled = sphere_points[None, :, :] * beads_radius[:, None, None]

    # Generate all partial bead positions by adding scaled sphere points to bead centers
    n_beads = density_beads.shape[0]
    # shape: (n_beads, num_points_per_sphere, 3)
    all_partial_beads = density_beads[:, None, :] + sphere_points_scaled
    # Reshape to (n_beads * num_points_per_sphere, 3) for histogramming
    all_partial_beads = all_partial_beads.reshape(-1, 3)

    # Distribute weights evenly among all sample points within each bead
    all_partial_weights = np.repeat(density_weights / num_points_per_sphere, num_points_per_sphere)

    # Use numpy's histogramdd for efficient 3D binning
    electron_density, voxel_edges = np.histogramdd(
        all_partial_beads,
        bins=(num_points_x, num_points_y, num_points_z),
        range=[[0, Lx], [0, Ly], [0, Lz]],
        weights=all_partial_weights,
        density=False
    )

    return electron_density, voxel_edges

def dummy_in_cell(beads_positions, beads_weights, beads_radius,
                  num_dummies, num_points_x, num_points_y, num_points_z, 
                  box_lengths):
    """
    Dummy particle method for density assignment using random sampling within spheres.
    
    This method creates multiple random "dummy" particles within each bead's spherical volume
    to represent the electron density distribution. Uses uniform random sampling within spheres.
    
    Parameters:
    -----------
    beads_positions : np.ndarray
        Bead center positions (N, 3)
    beads_weights : np.ndarray
        Bead electron weights (N,)
    beads_radius : np.ndarray
        Radius of each bead (N,)
    num_dummies : int
        Number of dummy particles per bead
    num_points_x, num_points_y, num_points_z : int
        Grid dimensions
    box_lengths : np.ndarray
        Box dimensions (3,)
        
    Returns:
    --------
    electron_density : np.ndarray
        3D density grid
    voxel_edges : list
        Grid edge positions
    """
    Lx, Ly, Lz = box_lengths
    
    # Generate indices for all dummy particles
    # For each bead, create num_dummies dummy particles
    repeated_indices = np.repeat(np.arange(beads_positions.shape[0]), num_dummies)
    total_density_beads = len(repeated_indices)

    # Generate random directions (normal distribution, then normalize to unit vectors)
    rand_dirs = np.random.normal(0, 1, size=(total_density_beads, 3))
    rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)
    
    # Generate random radii using cube root for uniform volume distribution
    bead_radius_for_density = beads_radius[repeated_indices].reshape(-1, 1)
    rand_radii = np.cbrt(np.random.uniform(0, 1, size=(total_density_beads, 1))) * bead_radius_for_density
    
    # Calculate final random offsets within each sphere
    random_offsets = rand_dirs * rand_radii

    # Compute dummy particle positions and apply periodic boundary conditions
    density_beads = beads_positions[repeated_indices] + random_offsets
    density_beads %= box_lengths  # Wrap particles outside box back inside
    
    # Distribute weights evenly among dummy particles
    density_weights = np.repeat(beads_weights, num_dummies) / num_dummies

    # Use numpy's histogramdd for 3D binning
    electron_density, voxel_edges = np.histogramdd(
        density_beads,
        density=False,
        weights=density_weights,
        bins=(num_points_x, num_points_y, num_points_z),
        range=[[0, Lx], [0, Ly], [0, Lz]]
    )
    return electron_density, voxel_edges


def compute_structure_factor(beads_positions, beads_types, box_lengths, q_max, q_min, dq, density_method="default"):
    """
    Compute structure factor using FFT-based method with various density assignment schemes.
    
    This function calculates the small-angle X-ray scattering (SAXS) structure factor S(q)
    by building an electron density grid and computing its Fourier transform.
    
    Parameters:
    -----------
    beads_positions : np.ndarray
        Bead positions (N, 3) in Angstroms
    beads_types : np.ndarray
        Bead type indices (N,)
    box_lengths : np.ndarray
        Box dimensions (3,) in Angstroms
    q_max : float
        Maximum q value (1/Angstrom)
    q_min : float
        Minimum q value (1/Angstrom)
    dq : float
        q-spacing for binning
    density_method : str
        Method for density assignment: "default", "cic", "voxelization", "dummy_in_cell"
        
    Returns:
    --------
    q_bin_centers : np.ndarray
        q-values at bin centers
    intensity_absolute : np.ndarray
        Absolute intensity I(q)
    """
    # Calculate number of q-bins and prepare input arrays
    nqval = int((q_max - q_min) / dq)
    beads_positions = np.asarray(beads_positions)
    beads_weight = type_weight[beads_types.astype(int) - 1]  # Electron counts per bead
    beads_weight = np.asarray(beads_weight)
    beads_radius = type_radius[beads_types.astype(int) - 1]  # Bead radii
    beads_radius = np.asarray(beads_radius)
    box_lengths = np.asarray(box_lengths)

    # Extract box dimensions and calculate system volume
    Lx, Ly, Lz = box_lengths  # Angstrom 
    sysvol = np.prod(box_lengths * 1e-10*1e2)  # Convert to cm^3

    # Compute grid size based on maximum q-value (Nyquist criterion)
    num_points_x = int(np.ceil(Lx * q_max/(2*np.pi)))
    num_points_y = int(np.ceil(Ly * q_max/(2*np.pi)))
    num_points_z = int(np.ceil(Lz * q_max/(2*np.pi)))

    # Classical electron radius for absolute intensity normalization
    r_e = 2.818e-13  # cm

    # Build electron density grid using specified method
    if density_method == "cic":
        # Cloud-in-cell interpolation method
        electron_density, voxel_edges = cloud_in_cell_vectorized(beads_positions, beads_weight, num_points_x, num_points_y, num_points_z, Lx, Ly, Lz)
    elif density_method == "voxelization":
        # Spherical voxelization with 100 sample points per sphere
        electron_density, voxel_edges = voxels_in_cell(beads_positions, beads_weight, beads_radius, 100, num_points_x, num_points_y, num_points_z, Lx, Ly, Lz)
    elif density_method == "dummy_in_cell":
        # Dummy particle method with 100 particles per bead
        electron_density, voxel_edges = dummy_in_cell(
            beads_positions, beads_weight, beads_radius,
            100, num_points_x, num_points_y, num_points_z, box_lengths
        )
    elif density_method == "default":
        # Simple histogram binning (no finite-size effects)
        electron_density, voxel_edges = np.histogramdd(
            beads_positions,
            density=False,
            weights=beads_weight,
            bins=(num_points_x, num_points_y, num_points_z),
            range=[[0, Lx], [0, Ly], [0, Lz]]
        )

    # Calculate voxel volume for density normalization
    voxel_volume = (
        (voxel_edges[0][1] - voxel_edges[0][0]) *
        (voxel_edges[1][1] - voxel_edges[1][0]) *
        (voxel_edges[2][1] - voxel_edges[2][0])
    )  # A^3

    # Verify electron count conservation
    assert np.allclose(np.sum(electron_density), np.sum(beads_weight), rtol=1)
    print(f"\n\tVoxel volume of {voxel_volume:.3e} A³, side length {voxel_edges[0][1] - voxel_edges[0][0]:.3e} A")
    print(f"\n\tFrame with {beads_positions.shape[0]} beads and {np.sum(beads_weight)} electrons")
    print(f"\tFrame electron density: {np.mean(electron_density)/voxel_volume:.2e} e-.A-3")
    
    # Compute contrast density (subtract solvent background)
    contrast_density = electron_density - rho_0*voxel_volume

    # Compute 3D Fourier transform of contrast density
    amplitude = np.fft.fftn(contrast_density * r_e)
    intensity_3d = np.abs(amplitude) ** 2

    # Generate reciprocal space (k-space) grid
    q_x = np.fft.fftfreq(num_points_x, d=Lx / (num_points_x * 2 * np.pi))
    q_y = np.fft.fftfreq(num_points_y, d=Ly / (num_points_y * 2 * np.pi))
    q_z = np.fft.fftfreq(num_points_z, d=Lz / (num_points_z * 2 * np.pi))

    # Compute q magnitudes for spherical averaging
    q_x_v, q_y_v, q_z_v = np.meshgrid(q_x, q_y, q_z, indexing="ij", sparse=True)
    q_magnitude = np.sqrt(q_x_v ** 2 + q_y_v ** 2 + q_z_v ** 2)


    # Perform spherical averaging to convert 3D intensity to 1D I(q)
    # Reduce q_max slightly to avoid edge effects in FFT
    q_max = np.max(q_magnitude) * 0.9  
    
    # Bin the 3D intensity by q-magnitude using histogram weighting
    h, edges = np.histogram(
        q_magnitude.ravel(),
        bins=nqval,
        range=(q_min, q_max),
        weights=intensity_3d.ravel(),
        density=False,
    )

    # Calculate bin centers and count how many voxels fall in each bin
    q_bin_centers = (edges[1:] + edges[:-1]) / 2.0
    counts, _ = np.histogram(q_magnitude.ravel(), bins=edges)

    # Average intensity within each q-bin (avoid division by zero)
    Iq_averaged = np.divide(
        h, counts, out=np.zeros_like(h, dtype=np.float64), where=counts != 0
    )

    # Convert to absolute intensity units (cm^-1)
    # Formula: I(q) = |F(q)|^2 / V_sample
    # Note: Optional corrections for sample thickness and transmission can be added
    intensity_absolute = Iq_averaged / sysvol

    # Validate output shapes and remove zero-intensity points
    if q_bin_centers.shape != intensity_absolute.shape:
        print("error of shape")
        return False
    
    mask = intensity_absolute != 0

    return q_bin_centers[mask], intensity_absolute[mask]


def count_pattern_simple_chunked(filename, pattern, chunk_size=2**24):
    """
    Count occurrences of a pattern in a file using chunked reading for memory efficiency.
    
    This function handles compressed files (.gz, .zst, .zstd) and processes large files
    in chunks to avoid memory issues when counting specific patterns like "ITEM: TIMESTEP".
    
    Parameters:
    -----------
    filename : str
        Path to the file to search
    pattern : str
        String pattern to count
    chunk_size : int
        Size of chunks to read at once (default: 16MB)
        
    Returns:
    --------
    int : Number of pattern occurrences found
    """
    try:
        # Determine file opening method based on extension
        if filename.endswith('.gz'):
            my_open = gzip.open
        elif filename.endswith('.zst') or filename.endswith('.zstd'):
            my_open = zstd.open
        else:
            my_open = open
            
        with my_open(filename, 'rt', encoding='utf-8') as f:
            count = 0
            while True:
                chunk = f.read(chunk_size)
                if not chunk:  # End of file reached, process remaining line by line
                    while True:
                        line = f.readline()
                        if line:
                            count += line.count(pattern)
                        else:
                            break
                    break
                else:
                    count += chunk.count(pattern)
            return count
    except FileNotFoundError:
        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

def read_lammps_dump_npt(filename, ts_in_file, skip_every_ts, number_atoms):
    """
    Read and parse a LAMMPS dump file from NPT ensemble simulations.
    
    This function handles trajectory files with varying box sizes (NPT ensemble)
    and implements robust error handling for incomplete or corrupted timesteps.
    It supports compressed files and memory-efficient processing.
    
    Parameters:
    -----------
    filename : str
        Path to the LAMMPS dump file
    ts_in_file : int
        Expected number of timesteps in the file
    skip_every_ts : int
        Skip factor - keep only every nth frame
    number_atoms : int
        Expected number of atoms per frame
        
    Returns:
    --------
    boxes : np.ndarray
        Box dimensions for each frame, shape (n_used_frames, 3)
    data_list : np.ndarray
        Trajectory data, shape (n_used_frames, number_atoms, 9)
        
    Notes:
    ------
    - If a timestep is repeated, overwrites its data and zeros out subsequent data
    - Removes all unused rows at the end
    - Supports .gz, .zst, and .zstd compressed files
    """
    # Initialize storage arrays
    boxes = np.zeros((ts_in_file, 3), dtype=float)
    data_list = np.zeros((ts_in_file, number_atoms, 9), dtype=float)
    timestep_to_index = {}  # Map timestep number to array index
    filled_indices = set()  # Track which indices have valid data
    ts = 0
    last_ts = -1

    # Determine file opening method based on compression
    if filename.endswith('.gz'):
        my_open = gzip.open
    elif filename.endswith('.zst') or filename.endswith('.zstd'):
        my_open = zstd.open
    else:
        my_open = open

    with my_open(filename, 'rt', encoding="UTF-8") as f:
        while True:
            line = f.readline()
            if not line:  # End of file reached
                break
            if "ITEM: TIMESTEP" in line:
                timestep = int(f.readline().strip())
                # Handle duplicate timesteps by overwriting and clearing subsequent data
                if timestep in timestep_to_index:
                    print(f"\nWarn: timestep {timestep} was already seen, overwriting...\n")
                    idx = timestep_to_index[timestep]
                    # Zero out all data after this timestep
                    boxes[idx+1:] = 0
                    data_list[idx+1:] = 0
                    ts = idx  # Reset to overwrite this index
                else:
                    ts += 1
                    idx = ts - 1
                    timestep_to_index[timestep] = idx
                last_ts = timestep
                print(f"Reading progress: {ts/ts_in_file*100:.2f}%", end="\r")
            if ts >= ts_in_file:
                print(f"Reading progress: {ts/ts_in_file*100:.2f}%")
                break
            if "ITEM: NUMBER OF ATOMS" in line:
                N = int(f.readline().strip())
                # Validate atom count consistency
                if N != number_atoms:
                    raise ValueError(f"Number of atoms in file ({N}) doesn't match expected ({number_atoms})")
            if "ITEM: BOX BOUNDS" in line:
                # Read three box boundary lines (lo hi for each dimension)
                box_lines = [f.readline() for _ in range(3)]
                box_data = []
                for box_line in box_lines:
                    lo, hi = box_line.split()
                    box_data.append(np.abs(float(hi)-float(lo)))
                # Only store box data for frames we're keeping
                if ts % skip_every_ts == 0:
                    boxes[idx] = box_data
            if "ITEM: ATOMS" in line:
                # Skip atom data if this timestep is not being kept
                if (ts) % skip_every_ts != 0:
                    try:
                        [next(f) for _ in range(N)]  # Skip N lines efficiently
                    except:
                        break
                    continue
                    
                # Read atom data for this timestep
                add_data = True
                data = np.zeros((N, 9), dtype=float)
                for j in range(N):
                    data_tmp = [float(k) for k in f.readline().split()]
                    num_cols = len(data_tmp)
                    if num_cols != 9:
                        print(f"Incomplete line at timestep {ts} with {num_cols} columns. Skipping.")
                        add_data = False
                        break
                    data[j] = data_tmp
                    
                # Store valid data and mark this index as filled
                if add_data:
                    data_list[idx, :, :] = data
                    filled_indices.add(idx)

    # Clean up arrays by removing unused rows (keep only filled indices)
    used_indices = sorted(list(filled_indices))
    boxes = boxes[used_indices]
    data_list = data_list[used_indices]

    # Validate that we have some data
    if data_list.shape[0] == 0:
        print("\nError: empty data list\n")
        quit(1)
    return np.array(boxes), np.array(data_list)


def count_atoms(filename):
    """
    Extract the number of atoms from the first frame of a LAMMPS dump file.
    
    This function reads through the file until it finds the first "ITEM: NUMBER OF ATOMS"
    entry and returns that count. Supports compressed files.
    
    Parameters:
    -----------
    filename : str
        Path to the LAMMPS dump file
        
    Returns:
    --------
    int : Number of atoms per frame
    
    Raises:
    -------
    ValueError : If no "ITEM: NUMBER OF ATOMS" found in file
    """
    if filename.endswith('.gz'):
        my_open = gzip.open
    elif filename.endswith('.zst') or filename.endswith('.zstd'):
        my_open = zstd.open
    else:
        my_open = open

    with my_open(filename, 'rt') as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError("No 'ITEM: NUMBER OF ATOMS' found in file.")
            if "ITEM: NUMBER OF ATOMS" in line:
                N_found = int(f.readline().strip())
                return N_found


def binned_mean_with_error(x, y, bin_width=2):
    """
    Compute binned statistics with error bars for scattered data.
    
    This function bins x-y data into regular intervals and computes mean and
    standard error for each bin. Useful for averaging noisy data or multiple
    measurements at similar x-values.
    
    Parameters:
    -----------
    x : np.ndarray
        Independent variable values
    y : np.ndarray  
        Dependent variable values
    bin_width : float
        Width of bins for averaging
        
    Returns:
    --------
    bin_centers : np.ndarray
        Center x-value of each bin
    bin_means : np.ndarray
        Mean y-value in each bin
    bin_errors : np.ndarray
        Standard error of mean in each bin
    """

    # Sort data by x-values for proper binning
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Create bin edges and centers
    bins = np.arange(min(x), max(x) + bin_width, bin_width)

    # Assign data points to bins
    binned_data = [y[np.logical_and(x >= b, x < b + bin_width)] for b in bins[:-1]]

    # Calculate statistics for each bin
    bin_centers = bins[:-1] + bin_width / 2
    bin_means = np.array([np.mean(bin_data) for bin_data in binned_data])
    bin_stds = np.array([np.std(bin_data, ddof=1)*2 for bin_data in binned_data])  # 2-sigma confidence
    bin_errors = bin_stds / np.sqrt([len(bin_data) for bin_data in binned_data])  # Standard error of mean

    return bin_centers, bin_means, bin_errors


def compute_one_structure_factor(typestodo, framenb, traj, boxes, q_max, q_min, dq, density_method="default", method="fft", accelerator="default"):
    """
    Compute the structure factor for one frame, supporting different methods and accelerators.

    Parameters
    ----------
    typestodo : np.ndarray
        Array of atom types to include.
    framenb : int
        Frame index to process.
    traj : np.ndarray
        Trajectory array of shape (n_frames, n_atoms, n_cols).
    boxes : np.ndarray
        Box lengths for each frame.
    q_max : float
        Maximum q value.
    q_min : float
        Minimum q value.
    dq : float
        q step size.
    cic : bool, optional
        Whether to use cloud-in-cell, by default True.
    method : str, optional
        "fft" (default) or "direct".
    accelerator : str, optional
        "default", "cuda", or "mlx".
    
    Returns
    -------
    tuple of np.ndarray
        Bin centers and structure factor.
    """
    # Check input types
    assert isinstance(typestodo, np.ndarray)
    assert isinstance(traj, np.ndarray)
    assert isinstance(boxes, np.ndarray)

    frame = traj[framenb]
    box = boxes[framenb]
    assert frame.shape[-1] >= 6

    # Extract atomic positions and types for the current frame
    beads_positions = frame[..., 3:6]
    types = frame[..., 2]
    working_positions = beads_positions[np.isin(types, typestodo)]
    working_types = types[np.isin(types, typestodo)]

    if len(working_positions) == 0:
        print("Error selecting the atom types")
        return False, False

    if method == "fft":
        if accelerator == "default":
            bin_centers, structure_factor = compute_structure_factor(
                working_positions, working_types, box, q_max, q_min, dq, density_method=density_method
            )
        else:
            raise ValueError(f"Accelerator '{accelerator}' not supported for method 'fft'")
    elif method == "direct":
        # Note: s_q_direct function implementation not included in this file
        raise NotImplementedError("Direct method requires s_q_direct function implementation")
        # bin_centers, structure_factor = s_q_direct(
        #     working_positions, working_types, box, qmax=q_max, qmin=q_min, dq=dq, nq_vec=1000, density_method=density_method, accelerator=accelerator
        # )
    else:
        raise ValueError(f"Unknown method '{method}'")

    return bin_centers, structure_factor


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
    
    # ===================================================================
    # SIMULATION PARAMETERS AND SCATTERING FORM FACTORS
    # ===================================================================
    
    # Define which atom types to include in scattering calculation
    typestodo = np.array([1, 2, 3, 4, 5, 6])
    
    # Set q-range for scattering calculation
    q_min = 2*np.pi/(np.max(boxes)*rc)  # Minimum q limited by box size
    q_max = 8e-1  # Maximum q value (1/Angstrom)
    
    # SAXS electron form factors for each bead type
    # Format: [S atoms + C atoms + H atoms + O atoms]
    type_weight = np.array([
        (16+3*8),      # Type 1: S + 3O atoms  
        (2*6+3*9+8),   # Type 2: 2C + 3F + O atoms
        (2*6+4*9),     # Type 3: 2C + 4F atoms
        (2*6+4*9),     # Type 4: 2C + 4F atoms  
        (3*8+7*1),     # Type 5: 3O + 7H atoms
        (3*8+6*1)      # Type 6: 3O + 6H atoms
    ]).astype(np.int32)  # Electron counts per bead type
    
    # Effective radii for each bead type (Angstroms)
    type_radius = np.array([0.361, 0.400, 0.385, 0.385, 0.381, 0.381]) * rc

    # Solvent electron density (water background)
    rho_0 = 30/(6.36**3)  # electrons/A^3 for DPD water beads
    

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
            dq=0.007,                    # q-spacing 
            density_method="dummy_in_cell",  # Use dummy particle method
            method="fft",                # Use FFT-based calculation
            accelerator="default"        # Use CPU implementation
        )
        
        # Store results
        q_values_agg.append(q_values_manual)
        s_q_agg.append(avg_structure_factor)

    # Save computed structure factors to file
    save_sq = basename + ".s_q.npz"
    np.savez_compressed(save_sq, q_values_agg=q_values_agg, s_q_agg=s_q_agg, allow_pickle=False)
    print(f"\nSaved structure factors to {save_sq}")
   
    # ===================================================================
    # VISUALIZATION AND OUTPUT
    # ===================================================================
    
    # Create structure factor plot
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
    plt.title("Small-Angle X-ray Scattering Structure Factor")
    plt.tight_layout()
    
    # Save plot
    plot_filename = basename + ".s_q.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_filename}")
    
    print("\nStructure factor calculation completed successfully!")

