
import numpy as np
import zstandard as zstd  # For compressed file reading
import gzip                # For gzip compressed files
import gc  # For garbage collection

# Physical constants and simulation parameters
rc = 6.589  # Angstroms, conversion factor from reduced units to Angstroms

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

# Global cache for precomputed Gaussian kernels
_gaussian_kernel_cache = {}


def _cache_key_for_gaussian_kernels(unique_radii, grid_spacings):
    """
    Create a hashable cache key for Gaussian kernel parameters.

    Parameters:
    -----------
    unique_radii : np.ndarray
        Unique bead radii
    grid_spacings : tuple
        Grid spacing in each dimension (dx, dy, dz)

    Returns:
    --------
    tuple : Hashable cache key
    """
    # Convert numpy array to tuple for hashing, round to avoid floating point precision issues
    # Sort the radii and round to handle floating point precision
    radii_sorted = np.sort(unique_radii)
    radii_tuple = tuple(np.round(radii_sorted, decimals=8))

    # Round grid spacings more aggressively to handle small variations in NPT simulations
    # Use fewer decimals to group similar grid spacings together
    # Round to nearest 0.01 to handle NPT box fluctuations
    spacings_tuple = tuple(np.round(np.array(grid_spacings), decimals=2))
    return (radii_tuple, spacings_tuple)

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

    OPTIMIZED VERSION: Uses chunked processing to reduce memory usage and improve performance.

    This method represents each bead as a sphere and samples points within that sphere
    to create a more accurate representation of the bead's finite size and shape.
    Uses chunked processing similar to dummy_in_cell for memory efficiency.

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
    n_beads = density_beads.shape[0]

    # Calculate optimal chunk size to limit memory usage (aim for ~100MB chunks)
    total_voxel_points = n_beads * num_points_per_sphere
    target_chunk_size = min(1000000, total_voxel_points)  # Max 1M voxel points per chunk
    chunk_size = max(1, target_chunk_size // num_points_per_sphere)  # Number of beads per chunk

    # Pre-calculate bin edges for efficiency
    x_edges = np.linspace(0, Lx, num_points_x + 1)
    y_edges = np.linspace(0, Ly, num_points_y + 1)
    z_edges = np.linspace(0, Lz, num_points_z + 1)
    voxel_edges = [x_edges, y_edges, z_edges]

    # Generate unit sphere points once (radius = 1.0) - shared across all chunks
    sphere_points = generate_points_in_sphere(center=[0, 0, 0], radius=1.0, num_points=num_points_per_sphere)
    num_points_per_sphere = sphere_points.shape[0]  # Update count in case of padding

    # Initialize the electron density grid
    electron_density = np.zeros((num_points_x, num_points_y, num_points_z), dtype=np.float64)

    # Process beads in chunks to reduce memory usage
    for chunk_start in range(0, n_beads, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_beads)

        # Extract chunk data
        chunk_positions = density_beads[chunk_start:chunk_end]
        chunk_weights = density_weights[chunk_start:chunk_end]
        chunk_radii = beads_radius[chunk_start:chunk_end]

        # Scale sphere points by each bead's radius in this chunk
        # Broadcasting: (1, num_points_per_sphere, 3) * (chunk_n_beads, 1, 1) -> (chunk_n_beads, num_points_per_sphere, 3)
        sphere_points_scaled = sphere_points[None, :, :] * chunk_radii[:, None, None]

        # Generate all partial bead positions by adding scaled sphere points to bead centers
        # shape: (chunk_n_beads, num_points_per_sphere, 3)
        chunk_partial_beads = chunk_positions[:, None, :] + sphere_points_scaled
        # Reshape to (chunk_n_beads * num_points_per_sphere, 3) for histogramming
        chunk_partial_beads = chunk_partial_beads.reshape(-1, 3)

        # Distribute weights evenly among all sample points within each bead in this chunk
        chunk_partial_weights = np.repeat(chunk_weights / num_points_per_sphere, num_points_per_sphere)

        # Apply periodic boundary conditions in-place
        chunk_partial_beads[:, 0] %= Lx
        chunk_partial_beads[:, 1] %= Ly
        chunk_partial_beads[:, 2] %= Lz

        # Add this chunk's contribution to the density grid
        chunk_density, _ = np.histogramdd(
            chunk_partial_beads,
            bins=(num_points_x, num_points_y, num_points_z),
            range=[[0, Lx], [0, Ly], [0, Lz]],
            weights=chunk_partial_weights,
            density=False
        )

        # Accumulate density (in-place addition)
        electron_density += chunk_density

        # Clean up chunk arrays to free memory immediately
        del sphere_points_scaled, chunk_partial_beads, chunk_partial_weights, chunk_density
        del chunk_positions, chunk_weights, chunk_radii

        # Force garbage collection to free memory immediately
        gc.collect()

    return electron_density, voxel_edges

def dummy_in_cell(beads_positions, beads_weights, beads_radius,
                  num_dummies, num_points_x, num_points_y, num_points_z,
                  box_lengths):
    """
    Dummy particle method for density assignment using random sampling within spheres.

    OPTIMIZED VERSION: Uses chunked processing to reduce memory usage and improve performance.

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
    n_beads = beads_positions.shape[0]

    # Calculate optimal chunk size to limit memory usage (aim for ~100MB chunks)
    total_dummies = n_beads * num_dummies
    target_chunk_size = min(1000000, total_dummies)  # Max 1M dummy particles per chunk
    chunk_size = max(1, target_chunk_size // num_dummies)  # Number of beads per chunk

    # Pre-calculate bin edges for efficiency
    x_edges = np.linspace(0, Lx, num_points_x + 1)
    y_edges = np.linspace(0, Ly, num_points_y + 1)
    z_edges = np.linspace(0, Lz, num_points_z + 1)
    voxel_edges = [x_edges, y_edges, z_edges]

    # Initialize the electron density grid
    electron_density = np.zeros((num_points_x, num_points_y, num_points_z), dtype=np.float64)

    # Process beads in chunks to reduce memory usage
    for chunk_start in range(0, n_beads, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_beads)
        chunk_n_beads = chunk_end - chunk_start

        # Extract chunk data
        chunk_positions = beads_positions[chunk_start:chunk_end]
        chunk_weights = beads_weights[chunk_start:chunk_end]
        chunk_radii = beads_radius[chunk_start:chunk_end]

        # Generate dummy particles for this chunk only
        chunk_total_dummies = chunk_n_beads * num_dummies

        # Generate random directions using optimized approach
        # Use more efficient random number generation
        rand_dirs = np.random.standard_normal((chunk_total_dummies, 3))
        # Normalize in-place to save memory
        norms = np.linalg.norm(rand_dirs, axis=1, keepdims=True)
        rand_dirs /= norms

        # Generate random radii with optimized memory usage
        rand_uniform = np.random.random(chunk_total_dummies)
        rand_radii = np.cbrt(rand_uniform)  # Cube root for uniform volume distribution

        # Expand bead data for dummy particles (more memory efficient)
        bead_indices = np.repeat(np.arange(chunk_n_beads), num_dummies)
        chunk_bead_positions = chunk_positions[bead_indices]
        chunk_bead_radii = chunk_radii[bead_indices]
        chunk_bead_weights = chunk_weights[bead_indices] / num_dummies

        # Calculate dummy positions more efficiently
        scaled_offsets = rand_dirs * (rand_radii[:, np.newaxis] * chunk_bead_radii[:, np.newaxis])
        dummy_positions = chunk_bead_positions + scaled_offsets

        # Apply periodic boundary conditions in-place
        dummy_positions[:, 0] %= Lx
        dummy_positions[:, 1] %= Ly
        dummy_positions[:, 2] %= Lz

        # Add this chunk's contribution to the density grid
        chunk_density, _ = np.histogramdd(
            dummy_positions,
            bins=(num_points_x, num_points_y, num_points_z),
            range=[[0, Lx], [0, Ly], [0, Lz]],
            weights=chunk_bead_weights,
            density=False
        )

        # Accumulate density (in-place addition)
        electron_density += chunk_density

        # Clean up chunk arrays to free memory immediately
        del rand_dirs, rand_uniform, rand_radii, scaled_offsets, dummy_positions, chunk_density
        del bead_indices, chunk_bead_positions, chunk_bead_radii, chunk_bead_weights

        # Force garbage collection to free memory immediately
        gc.collect()

    return electron_density, voxel_edges


def precompute_gaussian_kernels(unique_radii, grid_spacings):
    """
    Precompute 3D Gaussian kernels for all unique bead radii with caching.

    This function creates optimized Gaussian kernels that can be placed at bead positions
    using fast convolution methods, avoiding repeated exponential calculations.
    Results are cached to avoid recomputation for identical parameters.

    Parameters:
    -----------
    unique_radii : np.ndarray
        Unique bead radii in the system
    grid_spacings : tuple
        Grid spacing in each dimension (dx, dy, dz)

    Returns:
    --------
    kernels : dict
        Dictionary mapping radius to 3D Gaussian kernel array
    kernel_extents : dict
        Dictionary mapping radius to kernel size (for placement)
    """
    # Check cache first
    cache_key = _cache_key_for_gaussian_kernels(unique_radii, grid_spacings)
    #print(f"   Cache key: {cache_key}")
    #print(f"   Cache has {len(_gaussian_kernel_cache)} entries")
    if cache_key in _gaussian_kernel_cache:
        print(f"   Using cached Gaussian kernels for {len(unique_radii)} unique radii")
        return _gaussian_kernel_cache[cache_key]

    print(f"   Computing new Gaussian kernels for {len(unique_radii)} unique radii")
    kernels = {}
    kernel_extents = {}
    dx, dy, dz = grid_spacings

    for radius in unique_radii:
        sigma = radius / 3.0  # 3-sigma rule

        # Determine kernel size (extend to 3*sigma in each direction)
        # Add 1 to ensure odd dimensions for centered kernels
        nx = 2 * int(np.ceil(3 * sigma / dx)) + 1
        ny = 2 * int(np.ceil(3 * sigma / dy)) + 1
        nz = 2 * int(np.ceil(3 * sigma / dz)) + 1

        # Create coordinate grids centered at kernel center
        x = np.arange(nx) * dx - (nx // 2) * dx
        y = np.arange(ny) * dy - (ny // 2) * dy
        z = np.arange(nz) * dz - (nz // 2) * dz

        # Create 3D meshgrid
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Calculate squared distances
        r_squared = X**2 + Y**2 + Z**2
        r = np.sqrt(r_squared)

        # Apply radius cutoff
        mask = r <= radius

        # Create Gaussian kernel
        kernel = np.zeros((nx, ny, nz), dtype=np.float64)
        if np.any(mask):
            gaussian_values = np.exp(-r_squared[mask] / (2 * sigma**2))
            kernel[mask] = gaussian_values

            # Normalize kernel to unit weight for later scaling
            total_weight = np.sum(kernel)
            if total_weight > 0:
                kernel /= total_weight

        kernels[radius] = kernel
        kernel_extents[radius] = (nx, ny, nz)

    # Cache the results for future use
    result = (kernels, kernel_extents)
    _gaussian_kernel_cache[cache_key] = result

    return kernels, kernel_extents


def gaussian_in_cell(beads_positions, beads_weights, beads_radius,
                    num_points_x, num_points_y, num_points_z, box_lengths):
    """
    Optimized Gaussian density method using precomputed kernels and fast placement.

    This method represents each bead as a sphere with a Gaussian electron density distribution
    using precomputed 3D Gaussian kernels to avoid repeated exponential calculations.
    The standard deviation is set to radius/3 so that 99.7% of the density is contained
    within the bead radius (3-sigma rule).

    Parameters:
    -----------
    beads_positions : np.ndarray
        Bead center positions (N, 3)
    beads_weights : np.ndarray
        Bead electron weights (N,)
    beads_radius : np.ndarray
        Radius of each bead (N,) - used to determine Gaussian width
    num_points_x, num_points_y, num_points_z : int
        Grid dimensions
    box_lengths : np.ndarray
        Box dimensions (3,)

    Returns:
    --------
    electron_density : np.ndarray
        3D density grid with Gaussian contributions
    voxel_edges : list
        Grid edge positions
    """
    Lx, Ly, Lz = box_lengths
    n_beads = beads_positions.shape[0]

    # Pre-calculate grid coordinates and spacing
    x_edges = np.linspace(0, Lx, num_points_x + 1)
    y_edges = np.linspace(0, Ly, num_points_y + 1)
    z_edges = np.linspace(0, Lz, num_points_z + 1)
    voxel_edges = [x_edges, y_edges, z_edges]

    dx = Lx / num_points_x
    dy = Ly / num_points_y
    dz = Lz / num_points_z
    grid_spacings = (dx, dy, dz)

    # Initialize the electron density grid
    electron_density = np.zeros((num_points_x, num_points_y, num_points_z), dtype=np.float64)

    # Find unique radii and precompute kernels
    unique_radii = np.unique(beads_radius)
    print(f"   Precomputing {len(unique_radii)} Gaussian kernels for unique bead radii...")
    kernels, kernel_extents = precompute_gaussian_kernels(unique_radii, grid_spacings)

    # Group beads by radius for efficient processing
    radius_to_beads = {}
    for i, radius in enumerate(beads_radius):
        if radius not in radius_to_beads:
            radius_to_beads[radius] = []
        radius_to_beads[radius].append(i)

    # Process beads grouped by radius
    for radius, bead_indices in radius_to_beads.items():
        kernel = kernels[radius]
        nx, ny, nz = kernel_extents[radius]

        # Process all beads with this radius
        for bead_idx in bead_indices:
            bead_pos = beads_positions[bead_idx]
            bead_weight = beads_weights[bead_idx]

            # Calculate grid indices for bead center
            ix_center = int(np.round(bead_pos[0] / dx))
            iy_center = int(np.round(bead_pos[1] / dy))
            iz_center = int(np.round(bead_pos[2] / dz))

            # Calculate kernel placement bounds
            ix_start = ix_center - nx // 2
            iy_start = iy_center - ny // 2
            iz_start = iz_center - nz // 2


            # Handle periodic boundary conditions and grid bounds
            for kx in range(nx):
                for ky in range(ny):
                    for kz in range(nz):
                        # Calculate target grid position with periodic boundaries
                        ix = (ix_start + kx) % num_points_x
                        iy = (iy_start + ky) % num_points_y
                        iz = (iz_start + kz) % num_points_z

                        # Add scaled kernel contribution
                        electron_density[ix, iy, iz] += bead_weight * kernel[kx, ky, kz]

    print(f"   ✓ Gaussian kernels applied to {n_beads} beads")
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
        Method for density assignment: "default", "cic", "voxelization", "dummy_in_cell", "gaussian_in_cell"

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
        # Spherical voxelization with adaptive sample points per sphere
        electron_density, voxel_edges = voxels_in_cell(beads_positions, beads_weight, beads_radius, 100, num_points_x, num_points_y, num_points_z, Lx, Ly, Lz)
    elif density_method == "dummy_in_cell":
        # Dummy particle method with adaptive number of particles per bead
        electron_density, voxel_edges = dummy_in_cell(
            beads_positions, beads_weight, beads_radius,
            100, num_points_x, num_points_y, num_points_z, box_lengths
        )
    elif density_method == "gaussian_in_cell":
        # Gaussian density method using analytical Gaussian distributions
        electron_density, voxel_edges = gaussian_in_cell(
            beads_positions, beads_weight, beads_radius,
            num_points_x, num_points_y, num_points_z, box_lengths
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
    ts = -1
    output_idx = -1  # Track output index for kept frames

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
                    idx = ts
                    timestep_to_index[timestep] = idx
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
                    box_data.append(float(hi) - float(lo))  # Calculate box length
                # Only store box data for frames we're keeping
                if ts % skip_every_ts == 0:
                    output_idx += 1
                    boxes[output_idx] = box_data
            if "ITEM: ATOMS" in line:
                # extract column headers (not used here but could be validated)
                columns = line.strip().split()[2:]
                data_cols_len = len(columns)
                if "id mol type x y z" in line and data_cols_len >= 6:
                    pass
                else:
                    print(f"Unexpected atom line format at timestep {ts}: {line}")
                    continue

                # Skip atom data if this timestep is not being kept
                if (ts) % skip_every_ts != 0:
                    try:
                        [next(f) for _ in range(N)]  # Skip N lines efficiently
                    except:
                        break
                    continue

                # Read atom data for this timestep
                add_data = True
                data = np.zeros((N, data_cols_len), dtype=float)
                for j in range(N):
                    atom_line = f.readline()
                    if not atom_line:
                        print(f"Unexpected end of file at timestep {ts}, atom {j}")
                        add_data = False
                        break
                    data_tmp = [float(k) for k in atom_line.split()]
                    num_cols = len(data_tmp)
                    if num_cols != data_cols_len:
                        print(f"Incomplete line at timestep {ts} with {num_cols} columns. Skipping.")
                        add_data = False
                        break
                    data[j, :] = data_tmp
                # Store valid data and mark this index as filled
                if add_data:
                    # add zero padding if fewer than 9 columns
                    if data_cols_len < 9:
                        data = np.pad(data, ((0, 0), (0, 9 - data_cols_len)), 'constant')
                    data_list[output_idx, :, :] = data
                    filled_indices.add(output_idx)

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
            result = compute_structure_factor(
                working_positions, working_types, box, q_max, q_min, dq, density_method=density_method
            )
            if result is False:
                print("Error in compute_structure_factor")
                return False, False
            bin_centers, structure_factor = result
        else:
            raise ValueError(f"Accelerator '{accelerator}' not supported for method 'fft'")
    else:
        raise ValueError(f"Unknown method '{method}'")

    return bin_centers, structure_factor
