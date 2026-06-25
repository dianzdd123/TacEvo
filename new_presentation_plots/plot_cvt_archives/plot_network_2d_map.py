#| This file is a part of the pymap_elites framework.
#| Copyright 2019, INRIA
#| Main contributor(s):
#| Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
#| Eloise Dalin , eloise.dalin@inria.fr
#| Pierre Desreumaux , pierre.desreumaux@inria.fr
#|
#|
#| **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
#| mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
#|
#| This software is governed by the CeCILL license under French law
#| and abiding by the rules of distribution of free software.  You
#| can use, modify and/ or redistribute the software under the terms
#| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
#| following URL "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and rights to
#| copy, modify and redistribute granted by the license, users are
#| provided only with a limited warranty and the software's author,
#| the holder of the economic rights, and the successive licensors
#| have only limited liability.
#|
#| In this respect, the user's attention is drawn to the risks
#| associated with loading, using, modifying and/or developing or
#| reproducing the software by the user in light of its specific
#| status of free software, that may mean that it is complicated to
#| manipulate, and that also therefore means that it is reserved for
#| developers and experienced professionals having in-depth computer
#| knowledge. Users are therefore encouraged to load and test the
#| software's suitability as regards their requirements in conditions
#| enabling the security of their systems and/or data to be ensured
#| and, more generally, to use and operate it in the same conditions
#| as regards security.
#|
#| The fact that you are presently reading this means that you have
#| had knowledge of the CeCILL license and that you accept its terms.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import Voronoi
import sys
from sklearn.neighbors import KDTree
import matplotlib.cm as cm

# Set the colormap for the plot which is reversed so that low fitness values are lighter
my_cmap = cm.viridis_r

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Source: https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram/20678647#20678647
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def load_data(filename, dim_behav):
    """ Load data from archive file, ignoring the first column (reference). """
    print("Loading", filename)
    
    # Define which columns to load. We skip the first column (index 0) which is a string path.
    # We load the fitness (index 1) and the behavior descriptors (e.g., index 2, 3).
    cols_to_load = tuple(range(1, 2 + dim_behav))
    
    # Use the 'usecols' parameter to only load the specified numerical columns
    data = np.loadtxt(filename, usecols=cols_to_load)
    
    # Now that 'data' only contains the numerical values, we can slice it.
    # The first column of 'data' is our fitness.
    fit = data[:, 0:1]
    # The remaining columns are the behavior descriptors.
    desc = data[:, 1:1 + dim_behav]
    
    return fit, desc

def load_centroids(filename):
    """ Load centroids from a file """
    points = np.loadtxt(filename)
    return points

def plot_cvt(ax, centroids, fit, desc, min_fit, max_fit):
    """ Plot the CVT diagram based on the archive data """
    # Compute Voronoi tesselation
    print("Creating Voronoi diagram")
    vor = Voronoi(centroids[:, 0:2])
    regions, vertices = voronoi_finite_polygons_2d(vor)
    
    print(f"Using fitness range for color map: min={min_fit:.2f}, max={max_fit:.2f}")
    # Clip fitness values to the range to handle outliers
    fit_clipped = np.clip(fit, min_fit, max_fit)
    norm = matplotlib.colors.Normalize(vmin=min_fit, vmax=max_fit, clip=True)
    
    print("Building KD-Tree for nearest neighbor search")
    kdt = KDTree(centroids, leaf_size=30, metric='euclidean')

    # Draw the empty Voronoi cells (background)
    print("Drawing background grid")
    for region in regions:
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.1, edgecolor='black', facecolor='white', lw=1)

    print("Coloring cells based on archive data")
    # Iterate through each solution in the archive to color the corresponding cell
    for i in range(len(desc)):
        # Find the closest centroid to the current solution's behavior descriptor
        q = kdt.query([desc[i]], k=1)
        index = q[1][0][0]
        
        # Get the polygon for that centroid's cell
        region = regions[index]
        polygon = vertices[region]
        
        # Fill the polygon with a color corresponding to the clipped fitness
        ax.fill(*zip(*polygon), alpha=0.9, color=my_cmap(norm(fit_clipped[i])))
        
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(desc)} processed", end="\r", flush=True)
    
    print("\nPlotting scatter of solutions")
    # Overlay a scatter plot of the actual solutions using clipped fitness values
    fit_clipped_reshaped = fit_clipped.flatten()
    ax.scatter(desc[:, 0], desc[:, 1], c=fit_clipped_reshaped, cmap=my_cmap, norm=norm, s=10, zorder=2, edgecolor='black', linewidth=0.2)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(f'Usage: {sys.argv[0]} centroids_file.dat archive_file.dat [min_fit] [max_fit]')

    #  Data Loading 
    centroids_file = sys.argv[1]
    archive_file = sys.argv[2]
    
    # Check if the data files exist before trying to load them
    # Note: This is a basic check. In a real application, more robust error handling would be good.
    try:
        with open(centroids_file, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"Error: Centroids file not found at '{centroids_file}'")
        sys.exit(1)
        
    try:
        with open(archive_file, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"Error: Archive file not found at '{archive_file}'")
        sys.exit(1)

    centroids = load_centroids(centroids_file)
    dim_behav = centroids.shape[1]
    fit, beh = load_data(archive_file, dim_behav)

    #  Plotting Setup 
    # Determine the min and max fitness for the color scale
    if len(sys.argv) > 4:
        min_fit = float(sys.argv[3])
        max_fit = float(sys.argv[4])
    else:
        # Use percentiles to handle outliers and maintain gradient for majority of values
        # 95th percentile as upper bound to cap extreme outliers
        # 5th percentile as lower bound to handle any extreme low values
        if len(fit) > 0:
            min_fit = np.percentile(fit, 5)  # 5th percentile
            max_fit = np.percentile(fit, 95)  # 95th percentile
            
            # Ensure we don't go below the actual minimum if it's reasonable
            actual_min = np.min(fit)
            if actual_min < min_fit * 0.8:  # If actual min is significantly below 5th percentile
                min_fit = actual_min
            
            print(f"Using percentile-based range: min={min_fit:.4f} (5th percentile), max={max_fit:.4f} (95th percentile)")
            print(f"Actual data range: min={np.min(fit):.4f}, max={np.max(fit):.4f}")
        else:
            min_fit = 0
            max_fit = 1

    # Create the plot figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("MAP-Elites Force Network Archive", fontsize=28, fontweight='bold')
    ax.set_xlabel("Architecture Diversity", fontsize=22)
    ax.set_ylabel("Efficiency Ratio", fontsize=22)

    # Set larger font sizes for axis tick labels
    ax.tick_params(axis='both', which='major', labelsize=22)

    # Optional: Set axis limits based on centroids to frame the plot nicely
    ax.set_xlim(np.min(centroids[:, 0]) - 0.05, np.max(centroids[:, 0]) + 0.05)
    ax.set_ylim(np.min(centroids[:, 1]) - 0.05, np.max(centroids[:, 1]) + 0.05)
    ax.set_aspect('equal', adjustable='box')
    
    #  Generate Plot 
    plot_cvt(ax, centroids, fit, beh, min_fit, max_fit)

    # Add a color bar to the plot
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=min_fit, vmax=max_fit))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Fitness', fontsize=22)
    cbar.ax.tick_params(labelsize=18)

    #  Save Figure 
    output_filename_base = 'force_net_cvt_map'
    fig.savefig(f'{output_filename_base}.png', bbox_inches='tight')
    print(f"\nPlot saved as '{output_filename_base}.png'")