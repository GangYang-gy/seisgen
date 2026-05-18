# -------------------------------------------------------------------
# Optimized Tools to manage the pre-computed grids in the 3D background model.
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# Optimized: Gang (2025) using cKDTree
# -------------------------------------------------------------------

import h5py
import numpy as np
from scipy.spatial import cKDTree

POINT_KEYS = ["latitude",
             "longitude",
             "z",
             "depth",
             "utm_x",
             "utm_y",
             "utm_z",
             "slice_index",
             "element_index",
             "xi",
             "eta",
             "gamma"]

class DPointCloud:
    """
    Point-cloud manager for pre-computed grid information in the 3-D model.

    The HDF5 file is expected to contain all datasets listed in POINT_KEYS.
    This optimized version uses cKDTree for fast nearest-neighbor queries
    when depth is used as the vertical coordinate.
    """

    def __init__(self, file_path):
        self.b_pointcloud_initial = False
        if file_path is None:
            return

        try:
            with h5py.File(file_path, 'r') as f:
                self.mesh_lat           = f[POINT_KEYS[0]][:]
                self.mesh_long          = f[POINT_KEYS[1]][:]
                self.mesh_z             = f[POINT_KEYS[2]][:]    # Elevation in meters
                self.mesh_depth         = f[POINT_KEYS[3]][:]    # Depth in meters
                self.mesh_utm_x         = f[POINT_KEYS[4]][:]
                self.mesh_utm_y         = f[POINT_KEYS[5]][:]
                self.mesh_utm_z         = f[POINT_KEYS[6]][:]
                self.mesh_slice_index   = f[POINT_KEYS[7]][:]
                self.mesh_element_index = f[POINT_KEYS[8]][:]
                self.mesh_xi            = f[POINT_KEYS[9]][:]
                self.mesh_eta           = f[POINT_KEYS[10]][:]
                self.mesh_gamma         = f[POINT_KEYS[11]][:]

            # Build a KDTree for UTM coordinates using depth as the vertical coordinate.
            utm_points = np.vstack([self.mesh_utm_x, self.mesh_utm_y, self.mesh_depth]).T
            self.tree_utm = cKDTree(utm_points)

            # Build a KDTree for latitude/longitude coordinates using depth.
            latlong_points = np.vstack([111000.0*self.mesh_lat, 111000.0*self.mesh_long, self.mesh_depth]).T
            self.tree_latlong = cKDTree(latlong_points)

            self.n_grid = len(self.mesh_lat)
            self.b_pointcloud_initial = True
            
        except Exception as e:
            print("!!! Point cloud not found or failed to load")
            raise e

    def _check(self):
        if not self.b_pointcloud_initial:
            raise RuntimeError("!!! Point cloud not initialized")

    def find(self, x, y, z, n=1, mode='LATLONGZ', b_depth=True):
        """
        Determine the closest N points to (x, y, z) in the point cloud and return the information.

        :param x:       Either latitude or UTM X
        :param y:       Either longitude or UTM Y
        :param z:       Depth or elevation in meter
        :param n:       Number of nearest neighbors to return
        :param mode:    'LATLONGZ' or 'UTM'
        :param b_depth: True->use depth, False->use elevation
        :return:        Tuple of arrays corresponding to POINT_KEYS
        """
        self._check()
        n = int(n)
        if n <= 0:
            raise ValueError("n must be a positive integer.")

        # Record whether the user requests only one nearest point.
        return_scalar = (n == 1)

        # Avoid invalid indices when n is larger than the total number of grid points.
        n = min(n, self.n_grid)

        if mode.upper() == 'LATLONGZ':
            if not b_depth:
                # use elevation, compute distances by vectorized operations.
                dist_H = 111000.0 * np.sqrt((self.mesh_lat - x)**2 + (self.mesh_long - y)**2)
                dist = np.sqrt(dist_H**2 + (self.mesh_z - z)**2)

                if n > 1:
                    idx = np.argpartition(dist, n - 1)[:n]
                else:
                    idx = np.array([np.argmin(dist)])
   
            else:
                # Use depth. Query the cKDTree.
                query_point = [111000.0*x, 111000.0*y, z]
                _, idx = self.tree_latlong.query(query_point, k=n)
                idx = np.atleast_1d(idx).astype(int)
                dist_H = 111000.0 * np.sqrt(
                    (self.mesh_lat - x) ** 2 +
                    (self.mesh_long - y) ** 2
                )
                dist = np.sqrt(dist_H ** 2 + (self.mesh_depth - z) ** 2)

        elif mode.upper() == 'UTM':
            if not b_depth:
                # Use elevation. Compute distances by vectorized operations.
                dist = np.sqrt((self.mesh_utm_x - x)**2 + (self.mesh_utm_y - y)**2 + (self.mesh_z - z)**2)
                if n > 1:
                    idx = np.argpartition(dist, n - 1)[:n]
                else:
                    idx = np.array([np.argmin(dist)])

            else:
                # Use depth. Query the cKDTree.
                query_point = [x, y, z]
                _, idx = self.tree_utm.query(query_point, k=n)
                idx = np.atleast_1d(idx).astype(int)
                dist = np.sqrt(
                    (self.mesh_utm_x - x) ** 2 +
                    (self.mesh_utm_y - y) ** 2 +
                    (self.mesh_depth - z) ** 2
                )

        else:
            raise NotImplementedError("!!! Undefined mode!")

        # Make idx always an integer array.
        idx = np.atleast_1d(idx).astype(int)

        # Sort the selected points from nearest to farthest.
        idx = idx[np.argsort(dist[idx])]

        # If n == 1, return scalar values instead of one-element arrays.
        if return_scalar:
            idx = int(idx[0])

        return (self.mesh_lat[idx], self.mesh_long[idx], self.mesh_z[idx], self.mesh_depth[idx],
                self.mesh_utm_x[idx], self.mesh_utm_y[idx], self.mesh_utm_z[idx],
                self.mesh_slice_index[idx], self.mesh_element_index[idx],
                self.mesh_xi[idx], self.mesh_eta[idx], self.mesh_gamma[idx])
