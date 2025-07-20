# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate height fields for different terrains."""

from __future__ import annotations
import math
import numpy as np
import scipy.interpolate as interpolate
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.terrains.height_field.utils import height_field_to_mesh

import isaacsim.core.utils.prims as prim_utils

import rasterio
from pyproj import Transformer

if TYPE_CHECKING:
    from .hf_geo_terrain_cfg import HfGeographicTerrainCfg

@height_field_to_mesh
def geographic_terrain(difficulty: float, cfg: HfGeographicTerrainCfg) -> np.ndarray:
    """Generate a terrain with height sampled from a Digital Elevation Map (DEM). In GEOTIF formaat

    #.. image:: ../../_static/terrains/height_field/random_uniform_terrain.jpg
    #   :width: 40%
    #   :align: center

    Note:
        The :obj:`difficulty` parameter is ignored for this terrain.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    Raises:
        ValueError: N/A
    """
    # Verify/access the data (e.g., print metadata and sample elevation)
    print(f"Loading GeoTif Digital Elevation Map (DEM): {cfg.dem_tif_file}")
    with rasterio.open(cfg.dem_tif_file) as src:
        dem_tran = Transformer.from_crs(src.crs, 'EPSG:3857', always_xy=True)
        dem_raw = src.read(1)  # Read as 2D NumPy array (elevation in meters)

        utm_bl = dem_tran.transform(src.bounds.left,  src.bounds.bottom)
        utm_tr = dem_tran.transform(src.bounds.right, src.bounds.top   )
        utm_dx = utm_tr[0] - utm_bl[0]
        utm_dy = utm_tr[1] - utm_bl[1]
        px_dx = utm_dx / dem_raw.shape[0]
        py_dx = utm_dy / dem_raw.shape[1]
        
        px_sz = dem_raw.shape[0]
        py_sz = dem_raw.shape[1]
        
        hf_scale = (1.0 / cfg.vertical_scale)

        stat_min = np.min(dem_raw)     # Used as sim floor = 0.0
        stat_avg = np.average(dem_raw) # Used as No Data value
        stat_max = np.max(dem_raw)
        print(f"Resolution (CRS) | (lon, lat): {src.res}")
        print(f"Resolution (pixels): {px_sz} x {py_sz} ")
        print(f"Resolution (meters): {px_dx} x {py_dx} ")
        print(f"Geographic {src.bounds}")
        print(f"Geographic Size: {utm_dx} m x {utm_dy} m")
        print(f"Height Min: {stat_min} m")
        print(f"Height Avg: {stat_avg} m")
        print(f"Height Max: {stat_max} m")
        print(f"CRS: {src.crs}")
        print(f"Sampled elevation at center: {stat_avg} m (used as no_data value)")
        print(f"Quantized Height Feild Resolution: {hf_scale} (per meter)")
        print(f"Z-Axis Max Output Range(0, {cfg.vertical_scale*(math.pow(2,16)-1):.2f})")
        # TODO: testing to see if output height field requires odd resolutions
        if(px_sz % 2 == 0):
            px_sz -= 1
        if(py_sz % 2 == 0):
            py_sz -= 1
        hf_raw = np.full((px_sz, py_sz), (stat_avg*hf_scale) )
        for x in range(0, px_sz):
            for y in range(0, py_sz):
                hf_raw[x][y] = (dem_raw[x][y] - stat_min) * hf_scale
    
    # round off the interpolated heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)

# Parametric Wavlet terrain
@height_field_to_mesh
def wavelet_terrain(difficulty: float, cfg: HfGeographicTerrainCfg) -> np.ndarray:
    """Generate a terrain with height sampled from a Digital Elevation Map (DEM). In GEOTIF formaat

    #.. image:: ../../_static/terrains/height_field/random_uniform_terrain.jpg
    #   :width: 40%
    #   :align: center

    Note:
        The :obj:`difficulty` parameter is ignored for this terrain.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    Raises:
        ValueError: N/A
    """
    
    # switch parameters to discrete units
    # -- horizontal scale
    col_sz = int(cfg.size[0] / cfg.horizontal_scale)
    row_sz = int(cfg.size[1] / cfg.horizontal_scale)
    v_scale = (1.0 / cfg.vertical_scale)
    print(f"Resolution Pixels: {col_sz} x {row_sz} ")
    print(f"Quantized Height Feild Resolution: {v_scale} (per meter)")
    print(f"Z-Axis Max Output Range(0, {cfg.vertical_scale*(math.pow(2,16)-1):.2f})")
    # create a flat terrain at [cfg.no_data_val] meters in elevation  
    hf_raw = np.full((col_sz, row_sz), 0)
    kf = (4.0 / cfg.size[0])
    af = cfg.size[0] / 6.0
    xhalf = int(col_sz >> 1)
    yhalf = int(row_sz >> 1)
    for x in range(col_sz):
        for y in range(row_sz):
            fx = (x - xhalf) * cfg.horizontal_scale
            fy = (y - yhalf) * cfg.horizontal_scale
            hf_raw[x][y] = v_scale * af * math.exp(-( math.pow(kf * fx, 2) + math.pow(kf * fy, 2) ))
    
    xm = int(col_sz >> 1)
    ym = int(row_sz >> 1)
    center_height = hf_raw[xm][ym]
    print(f"\n\n    center height value is: {center_height}\n\n")

    # round off the interpolated heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)
