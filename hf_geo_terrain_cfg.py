# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab.terrains.height_field import HfTerrainBaseCfg
import hf_geo_terrain as hf_geo

#from isaaclab.terrains.height_field.utils import height_field_to_mesh
#from isaaclab.terrains.height_field import hf_terrains

@configclass
class HfGeographicTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a geographic DEM height field terrain."""
    function = hf_geo.geographic_terrain
    """The file path of the Digital Elevation Map (GEOTIF) file"""
    dem_tif_file: str = "NO_DEM_TIF_FILE"
    """The file path of the satellite imagery file (GEOTIF) file"""
    tci_tif_file: str = "NO_TCI_TIF_FILE"
    """The height of the DEM in pixels"""
    #dem_rows: int = -1
    """The width of the DEM in pixels"""
    #dem_cols: int = -1
    """The width of the DEM in pixels"""
    #no_data_val: float = 0.0
    

    
    



