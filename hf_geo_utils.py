import math
import numpy as np
import rasterio
from pyproj import Transformer

def read_raster_info(dem_geotif_file_path:str, vertical_scale:float=0.05, dbg:int|bool=0):
    #gdalwarp -s_srs EPSG:3857 -t_srs EPSG:3857 -te 11520966.628110012 6956892.510042363 11539205.859441490 6975131.741373841 -ts 1107 1107 -r bilinear -of GTiff -co COMPRESS=DEFLATE DEM_SRC_PWM.tif ../../../../toolkit_install/IsaacLab/scripts/geo_dev/in/DEM_clipped_11km_square.tif
    if dbg: print(f"read_raster_info(): {dem_geotif_file_path}")
    with rasterio.open(dem_geotif_file_path) as src:
        utm_tran = Transformer.from_crs(src.crs, 'EPSG:32648', always_xy=True)
        geo_tran = Transformer.from_crs('EPSG:32648', 'EPSG:4326')
        dem_raw = src.read(1)  # Read as 2D NumPy array (elevation in meters)
        crs_bl = (src.bounds.left,  src.bounds.bottom)
        crs_tr = (src.bounds.right, src.bounds.top   )
        utm_bl = utm_tran.transform(crs_bl[0],  crs_bl[1])
        utm_tr = utm_tran.transform(crs_tr[0],  crs_tr[1])
        crs_dx = crs_tr[0] - crs_bl[0]
        crs_dy = crs_tr[1] - crs_bl[1]
        utm_dx = abs(utm_tr[0] - utm_bl[0])    # width of utm region in meters
        utm_dy = abs(utm_tr[1] - utm_bl[1])    # height of utm region in meters
        px_dx = abs(utm_dx / dem_raw.shape[0]) # width of pixel in meters
        py_dx = abs(utm_dy / dem_raw.shape[1]) # height of pixel in meters
        px_sz = int(dem_raw.shape[0])
        py_sz = int(dem_raw.shape[1])
        hf_scale = (1.0 / vertical_scale)
        stat_min = np.min(dem_raw)     # Used as sim floor = 0.0
        stat_avg = np.average(dem_raw) # Used as No Data value
        stat_max = np.max(dem_raw)
        # TODO: Checking if a -(px_dx/2.0) origin offset can improve GPS alignment in QGroundControl
        utm_cen_x = utm_bl[0] + (utm_dx/2.0) + (px_dx/1.0)
        utm_cen_y = utm_bl[1] + (utm_dy/2.0) + (py_dx/1.0)
        geo_cen = geo_tran.transform(utm_cen_x, utm_cen_y) 
        info = {}
        info['utm_cen'] = (utm_cen_x, utm_cen_y) # tuple (easting, northing)
        info['geo_cen'] = geo_cen                # tuple (lat, lon)
        info['crs_res'] = src.res
        info['crs_size'] = (crs_dx, crs_dy)
        info['pix_size'] = (px_sz, py_sz)
        info['geo_size'] = (utm_dx, utm_dy) # TODO: INVESTIGATE non square aspect ratios
        info['geo_res'] = (px_dx, py_dx)
        info['bounds'] = src.bounds
        info['min_alt'] = stat_min
        info['avg_alt'] = stat_avg
        info['max_alt'] = stat_max
        info['cen_alt'] = float( dem_raw[int(px_sz/2)][int(py_sz/2)] )
        info['crs'] = src.crs 
        info['z_step'] = hf_scale 
        info['z_max'] = vertical_scale * (math.pow(2,16)-1)
        if(dbg):
            print(f"Resolution          (CRS): {info['crs_res']}")
            print(f"Resolution       (pixels): {info['pix_size'][0]} x {info['pix_size'][1]} ")
            print(f"Resolution (meters/pixel): {info['geo_res'][0]} x {info['geo_res'][1]} ")
            print(f"Geographic {info['bounds']}")
            print(f"CRS Size: {info['crs_size'][0]} m x {info['crs_size'][1]} m")
            print(f"Geo Size: {info['geo_size'][0]} m x {info['geo_size'][1]} m")
            print(f"UTM Cent | (East, North): ({info['utm_cen'][0]}, {info['utm_cen'][1]})")
            print(f"Geo Cent |    (lon, lat): ({info['geo_cen'][1]}, {info['geo_cen'][0]})")
            print(f"Height Min: {info['min_alt']} m")
            print(f"Height Avg: {info['avg_alt']} m")
            print(f"Height Max: {info['max_alt']} m")
            print(f"CRS: {info['crs']}")
            print(f"Sampled elevation at center: {info['cen_alt']} m (used as no_data value)")
            print(f"Quantized Height Feild Resolution: {info['z_step']} (per meter)")
            print(f"Z-Axis Max Output Range(0, {info['z_max']:.2f})")
        return info