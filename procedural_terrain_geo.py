"""
This script demonstrates procedural terrains with flat patches.

Example usage:

.. code-block:: bash

    # Generate terrain with height color scheme
    ./isaaclab.sh -p scripts/demos/procedural_terrain.py --color_scheme height

    # Generate terrain with random color scheme
    ./isaaclab.sh -p scripts/demos/procedural_terrain.py --color_scheme random

    # Generate terrain with no color scheme
    ./isaaclab.sh -p scripts/demos/procedural_terrain.py --color_scheme none

    # Generate terrain with curriculum
    ./isaaclab.sh -p scripts/demos/procedural_terrain.py --use_curriculum

    # Generate terrain with curriculum along with flat patches
    ./isaaclab.sh -p scripts/demos/procedural_terrain.py --use_curriculum --show_flat_patches

"""
import argparse
import numpy as np
import math
import random
import torch
import os.path
from scipy.spatial.transform import Rotation
import rasterio
from pyproj import Transformer
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="This script demonstrates procedural terrain generation.")
parser.add_argument(
    "--color_scheme",
    type=str,
    default="none",
    choices=["height", "random", "none"],
    help="Color scheme to use for the terrain generation.",
)
parser.add_argument(
    "--use_curriculum",
    action="store_true",
    default=False,
    help="Whether to use the curriculum for the terrain generation.",
)
parser.add_argument(
    "--show_flat_patches",
    action="store_true",
    default=False,
    help="Whether to show the flat patches computed during the terrain generation.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
#Rest everything follows running live:
import carb
import omni.kit.app
import omni.ext
import omni.timeline
from omni.isaac.core.world import World
from isaacsim.core.utils.viewports import set_camera_view
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBase
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.terrains as terrain
from hf_geo_terrain_cfg import HfGeographicTerrainCfg
from hf_geo_terrain import geographic_terrain, wavelet_terrain
# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
"""            flat_patch_sampling={"my_flat_patch_smp_cfg":
                terrain.FlatPatchSamplingCfg(
                    num_patches=0,
                    patch_radius=1.0,
                    x_range=(-1000000.0, 1000000.0),
                    y_range=(-1000000.0, 1000000.0),
                    z_range=(-1000000.0, 1000000.0),
                    max_height_diff=0.05
                )
            },
"""

def read_raster_info(dem_geotif_file_path:str, vertical_scale:float, dbg:int|bool=0):
    #gdal_translate -of GTiff \
    # -co COMPRESS=DEFLATE \
    # -co PREDICTOR=2 -co \
    # ZLEVEL=9 -outsize 2500 2500 \
    # /home/austin/Documents/src/UXV/PX4_Playground/ros_ws/GIS/tiles/DEM_MERGED.tif \
    # /home/austin/Documents/src/UXV/PX4_Playground/ros_ws/GIS/tiles/DEM_MERGED_SQUARE.tif
    # Verify/access the data (e.g., print metadata and sample elevation)
    print(f"Loading GeoTif Digital Elevation Map (DEM): {dem_geotif_file_path}")
    with rasterio.open(dem_geotif_file_path) as src:
        dem_tran = Transformer.from_crs(src.crs, 'EPSG:3857', always_xy=True)
        dem_raw = src.read(1)  # Read as 2D NumPy array (elevation in meters)
        crs_bl = (src.bounds.left,  src.bounds.bottom)
        crs_tr = (src.bounds.right, src.bounds.top   )
        utm_bl = dem_tran.transform(crs_bl[0],  crs_bl[1])
        utm_tr = dem_tran.transform(crs_tr[0],  crs_tr[1])
        crs_dx = crs_tr[0] - crs_bl[0]
        crs_dy = crs_tr[1] - crs_bl[1]
        utm_dx = abs(utm_tr[0] - utm_bl[0])
        utm_dy = abs(utm_tr[1] - utm_bl[1])
        px_dx = abs(utm_dx / dem_raw.shape[0])
        py_dx = abs(utm_dy / dem_raw.shape[1])
        px_sz = int(dem_raw.shape[0])
        py_sz = int(dem_raw.shape[1])
        hf_scale = (1.0 / vertical_scale)
        stat_min = np.min(dem_raw)     # Used as sim floor = 0.0
        stat_avg = np.average(dem_raw) # Used as No Data value
        stat_max = np.max(dem_raw)
        info = {}
        info['geo_cen'] = (crs_bl[0] + (crs_dx/2.0), crs_bl[1] + (crs_dy/2.0))
        info['crs_res'] = src.res
        info['crs_size'] = (crs_dx, crs_dy)
        info['pix_size'] = (px_sz, py_sz)
        info['geo_size'] = (utm_dx, utm_dy) # TODO: INVESTIGATE non square aspect ratios
        info['geo_res'] = (px_dx, py_dx)
        info['bounds'] = src.bounds
        info['min_alt'] = stat_min
        info['avg_alt'] = stat_avg
        info['max_alt'] = stat_max
        info['cen_alt'] = dem_raw[int(px_sz/2)][int(py_sz/2)]
        info['crs'] = src.crs 
        info['z_step'] = hf_scale 
        info['z_max'] = math.pow(2,16)-1
        if(dbg):
            print(f"Resolution (CRS) | (lon, lat): {src.res}")
            print(f"Resolution (pixels): {px_sz} x {py_sz} ")
            print(f"Resolution (meters): {px_dx} x {py_dx} ")
            print(f"Geographic {src.bounds}")
            print(f"CRS Size: {crs_dx} m x {crs_dy} m")
            print(f"Geo Size: {utm_dx} m x {utm_dy} m")
            print(f"Geo Cent: ({info['geo_cen'][0]}, {info['geo_cen'][1]}) | (lon, lat)")
            print(f"Height Min: {stat_min} m")
            print(f"Height Avg: {stat_avg} m")
            print(f"Height Max: {stat_max} m")
            print(f"CRS: {src.crs}")
            print(f"Sampled elevation at center: {stat_avg} m (used as no_data value)")
            print(f"Quantized Height Feild Resolution: {hf_scale} (per meter)")
            print(f"Z-Axis Max Output Range(0, {vertical_scale*(math.pow(2,16)-1):.2f})")
        return info

dem_file_path = "in/DEM_WARP_SQUARE.tif"
ri = read_raster_info(dem_file_path, v_scale, dbg=1)
spawn_dz = 4.2
v_scale = 0.05
h_scale      = (ri['geo_res'][0] / 2.0)                             # TODO: Figure out why scale is off by 2
terrain_size = (ri['geo_size'][0] / 2.0, ri['geo_size'][1] / 2.0)   # TODO: Figure out why scale is off by 2
# averag of x and y geographic step per pixel in meters
MY_TERRAIN_CFG = terrain.TerrainGeneratorCfg(
    size=terrain_size,
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=h_scale,
    vertical_scale=v_scale,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "hf_terrain_0": HfGeographicTerrainCfg(
            function=geographic_terrain, #wavelet_terrain,  
            proportion=1.0,
            size=terrain_size,
            flat_patch_sampling=None,
            border_width=0.0,
            horizontal_scale=h_scale,
            vertical_scale=v_scale,
            slope_threshold=None,
            dem_tif_file=dem_file_path,
            tci_tif_file="in/DEM_WARP_SQUARE.tif",
        ),
    },
)

def enable_pegasus_ext():
    pegasus_ext_path = "/home/austin/Documents/src/UXV/toolkit_install/PegasusSimulator/extensions"
    settings = carb.settings.get_settings()
    ext_folders_key = "/exts/omni.kit.window.extensions/extFolders"
    # Add PegasusSimulator/extensions path to live running Isaac Sim
    ext_folders = settings.get_as_string(ext_folders_key) or []
    if pegasus_ext_path not in ext_folders:
        ext_folders.append(pegasus_ext_path)
        settings.set(ext_folders_key, ext_folders)
    # Enable the Pegasus extension
    # Use set_extension_enabled_immediate for immediate effect without restart
    app_interface = omni.kit.app.get_app_interface()
    ext_manager = app_interface.get_extension_manager()
    ext_manager.set_extension_enabled_immediate("pegasus.simulator", True) #"pegasus.simulator" = { version = "4.5.0" }
    simulation_app.update()

def design_scene() -> tuple[dict, torch.Tensor]:
    """Designs the scene."""
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Parse terrain generation
    terrain_gen_cfg = MY_TERRAIN_CFG.replace(curriculum=args_cli.use_curriculum, color_scheme=args_cli.color_scheme)
    '''
    # Add flat patch configuration
    # Note: To have separate colors for each sub-terrain type, we set the flat patch sampling configuration name
    #   to the sub-terrain name. However, this is not how it should be used in practice. The key name should be
    #   the intention of the flat patch. For instance, "source" or "target" for spawn and command related flat patches.
    '''
    if args_cli.show_flat_patches:
        for sub_terrain_name, sub_terrain_cfg in terrain_gen_cfg.sub_terrains.items():
            sub_terrain_cfg.flat_patch_sampling = {
                sub_terrain_name: terrain.FlatPatchSamplingCfg(num_patches=10, patch_radius=0.5, max_height_diff=0.05)
            }

    # Handler for terrains importing
    terrain_importer_cfg = terrain.TerrainImporterCfg(
        num_envs=1,
        env_spacing=None,
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type="generator",
        terrain_generator=terrain_gen_cfg,
        debug_vis=True
    )
    # Remove visual material for height and random color schemes to use the default material
    if args_cli.color_scheme in ["height", "random"]:
        terrain_importer_cfg.visual_material = None
    # Create terrain importer
    terrain_importer = terrain.TerrainImporter(terrain_importer_cfg)

    # Show the flat patches computed
    if args_cli.show_flat_patches:
        # Configure the flat patches
        vis_cfg = VisualizationMarkersCfg(prim_path="/Visuals/TerrainFlatPatches", markers={})
        for name in terrain_importer.flat_patches:
            vis_cfg.markers[name] = sim_utils.CylinderCfg(
                radius=0.5,  # note: manually set to the patch radius for visualization
                height=0.1,
                visual_material=sim_utils.GlassMdlCfg(glass_color=(random.random(), random.random(), random.random())),
            )
        flat_patches_visualizer = VisualizationMarkers(vis_cfg)

        # Visualize the flat patches
        all_patch_locations = []
        all_patch_indices = []
        for i, patch_locations in enumerate(terrain_importer.flat_patches.values()):
            num_patch_locations = patch_locations.view(-1, 3).shape[0]
            # store the patch locations and indices
            all_patch_locations.append(patch_locations.view(-1, 3))
            all_patch_indices += [i] * num_patch_locations
        # combine the patch locations and indices
        flat_patches_visualizer.visualize(torch.cat(all_patch_locations), marker_indices=all_patch_indices)

    # return the scene information
    scene_entities = {"terrain": terrain_importer}
    return scene_entities, terrain_importer.env_origins


# def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, AssetBase] | None, origins: torch.Tensor | None):
#     """Runs the simulation loop."""
#     # Simulate physics
#     while simulation_app.is_running():
#         # perform step
#         sim.step()

class PegasusApp:
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """
    def __init__(self):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """
        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = omni.timeline.get_timeline_interface()
        # Start the Pegasus Interface
        self.pg = PegasusInterface()
        # self.pg.set_new_default_global_coordinates(ri['geo_cen'][0], ri['geo_cen'][1], ri['cen_alt'] + spawn_dz)
        # Acquire the World, .i.e, the singleton that controls that is a one stop shop for setting up physics, 
        # spawning asset primitives, etc.
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        # Launch one of the worlds provided by NVIDIA
        # self.pg.load_environment(SIMULATION_ENVIRONMENTS["Default Environment"])
        
        

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """
        # Start the simulation
        self.timeline.play()
        # The "infinite" loop
        while simulation_app.is_running() and not self.stop_sim:
            # Update the UI of the app and perform the physics step
            self.world.step(render=True)
        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

def main():
    #enable_pegasus_ext()

    # design scene
    #scene_entities = None
    #scene_origins = None
    # Play the simulator
    # sim.reset()
    # Now we are ready!
    
    # Instantiate the template app
    pg_app = PegasusApp()

    """Main function."""
    set_camera_view(eye=[3, 3, spawn_dz+3], target=[0, 0, spawn_dz])
    
    scene_entities, scene_origins = design_scene()
    
    # Create the vehicle
    # Try to spawn the selected robot in the world to the specified namespace
    config_multirotor = MultirotorConfig()
    # Create the multirotor configuration
    mavlink_config = PX4MavlinkBackendConfig({
        "vehicle_id": 0,
        "px4_autolaunch": True,
        "px4_dir": pg_app.pg.px4_path,
        "px4_vehicle_model": pg_app.pg.px4_default_airframe # CHANGE this line to 'iris' if using PX4 version bellow v1.14
    })
    config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]
    Multirotor(
        "/World/quadrotor",
        ROBOTS['Iris'],
        0,
        [0.0, 0.0, spawn_dz],
        Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
        config=config_multirotor,
    )
    # Reset the simulation environment so that all articulations (aka robots) are initialized
    pg_app.world.reset()
    # Auxiliar variable for the timeline callback example
    pg_app.stop_sim = False
    print("[INFO]: Setup complete...")
    # Run the application loop
    pg_app.run()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
