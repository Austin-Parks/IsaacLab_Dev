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
import yaml
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
#from isaacsim import SimulationApp
import omni.kit.commands
from pxr import UsdShade, Sdf
from isaacsim.core.utils.viewports import set_camera_view
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBase
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.terrains as terrain
#from isaaclab.sim import M
from hf_geo_utils import *
from hf_geo_terrain_cfg import HfGeographicTerrainCfg
from hf_geo_terrain import geographic_terrain, read_raster_info, wavelet_terrain


########################################################################################################################

#def enable_pegasus_ext():
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
print("[Pegasus Simulator Extension enabled]")
#enable_pegasus_ext()

#def get_my_terrain_cfg() -> terrain.TerrainGeneratorCfg:
tci_file_path = "in/TCI_clipped_15km_square.png"
dem_file_path = "in/DEM_clipped_15km_square.tif"
v_scale = 0.05
ri = read_raster_info(dem_file_path, v_scale, dbg=1)
avg_geo_res  = ((ri['geo_res' ][0] + ri['geo_res' ][1]) / 2.0) #/ 1.68 # TODO: Why are we off by random factor here???
avg_geo_size = ((ri['geo_size'][0] + ri['geo_size'][1]) / 2.0) #/ 1.68 # TODO: Why are we off by random factor here???
h_scale      = avg_geo_res
terrain_size = (avg_geo_size, avg_geo_size)
print(f"         h_scale: {h_scale}")
print(f"    terrain size: {terrain_size}\n")
my_terrain_cfg = terrain.TerrainGeneratorCfg(
    size=terrain_size,
    border_width=0,
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
            slope_threshold=0.75,
            dem_tif_file=dem_file_path,
            tci_tif_file=tci_file_path,
        ),
    },
)
pegasus_cfg_yaml = "/home/austin/Documents/src/UXV/toolkit_install/PegasusSimulator/extensions/pegasus.simulator/config/configs.yaml"
with open(pegasus_cfg_yaml) as f:
    data = yaml.safe_load(f)
    print(f"input Pegasus Simulator cfg data:\n{data}")
    data['global_coordinates']['altitude']  = ri['cen_alt']
    data['global_coordinates']['longitude'] = ri['geo_cen'][1]
    data['global_coordinates']['latitude']  = ri['geo_cen'][0]
    print(f"writing new Pegasus Simulator cfg values to file:\n{data}")
    with open(pegasus_cfg_yaml, "w") as f:
        yaml.dump(data, f)
#return my_terrain_cfg.replace(curriculum=args_cli.use_curriculum, color_scheme=args_cli.color_scheme)

# Import the Pegasus API for simulating drones
from omni.isaac.core.world import World
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

#from isaacsim.core.api.world import World
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


def design_scene() -> tuple[dict, torch.Tensor]:      # terrain_gen_cfg:terrain.TerrainGeneratorCfg
    """Designs the scene."""
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Parse terrain generation
    terrain_gen_cfg = my_terrain_cfg.replace(curriculum=args_cli.use_curriculum, color_scheme=args_cli.color_scheme)
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


def main():
    spawn_dz = 2.2

    
    #my_terrain_cfg = get_my_terrain_cfg()
    # Instantiate the template app
    pg_app = PegasusApp()
    
    scene_entities, scene_origins = design_scene()
    set_camera_view(eye=[0.0, -14000.0, 4000.0], target=[0, 0, spawn_dz])

    # Create the vehicle
    # Try to spawn the selected robot in the world to the specified namespace
    spwn_sz = int(1)
    spwn_dh = 3.0
    for x in range(0, spwn_sz):
        for y in range(0, spwn_sz):
            vid = (x*spwn_sz) + y
            config_multirotor = MultirotorConfig()
            # Create the multirotor configuration
            mavlink_config = PX4MavlinkBackendConfig({
                "vehicle_id": vid,
                "px4_autolaunch": True,
                "px4_dir": pg_app.pg.px4_path,
                "px4_vehicle_model": pg_app.pg.px4_default_airframe # CHANGE this line to 'iris' if using PX4 version bellow v1.14
            })
            config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]
            Multirotor(
                "/World/quadrotor",
                ROBOTS['Iris'],
                vid,
                [(x * spwn_dh), (y * spwn_dh), spawn_dz],
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
