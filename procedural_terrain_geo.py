"""
The script generates a 15km x 15km square chunk of terrain from a geographic DEM (Digital Elevation Map) in GeoTIFF
format at ~11m per pixel resolution. It incorporates PX4 Firmware SITL simulation via Pegasus Simulator (v4.5.0) for
drone control. The terrain loads correctly, matches scale in QGroundControl, and drones spawn, connect, take off, and
interact with the terrain. The goal is to apply a texture from a 15km x 15km PNG file (1m resolution, 15000x15000pixels)
 to the terrain mesh, using OmniPBR material with projection mapping to fit 1:1 without UVs on the height field mesh.

Example usage:

.. code-block:: bash
    # Generate terrain from DEM geotif image and apply satellite imagery texture.
    $ISAACSIM_PATH/python.sh procedural_terrain_geo.py
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
# We must get the isaac simulator GUI window up and running before we can continue importing modules...
# append AppLauncher cli args
parser = argparse.ArgumentParser(description="This script demonstrates procedural terrain generation.")
""" parser.add_argument(
    "--color_scheme",
    type=str,
    default="none",
    choices=["height", "random", "none", "texture"],
    help="Color scheme to use for the terrain generation.",
) """
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# Rest everything follows running live:
import carb
import omni.kit.app
import omni.ext
import omni.timeline
import omni.kit.commands
import omni.usd
from pxr import UsdGeom, UsdShade, Sdf, Gf, Vt
from isaacsim.core.utils.viewports import set_camera_view
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBase
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.terrains as terrain
################################### This is my code ####################################################################
from hf_geo_utils import *
from hf_geo_terrain_cfg import HfGeographicTerrainCfg
from hf_geo_terrain import geographic_terrain, read_raster_info, wavelet_terrain
########################################################################################################################
tci_file_path = os.path.abspath("in/TCI_clipped_15km_square.png")  # Make absolute
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
    slope_threshold=None, # 0.75,
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
            tci_tif_file=tci_file_path,
        ),
    },
)
# Pre-configure the Pegasus PegasusSimulator/extensions/pegasus.simulator/config/configs.yaml file with
# the HOME WGS84 coordinates set to the center of the square terrain chunk we generate (from DEM) before 
# we add and enable the Pegasus Simulator extension, so it loads with correct GPS world origin...
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
# Import the Pegasus API for simulating drones. Must do this before attempting pegasus imports
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
    print("[Pegasus Simulator Extension enabled]")
enable_pegasus_ext()
# Finally, we can import the Pegasus Simulator APIs
from omni.isaac.core.world import World
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

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
        # Acquire the World, .i.e, the singleton that controls that is a one stop shop for setting up physics, 
        # spawning asset primitives, etc.
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        # Enviornment will be dynamically generated before spawning drones...
        
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

""" def set_ground_texture():
    # Apply texture to the generated terrain using OmniPBR with triplanar mapping
    stage = omni.usd.get_context().get_stage()
    mat_prim_path = "/Looks/TerrainTextureMaterial"
    success, result = omni.kit.commands.execute('CreateMdlMaterialPrimCommand',
        mtl_url='OmniPBR.mdl',
        mtl_name='OmniPBR',
        mtl_path=mat_prim_path
    )
    if not success:
        print("Failed to create MDL material:", result)
        return {}, torch.tensor([])  # Early exit if fail
    mat = UsdShade.Material(stage.GetPrimAtPath(mat_prim_path))
    shader_path = f"{mat_prim_path}/Shader"
    shader = UsdShade.Shader(stage.GetPrimAtPath(shader_path))
    print(f"Shader inputs:", [i.GetFullName() for i in shader.GetInputs()]) # type: ignore
    print(f"Texture path exists:", os.path.exists(tci_file_path))

    # Texture input
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(tci_file_path) # type: ignore
    # Enable texture usage

    # Enable UVW box projection (for terrains without UVs, triplanar-like)
    shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True) # type: ignore
    # Scale and offset in one input
    tri_scale = float(1.0 / terrain_size[0])  # Fit once across
    print("Calculated texture scale:", tri_scale)  # Debug: ~6.666e-5
    # Rotation (optional)
    shader.CreateInput("texture_rotate", Sdf.ValueTypeNames.Float).Set(0.0) # type: ignore
    # No color tint (white = pure texture)
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set((1.0, 1.0, 1.0)) # type: ignore
    # Scale texture to stretch across entire chunk of terrain 
    shader.CreateInput("texture_translate", Sdf.ValueTypeNames.TexCoord2f).Set((0.5, 0.5))  # type: ignore
    shader.CreateInput("texture_scale", Sdf.ValueTypeNames.TexCoord2f).Set((tri_scale, tri_scale))  # type: ignore

    # Finish up the sher pipeline
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface") # type: ignore
    mat.CreateVolumeOutput().ConnectToSource(shader.ConnectableAPI(), "volume") # type: ignore
    # Bind to the actual Mesh prim (child)
    terrain_mesh_path = "/World/ground/terrain/mesh"  # From your debug print
    terrain_mesh = stage.GetPrimAtPath(terrain_mesh_path)
    if terrain_mesh.IsValid():
        UsdShade.MaterialBindingAPI(terrain_mesh).Bind(mat) # type: ignore
        print("Material bound successfully to", terrain_mesh_path)
        simulation_app.update()  # Force render refresh
        simulation_app.update()  # Double update for safety
    else:
        print(f"Error: Terrain mesh at {terrain_mesh_path} is invalid!")
    print(f"Applied material {UsdShade.MaterialBindingAPI(terrain_mesh).GetDirectBinding().GetMaterialPath()} to terrain:") # type: ignore """
    
""" def set_ground_texture():
    stage = omni.usd.get_context().get_stage()
    mat_prim_path = "/Looks/TerrainTextureMaterial"
    success, _ = omni.kit.commands.execute(
        "CreateMdlMaterialPrimCommand",
        mtl_url="OmniPBR.mdl",
        mtl_name="OmniPBR",
        mtl_path=mat_prim_path,
    )
    if not success:
        print("Failed to create MDL material")
        return

    mat = UsdShade.Material(stage.GetPrimAtPath(mat_prim_path))
    shader = UsdShade.Shader(stage.GetPrimAtPath(f"{mat_prim_path}/Shader"))

    # 1) Texture
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(tci_file_path)  # type: ignore

    # 2) Use UVW projection, but in **world space**
    shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)         # type: ignore # enable projection
    shader.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(True)     # type: ignore # True = world space  ✅

    # 3) Scale/offset: let the bbox map to 0–1 (fit once). Do NOT shrink by 1/terrain_size.
    shader.CreateInput("texture_scale", Sdf.ValueTypeNames.TexCoord2f).Set(Gf.Vec2f(1.0 / avg_geo_size, 1.0 / avg_geo_size)) # type: ignore
    shader.CreateInput("texture_translate", Sdf.ValueTypeNames.TexCoord2f).Set(Gf.Vec2f(0.5, 0.5)) # type: ignore
    shader.CreateInput("texture_rotate", Sdf.ValueTypeNames.Float).Set(0.0) # type: ignore

    # 4) No tint
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0)) # type: ignore

    # Bind
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface") # type: ignore
    terrain_mesh_path = "/World/ground/terrain/mesh"
    terrain_mesh = stage.GetPrimAtPath(terrain_mesh_path)
    if terrain_mesh and terrain_mesh.IsValid():
        UsdShade.MaterialBindingAPI(terrain_mesh).Bind(mat) # type: ignore
        print("Material bound successfully  to", terrain_mesh_path)"""

def set_ground_texture():
    stage = omni.usd.get_context().get_stage()
    mat_prim_path = "/Looks/TerrainTextureMaterial"
    success, _ = omni.kit.commands.execute(
        "CreateMdlMaterialPrimCommand",
        mtl_url="OmniPBR.mdl",
        mtl_name="OmniPBR",
        mtl_path=mat_prim_path,
    )
    if not success:
        print("Failed to create MDL material")
        return

    mat = UsdShade.Material(stage.GetPrimAtPath(mat_prim_path))
    shader = UsdShade.Shader(stage.GetPrimAtPath(f"{mat_prim_path}/Shader"))

    # Texture setup
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(tci_file_path)  # type: ignore

    # Disable projection (we'll use explicit UVs instead)
    shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(False)  # type: ignore

    # Prevent repetition/mirroring artifacts
    shader.CreateInput("texture_wrap_s", Sdf.ValueTypeNames.Token).Set("clamp")  # type: ignore
    shader.CreateInput("texture_wrap_t", Sdf.ValueTypeNames.Token).Set("clamp")  # type: ignore

    # No tint or rotation needed
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))  # type: ignore
    shader.CreateInput("texture_rotate", Sdf.ValueTypeNames.Float).Set(0.0)  # type: ignore

    # Scale/translate: Since we're using UVs normalized to 0-1, set to full size (no shrinkage)
    shader.CreateInput("texture_scale", Sdf.ValueTypeNames.TexCoord2f).Set(Gf.Vec2f(1.0, 1.0))  # type: ignore
    shader.CreateInput("texture_translate", Sdf.ValueTypeNames.TexCoord2f).Set(Gf.Vec2f(0.0, 0.0))  # type: ignore

    # Connect outputs
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")  # type: ignore

    # Get the terrain mesh
    terrain_mesh_path = "/World/ground/terrain/mesh"
    terrain_mesh = UsdGeom.Mesh(stage.GetPrimAtPath(terrain_mesh_path))
    # if not terrain_mesh or not terrain_mesh.IsValid():
    #     print(f"Error: Terrain mesh at {terrain_mesh_path} is invalid!")
    #     return

    # Assign explicit UVs (st primvar) to the mesh vertices
    points = terrain_mesh.GetPointsAttr().Get()
    if not points:
        print("Error: No points found on terrain mesh!")
        return
    
    num_verts = len(points)
    
    """ print(f"\npoints type: {type(points)}\n")
    
    # Infer grid dimensions from terrain cfg (rows/cols based on heightfield shape)
    num_rows = int(terrain_size[1] / h_scale) + 1  # +1 for grid points
    num_cols = int(terrain_size[0] / h_scale) + 1  # +1 for grid points
    if num_verts != num_rows * num_cols:
        print(f"Error: Vertex count mismatch! Expected {num_rows * num_cols}, got {num_verts}")
        return """

    # Infer actual grid dimensions from ordered points (row-major: Y outer, X inner)
    if num_verts > 0:
        current_y = points[0][1]
        col_count = 1
        for p in points[1:]:
            if abs(p[1] - current_y) < 1e-6:  # FP tolerance for equality
                col_count += 1
            else:
                break
        num_cols = col_count
        num_rows = num_verts // num_cols
        if num_verts != num_rows * num_cols:
            print(f"Error: Irregular grid! Vertex count {num_verts} not divisible by inferred cols {num_cols}")
            return
        print(f"Inferred grid: {num_rows} rows x {num_cols} cols (total verts: {num_verts})")
    else:
        print("Error: No points!")
        return
    
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    extent_x = max_x - min_x
    extent_y = max_y - min_y

    # Generate UVs: Normalize based on XY position (0-1 across terrain)
    # Account for potential Y-flip in PNG (v=1 at bottom, assuming standard top-left origin in image)
    uvs = []
    for p in points:
        u = (p[0] - min_x) / extent_x if extent_x > 0 else 0.0
        #v = 1.0 - ((p[1] - min_y) / extent_y) if extent_y > 0 else 0.0  # Flip V to match typical image orientation
        v = (p[1] - min_y) / extent_y if extent_y > 0 else 0.0  # No Flip V to match typical image orientation
        uvs.append(Gf.Vec2f(u, v))

    """ # Set UVs as primvar (interpolation='vertex' for per-vertex UVs)
    st_primvar = terrain_mesh.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st_primvar.Set(uvs) """

    # Set UVs as primvar (interpolation='vertex' for per-vertex UVs)
    primvars_api = UsdGeom.PrimvarsAPI(terrain_mesh.GetPrim())
    st_primvar = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st_primvar.Set(uvs)

    # Bind the material
    UsdShade.MaterialBindingAPI(terrain_mesh.GetPrim()).Bind(mat)  # type: ignore
    print("Material bound successfully to", terrain_mesh_path)
    print("UVs assigned to mesh for 1:1 planar mapping.")

def design_scene() -> tuple[dict, torch.Tensor]:
    """Designs the scene."""
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    # Parse terrain generation
    terrain_gen_cfg = my_terrain_cfg.replace(curriculum=False, color_scheme="none")
    # Handler for terrains importing
    terrain_importer_cfg = terrain.TerrainImporterCfg(
        num_envs=1,
        env_spacing=None,
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type="generator",
        terrain_generator=terrain_gen_cfg,
        debug_vis=False
    )
    # We must explicitly disable default material
    # Inorder to be able to modify it later...
    terrain_importer_cfg.visual_material = None
    # Create terrain importer
    terrain_importer = terrain.TerrainImporter(terrain_importer_cfg)
    set_ground_texture()
    # return the scene information
    scene_entities = {"terrain": terrain_importer}
    return scene_entities, terrain_importer.env_origins

def main():
    spwn_dxy = 3.0 # horizontal grid spacing when spawning multiple uavs
    spawn_dz = 2.2 # altitude to spawn drones. TODO: automatically sample from DEM height in future
    set_camera_view(eye=[0.0, -14000.0, 4000.0], target=[0, 0, spawn_dz])
    # Instantiate the template app
    pg_app = PegasusApp()
    # Generate terrain, texture material, and bind to one another
    scene_entities, scene_origins = design_scene()
    # Create the vehicle
    # Try to spawn the selected UAV(s) into the world to the specified namespace
    spwn_sz = int(1)
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
                [(x * spwn_dxy), (y * spwn_dxy), spawn_dz],
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