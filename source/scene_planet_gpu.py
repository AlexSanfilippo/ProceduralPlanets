"""
Draw a triangular face
"""
from random import random

import glfw
import numpy as np
import pyglm
from OpenGL.GL import *
import pyrr

from engine.shader_program import create_shader
from engine.texture_loader import load_texture
from engine.camera import SimulationCamera, RollableCamera
from engine.skybox import Skybox
from math import sin, cos
from glm import cos, radians
import logging
import engine.point_light_cube as plc
from engine.procedural_mesh import CubeMeshStatic
from engine.gui_batched import GUIBatched
from engine.screen_capture import capture_screenshot, write_fbo_to_gif, save_to_gif
from noise_generators import fbm_noise, fbm_terrain_3d
from planet_mesh import TriangleIndexed, Cube, Icosphere, TriangleSubdivided, IcosphereSubdivided, PlanetMesh, \
    PlanetMeshGPU, PlanetDiscreteLOD, Line
import ctypes
import numpy as np
from OpenGL.GL import *
from pyglm.glm import vec3

from signed_distance_function import Sphere, SphereTransparent, Atmosphere

#todo: move to config file
config_scene = {
    "background_color": [0.0, 0.1, 0.1, 1.0],
    "initial_camera_pos": [0.0, 20.0, 20.0],
    "initial_camera_front": [0.0, -0.5, -1.0],
    "name": "Minimal Scene",
    "enable_backface_culling": True,
    "cull_face_mode": "BACK",  # options: BACK, FRONT
}


logger = logging.getLogger(name=__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


"""===============GLOBAL VARIABLES======================="""
WIDTH, HEIGHT = 1728, 972
# WIDTH, HEIGHT = 400, 200
WINDOW_POSITION = (40, 40)
WRITE_TO_GIF = False
DRAW_GUI = True
lastX, lastY = WIDTH / 2, HEIGHT / 2
DRAW_DISTANCE = 30000
NEAR_PLANE_MIN = 0.1
NEAR_PLANE_MAX = 10.0
NEAR_PLANE = NEAR_PLANE_MAX
DRAW_DEBUG_LINES = False
camera_speed = 25.0
PLANET_RADIUS = 320.0
PLANET_AXIS = vec3(0.5, 1.0, 0.0)  # defines the north-south poles
VIEW_MODE = 0          # 0: terrain, 1: normals, 2: heat map
GLOBAL_TEMPERATURE = 0.0
GLOBAL_RAINFALL_REDUCTION = 0.0

#key-input globals
first_mouse = True
left, right, forward, backward, make_new_surface = False, False, False, False, False
player_left, player_right, player_forward, player_backward = False, False, False, False
yaw_counterclockwise, yaw_clockwise = False, False
up, down = False, False
pause = False
switch_view_mode = False
# initializing glfw library
if not glfw.init():
    raise Exception("glfw can not be initialized!")

# creating the window
window = glfw.create_window(WIDTH, HEIGHT, config_scene['name'], None, None)

# check if window was created
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

# set window position
glfw.set_window_pos(window, *WINDOW_POSITION)

# the keyboard input callback
def key_input_clb(window, key, scancode, action, mode):
    global left, right, forward, backward, make_new_surface, player_left, player_right, player_forward, \
        player_backward, yaw_counterclockwise, yaw_clockwise, \
        pause, up, down, wrote_to_gif, switch_view_mode, camera_speed, DRAW_GUI, WRITE_TO_GIF

    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)
    if key == glfw.KEY_W and action == glfw.PRESS:
        forward = True
    elif key == glfw.KEY_W and action == glfw.RELEASE:
        forward = False
    if key == glfw.KEY_S and action == glfw.PRESS:
        backward = True
    elif key == glfw.KEY_S and action == glfw.RELEASE:
        backward = False
    if key == glfw.KEY_A and action == glfw.PRESS:
        left = True
    elif key == glfw.KEY_A and action == glfw.RELEASE:
        left = False
    if key == glfw.KEY_D and action == glfw.PRESS:
        right = True
    elif key == glfw.KEY_D and action == glfw.RELEASE:
        right = False
    if key == glfw.KEY_Q and action == glfw.PRESS:
        yaw_clockwise = True
    elif key == glfw.KEY_Q and action == glfw.RELEASE:
        yaw_clockwise = False
    if key == glfw.KEY_E and action == glfw.PRESS:
        yaw_counterclockwise = True
    elif key == glfw.KEY_E and action == glfw.RELEASE:
        yaw_counterclockwise = False
    if key == glfw.KEY_TAB and action == glfw.PRESS:
        up = True
    elif key == glfw.KEY_TAB and action == glfw.RELEASE:
        up = False
    if key == glfw.KEY_LEFT_SHIFT and action == glfw.PRESS:
        down = True
    elif key == glfw.KEY_LEFT_SHIFT and action == glfw.RELEASE:
        down = False
    if key == glfw.KEY_1 and action == glfw.PRESS:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    if key == glfw.KEY_2 and action == glfw.PRESS:
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    if key == glfw.KEY_P and action == glfw.PRESS:
        pause = not pause
    if key == glfw.KEY_V and action == glfw.PRESS:
        switch_camera_mode()
    if key == glfw.KEY_F12 and action == glfw.PRESS:
        capture_screenshot(width=WIDTH, height=HEIGHT)
    if key == glfw.KEY_H and action == glfw.PRESS:
        planet.lod_increase()
    if key == glfw.KEY_J and action == glfw.PRESS:
        planet.rotate_towards_position(position = active_camera.camera_pos, line_shader=shader_line, view=active_camera.get_view_matrix())
    if key == glfw.KEY_K and action == glfw.PRESS:
        planet.reset_rotation()
    if key == glfw.KEY_9 and action == glfw.PRESS:
        planet.rotate_planet(axis=(0.25, 1.0, 0.44), angle_degrees=10)
    if key == glfw.KEY_N and action == glfw.PRESS:
        print_camera_position()
    if key == glfw.KEY_LEFT_ALT and action == glfw.PRESS:
        camera_speed = max(1.0, camera_speed - 5.0)
        print(f"Camera speed: {camera_speed:.1f}")
    if key == glfw.KEY_LEFT_CONTROL and action == glfw.PRESS:
        camera_speed += 5.0
        print(f"Camera speed: {camera_speed:.1f}")
    if key == glfw.KEY_F10 and action == glfw.PRESS:
        DRAW_GUI = not DRAW_GUI
        print(f"GUI drawing: {'ON' if DRAW_GUI else 'OFF'}")
    if key == glfw.KEY_F11 and action == glfw.PRESS:
        WRITE_TO_GIF = not WRITE_TO_GIF
        print(f"Write to GIF: {'ON' if WRITE_TO_GIF else 'OFF'}")


def mouse_look_clb(window, xpos, ypos):
    global first_mouse, lastX, lastY
    if first_mouse:
        lastX = xpos
        lastY = ypos
        first_mouse = False
    xoffset = xpos - lastX
    yoffset = lastY - ypos
    lastX = xpos
    lastY = ypos
    cam.process_mouse_movement(xoffset, yoffset)


def scroll_callback(window, xoffset, yoffset):
    sim_cam.process_mouse_scroll(xoffset, yoffset)


def mouse_button_callback(window, button, action, mods):
    pass   # gui_batched click detection is handled continuously in the render loop


def window_resize_clb(window, width, height):
    glViewport(0, 0, width, height)


def do_movement(speed=1.0):
    """
    do the camera movement, call this function in the main loop
    :param speed:
    """
    if left:
        active_camera.process_keyboard("LEFT", speed)
    if right:
        active_camera.process_keyboard("RIGHT", speed)
    if forward:
        active_camera.process_keyboard("FORWARD", speed)
    if backward:
        active_camera.process_keyboard("BACKWARD", speed)
    if yaw_clockwise:
        if use_sim_cam:
            sim_cam.process_keyboard("YAW_CLOCKWISE", speed * 2)
        else:
            cam.process_keyboard("ROLL_LEFT", speed * 30)
    if yaw_counterclockwise:
        if use_sim_cam:
            sim_cam.process_keyboard("YAW_COUNTERCLOCKWISE", speed * 2)
        else:
            cam.process_keyboard("ROLL_RIGHT", speed * 30)
    if up:
        active_camera.process_keyboard("UP", speed)
    if down:
        active_camera.process_keyboard("DOWN", speed)




# set the callback function for window resize
glfw.set_window_size_callback(window, window_resize_clb)
# set the mouse position callback
glfw.set_cursor_pos_callback(window, mouse_look_clb)
# set the keyboard input callback
glfw.set_key_callback(window, key_input_clb)
# set the scroll-wheel input callback
glfw.set_scroll_callback(window, scroll_callback)
glfw.set_mouse_button_callback(window, mouse_button_callback)
# capture the mouse cursor
# glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
# glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_CAPTURED)



# make the context current
glfw.make_context_current(window)

"""CAMERA SETUP"""
sim_cam = SimulationCamera(camera_pos=[PLANET_RADIUS * 2.0, 20.0, 0.0])
cam = RollableCamera(camera_pos=[PLANET_RADIUS * 2.0, 20.0, 20.0], mouse_sensitivity=0.1)
use_sim_cam = True
active_camera = sim_cam

def switch_camera_mode():
    global use_sim_cam, active_camera, lastX, lastY
    use_sim_cam = not use_sim_cam
    if use_sim_cam:
        lastX, lastY = WIDTH / 2, HEIGHT / 2
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)  # glfw.CURSOR_CAPTURED,
        active_camera = sim_cam
    else:
        lastX, lastY = WIDTH / 2, HEIGHT / 2
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        active_camera = cam



def print_camera_position():
    global active_cam
    print(f'Camera Position: {active_camera.camera_pos}')



if config_scene['enable_backface_culling']:
    glEnable(GL_CULL_FACE)
if config_scene['cull_face_mode'] == 'BACK':
    glCullFace(GL_BACK)
elif config_scene['cull_face_mode'] == 'FRONT':
    glCullFace(GL_FRONT)

glClearColor(0, 0.1, 0.1, 1)
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


"""direction light settings"""
# create the light cube
my_plc = plc.PointLightCube(pos=[1305.0, 0.0, 0.0], ambient=[0.0] * 3, diffuse=[1.0] * 3,
                            specular=[0.0] * 3, constant=1.0, linear=0.000014 , quadratic=0.000000007)
light_cubes = [my_plc]


textures = glGenTextures(12)
load_texture("engine/textures/button_atlas_gradient.png", textures[0])
load_texture("engine/fonts/my_font.png", textures[1])
load_texture("engine/textures/banana.png", textures[2])
load_texture("engine/textures/dirt.jpg", textures[3])

biome_tex = glGenTextures(1)
load_texture("engine/textures/biome_map_low_resolution.png", biome_tex)

biome_true_tex = glGenTextures(1)
# load_texture("engine/textures/biome_map_low_resolution_photo_colors.png", biome_true_tex)
load_texture("engine/textures/biome_map_blurred_3.png", biome_true_tex)
# load_texture("engine/textures/biome_map_alien_1.png", biome_true_tex)

biome_key_tex = glGenTextures(1)
load_texture("engine/textures/biome_key.png", biome_key_tex)

"""Shader Compilation"""
#New shader for position and normal only,with 3d camera
shader_program_pos_normal = create_shader(vertex_file='engine/shaders/pos_norm.vs', fragment_file='engine/shaders/pos_norm.fs')
# shader_point_light = create_shader(vertex_file='engine/shaders/pos_norm.vs', fragment_file='engine/shaders/pos_norm_lit.fs')
shader_point_light = create_shader(vertex_file='engine/shaders/pos_norm.vs', fragment_file='engine/shaders/terrain_height_coloring.fs')
shader_terrain_gpu = create_shader(vertex_file='engine/shaders/terrain_planet.vs', fragment_file='engine/shaders/terrain_coloring_gpu.fs')
shader_line = create_shader(vertex_file='engine/shaders/line.vs', fragment_file='engine/shaders/line.fs')

projection = pyrr.matrix44.create_perspective_projection_matrix(45, WIDTH / HEIGHT, NEAR_PLANE, DRAW_DISTANCE)

"""GUI CREATION"""

# ── Batched GUI setup ──────────────────────────────────────────────────────────
_ui_atlas_path = "engine/textures/ui_gray_atlas.png"
_ui_atlas_tex = glGenTextures(1)
load_texture(_ui_atlas_path, _ui_atlas_tex)

gui_batched = GUIBatched(
    screen_size=(WIDTH, HEIGHT),
    texture_atlas=_ui_atlas_tex,
    atlas_map={
        "button": ((39, 0), (77, 108)),
        "slider_background": ((0, 0), (37, 109)),
    },
    atlas_image_path=_ui_atlas_path,
    font_texture=textures[1],
    font_fnt_path="engine/fonts/my_font.fnt",
)

_atm_radius_label = None   # assigned after the text element is created below

def set_atmosphere_radius(value):
    sdf_atmosphere.radius = float(value)
    if _atm_radius_label is not None:
        _atm_radius_label.update_text(f"{float(value):.2f}")

"""Planet Controls"""
planet_settings = {
    "lacunarity": 2.400,
    "gain": 0.750,
    "amplitude": 1.25,
    "frequency": 0.010,
    "seed": int(random()*500),
    "subdivisions": 4,
    "octaves": 8,
    "noise_method": fbm_terrain_3d,
    "displacement_amplitude": 10.0,
    "planet_type": "Earth",
    "draw_hydrosphere": 1,
}

def next_seed():
    global planet_settings
    planet_settings["seed"] = random() * 500
    print(f"New seed: {planet_settings['seed']:.3f}")
    regenerate_planet()

def update_view_mode_for_planet_type():
    """Set default view mode and button visibility based on planet type"""
    global VIEW_MODE
    planet_type = planet_settings["planet_type"]
    
    if planet_type == "Earth":
        # Earth: default to Biome True, show all view modes
        VIEW_MODE = 5
        gui_batched.switch_context_status("view_mode_buttons", True)
    else:
        # Moon/Mars: only show Heightmap view mode
        VIEW_MODE = 6
        gui_batched.switch_context_status("view_mode_buttons", False)
    
    names = {0: "Terrain", 1: "Normals", 2: "Heat", 3: "Rainfall", 4: "Biomes", 5: "Biome True", 6: "Heightmap"}
    print(f"Planet type: {planet_type} | View mode: {names[VIEW_MODE]}")

def cycle_planet_type():
    global planet_settings
    planet_types = ["Moon", "Earth", "Mars"]
    current_index = planet_types.index(planet_settings["planet_type"])
    new_index = (current_index + 1) % len(planet_types)
    planet_settings["planet_type"] = planet_types[new_index]
    print(f"New planet type: {planet_settings['planet_type']}")
    planet.set_planet_type(planet_settings["planet_type"])
    update_view_mode_for_planet_type()

planet = PlanetDiscreteLOD(
        subdivisions = 4,
        shader_program=shader_terrain_gpu,
        position=vec3(0.0, 0.0, 0.0),
        scale=PLANET_RADIUS,
        projection=projection,
        planet_type = planet_settings['planet_type'],
        octaves=planet_settings['octaves'],
        lacunarity=planet_settings['lacunarity'],
        gain=planet_settings['gain'],
        amplitude=planet_settings['amplitude'],
        frequency=planet_settings['frequency'],
)

# Set default view mode based on initial planet type
update_view_mode_for_planet_type()

# create a skybox using banana texture on all six faces
#use the nebula textures for a more interesting skybox
skybox_paths = ["engine/textures/nebula/skybox_left.png", "engine/textures/nebula/skybox_right.png", "engine/textures/nebula/skybox_up.png",] \
                + ["engine/textures/nebula/skybox_down.png", "engine/textures/nebula/skybox_front.png", "engine/textures/nebula/skybox_back.png"]
skybox = Skybox(skybox_paths, scale=10000)


# == Create of SDFs ==

sdf_ocean = SphereTransparent(
    position=vec3(0.0, 0.0, 0.0),
    radius=PLANET_RADIUS + .2,
    color=vec3(0.1, 0.2, 0.6),
    # color=vec3(0.6, 0.2, 0.1),
    transparency=0.05,
    max_depth=1.0,
    near_plane=NEAR_PLANE,
    far_plane=DRAW_DISTANCE,
)


sdf_atmosphere = Atmosphere(
    position=vec3(0.0, 0.0, 0.0),
    # radius=PLANET_RADIUS * 1.05,
    radius=PLANET_RADIUS * 1.03,
    color=vec3(0.53, 0.81, 0.98),
    transparency=0.0,
    min_depth=0.0,
    max_depth=PLANET_RADIUS * 0.40,
    near_plane=NEAR_PLANE,
    far_plane=DRAW_DISTANCE,
)


def regenerate_planet():
    global planet, planet_settings
    planet.cleanup()
    planet = PlanetDiscreteLOD(
        subdivisions=planet_settings["subdivisions"],
        shader_program=shader_terrain_gpu,
        position=vec3(0.0, 0.0, 0.0),
        scale=PLANET_RADIUS,
        projection=projection,
        planet_type=planet_settings["planet_type"],
        octaves = planet_settings['octaves'],
        lacunarity = planet_settings['lacunarity'],
        gain = planet_settings['gain'],
        amplitude = planet_settings['amplitude'],
        frequency = planet_settings['frequency'],
    )

def change_planet_setting(delta=0.05, setting="lacunarity"):
    global planet_settings
    planet_settings[setting] = delta + planet_settings[setting]
    planet.update_noise_parameter(parameter=setting, value=planet_settings[setting])
    print(f"Changed {setting} to {planet_settings[setting]:.3f}")


control_value_elements = {}


def toggle_controls_panel():
    gui_batched.toggle_context_status("controls_panel")
    enabled = gui_batched._context_status.get("controls_panel", True)
    print(f"Controls panel: {'ON' if enabled else 'OFF'}")


def toggle_view_mode_buttons():
    gui_batched.toggle_context_status("view_mode_buttons")
    enabled = gui_batched._context_status.get("view_mode_buttons", True)
    print(f"View mode buttons: {'ON' if enabled else 'OFF'}")


def set_planet_setting_from_slider(value, setting, digits=3, clamp_min=None, clamp_max=None):
    new_value = float(value)
    if clamp_min is not None:
        new_value = max(clamp_min, new_value)
    if clamp_max is not None:
        new_value = min(clamp_max, new_value)
    planet_settings[setting] = new_value
    planet.update_noise_parameter(parameter=setting, value=new_value)
    if setting in control_value_elements:
        control_value_elements[setting].update_text(text=f"{new_value:.{digits}f}")


def change_seed(delta):
    new_seed = float(np.clip(float(planet_settings["seed"]) + float(delta), 0.0, 500.0))
    planet_settings["seed"] = new_seed
    planet.update_noise_parameter(parameter="seed", value=new_seed)
    if "seed" in control_value_elements:
        control_value_elements["seed"].update_text(text=f"{int(new_seed)}")


def set_sea_level_from_slider(value):
    sdf_ocean.radius = float(value)
    if "sea_level" in control_value_elements:
        control_value_elements["sea_level"].update_text(text=f"{sdf_ocean.radius:.2f}")


def set_global_temperature_from_slider(value):
    global GLOBAL_TEMPERATURE
    GLOBAL_TEMPERATURE = float(value)
    if "temperature" in control_value_elements:
        control_value_elements["temperature"].update_text(text=f"{GLOBAL_TEMPERATURE:.2f}")


def set_global_rainfall_reduction_from_slider(value):
    global GLOBAL_RAINFALL_REDUCTION
    GLOBAL_RAINFALL_REDUCTION = float(value)
    if "rainfall_reduction" in control_value_elements:
        control_value_elements["rainfall_reduction"].update_text(text=f"{GLOBAL_RAINFALL_REDUCTION:.2f}")


def set_atmosphere_max_depth_from_slider(value):
    sdf_atmosphere.max_depth = float(value) * PLANET_RADIUS
    if "atmos_opacity" in control_value_elements:
        control_value_elements["atmos_opacity"].update_text(text=f"{float(value):.2f}")


def change_octaves(delta):
    new_octaves = int(np.clip(int(planet_settings["octaves"]) + int(delta), 1, 20))
    planet_settings["octaves"] = new_octaves
    planet.update_noise_parameter(parameter="octaves", value=new_octaves)
    if "octaves" in control_value_elements:
        control_value_elements["octaves"].update_text(text=f"{int(new_octaves)}")


def set_atmosphere_color_component(value, component):
    """Update a single RGB component of the atmosphere color"""
    new_value = float(value)
    sdf_atmosphere.color[component] = new_value
    color_labels = ['atmos_color_r', 'atmos_color_g', 'atmos_color_b']
    if color_labels[component] in control_value_elements:
        control_value_elements[color_labels[component]].update_text(text=f"{new_value:.2f}")


def set_ocean_color_component(value, component):
    """Update a single RGB component of the ocean color"""
    new_value = float(value)
    sdf_ocean.color[component] = new_value
    color_labels = ['ocean_color_r', 'ocean_color_g', 'ocean_color_b']
    if color_labels[component] in control_value_elements:
        control_value_elements[color_labels[component]].update_text(text=f"{new_value:.2f}")


def toggle_atmos_color_sliders():
    gui_batched.toggle_context_status("atmos_color_sliders")
    enabled = gui_batched._context_status.get("atmos_color_sliders", True)
    print(f"Atmosphere color sliders: {'ON' if enabled else 'OFF'}")


def toggle_ocean_color_sliders():
    gui_batched.toggle_context_status("ocean_color_sliders")
    enabled = gui_batched._context_status.get("ocean_color_sliders", True)
    print(f"Ocean color sliders: {'ON' if enabled else 'OFF'}")


def add_control_slider(label, key, value, min_value, max_value, y_pos, callback):
    gui_batched.add_text_element(
        texture_name="button",
        position=(-0.90, y_pos),
        scale=(0.12, 0.03),
        text=label,
        font_size=0.18,
        context_id="controls_panel",
    )
    control_value_elements[key] = gui_batched.add_text_element(
        texture_name="button",
        position=(-0.32, y_pos),
        scale=(0.10, 0.03),
        text=f"{value:.3f}",
        font_size=0.18,
        context_id="controls_panel",
    )
    gui_batched.add_slider(
        bg_texture_name="slider_background",
        knob_texture_name="button",
        position=(-0.60, y_pos),
        scale=(0.18, 0.025),
        min_value=min_value,
        max_value=max_value,
        value=value,
        callback=callback,
        context_id="controls_panel",
    )


gui_batched.add_text_button(
    texture_name="button",
    position=(0.85, -0.95),
    scale=(0.10, 0.05),
    click_function=cycle_planet_type,
    text="Planet Type",
    font_size=0.2,
    rotate_90_cw=True,
)

gui_batched.add_text_button(
    texture_name="button",
    position=(-0.90, 0.96),   # top_left: (-1.0+0.10, 1.0-0.04)
    scale=(0.10, 0.04),
    click_function=toggle_controls_panel,
    text="Controls",
    font_size=0.2,
    rotate_90_cw=True,
    context_status=True,
)

gui_batched.add_text_button(
    texture_name="button",
    position=(0.90, 0.96),   # top_right: (1.0-0.10, 1.0-0.04)
    scale=(0.10, 0.04),
    click_function=toggle_view_mode_buttons,
    text="View Mode",
    font_size=0.2,
    rotate_90_cw=True,
)

gui_batched.add_text_button(
    texture_name="button",
    position=(-0.50, 0.96),
    scale=(0.12, 0.04),
    click_function=toggle_atmos_color_sliders,
    text="Atmos. Color",
    font_size=0.2,
    rotate_90_cw=True,
)

gui_batched.add_text_button(
    texture_name="button",
    position=(-0.30, 0.96),
    scale=(0.12, 0.04),
    click_function=toggle_ocean_color_sliders,
    text="Ocean Color",
    font_size=0.2,
    rotate_90_cw=True,
)

add_control_slider(
    label="Lacunarity",
    key="lacunarity",
    value=planet_settings["lacunarity"],
    min_value=0.5,
    max_value=4.0,
    y_pos=0.90,
    callback=lambda value: set_planet_setting_from_slider(value, "lacunarity", digits=3),
)

add_control_slider(
    label="Frequency",
    key="frequency",
    value=planet_settings["frequency"],
    min_value=0.0,
    max_value=0.01,
    y_pos=0.80,
    callback=lambda value: set_planet_setting_from_slider(value, "frequency", digits=4),
)

gui_batched.add_text_element(
    texture_name="button",
    position=(-0.90, 0.70),
    scale=(0.12, 0.03),
    text="Amplitude",
    font_size=0.18,
    context_id="controls_panel",
)
control_value_elements["amplitude"] = gui_batched.add_text_element(
    texture_name="button",
    position=(-0.32, 0.70),
    scale=(0.10, 0.03),
    text=f"{planet_settings['amplitude']:.3f}",
    font_size=0.18,
    context_id="controls_panel",
)
gui_batched.add_slider(
    bg_texture_name="slider_background",
    knob_texture_name="button",
    position=(-0.60, 0.70),
    scale=(0.18, 0.025),
    min_value=0.0,
    max_value=10.0,
    value=planet_settings["amplitude"],
    callback=lambda value: set_planet_setting_from_slider(value, "amplitude", digits=3),
    context_id="controls_panel",
)

add_control_slider(
    label="Gain",
    key="gain",
    value=planet_settings["gain"],
    min_value=0.0,
    max_value=1.0,
    y_pos=0.60,
    callback=lambda value: set_planet_setting_from_slider(value, "gain", digits=3),
)

add_control_slider(
    label="Sea Level",
    key="sea_level",
    value=sdf_ocean.radius,
    min_value=PLANET_RADIUS - 5.0,
    max_value=PLANET_RADIUS + 5.0,
    y_pos=0.50,
    callback=set_sea_level_from_slider,
)

add_control_slider(
    label="Temperature",
    key="temperature",
    value=GLOBAL_TEMPERATURE,
    min_value=-1.0,
    max_value=1.0,
    y_pos=0.40,
    callback=set_global_temperature_from_slider,
)

add_control_slider(
    label="Rainfall Red.",
    key="rainfall_reduction",
    value=GLOBAL_RAINFALL_REDUCTION,
    min_value=0.0,
    max_value=1.0,
    y_pos=0.30,
    callback=set_global_rainfall_reduction_from_slider,
)

gui_batched.add_text_element(
    texture_name="button",
    position=(-0.90, 0.20),
    scale=(0.12, 0.03),
    text="Seed",
    font_size=0.18,
    context_id="controls_panel",
)
control_value_elements["seed"] = gui_batched.add_text_element(
    texture_name="button",
    position=(-0.32, 0.20),
    scale=(0.10, 0.03),
    text=f"{int(planet_settings['seed'])}",
    font_size=0.18,
    context_id="controls_panel",
)
gui_batched.add_text_button(
    texture_name="button",
    position=(-0.70, 0.20),
    scale=(0.10, 0.04),
    click_function=lambda: change_seed(-1),
    text="Seed -",
    font_size=0.2,
    rotate_90_cw=True,
    context_id="controls_panel",
)
gui_batched.add_text_button(
    texture_name="button",
    position=(-0.50, 0.20),
    scale=(0.10, 0.04),
    click_function=lambda: change_seed(1),
    text="Seed +",
    font_size=0.2,
    rotate_90_cw=True,
    context_id="controls_panel",
)

# Atmosphere radius slider
gui_batched.add_text_element(
    texture_name="button",
    position=(-0.90, 0.10),
    scale=(0.12, 0.03),
    text="Atmosphere",
    font_size=0.18,
    context_id="controls_panel",
)
_atm_radius_label = gui_batched.add_text_element(
    texture_name="button",
    position=(-0.32, 0.10),
    scale=(0.10, 0.03),
    text=f"{PLANET_RADIUS * 1.03:.2f}",
    font_size=0.18,
    context_id="controls_panel",
)
gui_batched.add_slider(
    bg_texture_name="slider_background",
    knob_texture_name="button",
    position=(-0.60, 0.10),
    scale=(0.18, 0.025),
    min_value=PLANET_RADIUS * 0.95,
    max_value=PLANET_RADIUS * 1.15,
    value=PLANET_RADIUS * 1.03,
    callback=set_atmosphere_radius,
    context_id="controls_panel",
)

# Atmosphere opacity slider
gui_batched.add_text_element(
    texture_name="button",
    position=(-0.90, 0.00),
    scale=(0.12, 0.03),
    text="Atmos. Opacity",
    font_size=0.18,
    context_id="controls_panel",
)
current_atmos_opacity = sdf_atmosphere.max_depth / PLANET_RADIUS
control_value_elements["atmos_opacity"] = gui_batched.add_text_element(
    texture_name="button",
    position=(-0.32, 0.00),
    scale=(0.10, 0.03),
    text=f"{current_atmos_opacity:.2f}",
    font_size=0.18,
    context_id="controls_panel",
)
gui_batched.add_slider(
    bg_texture_name="slider_background",
    knob_texture_name="button",
    position=(-0.60, 0.00),
    scale=(0.18, 0.025),
    min_value=0.0,
    max_value=1.0,
    value=current_atmos_opacity,
    callback=set_atmosphere_max_depth_from_slider,
    context_id="controls_panel",
)

# Octaves controls (similar to Seed controls)
gui_batched.add_text_element(
    texture_name="button",
    position=(-0.90, -0.10),
    scale=(0.12, 0.03),
    text="Octaves",
    font_size=0.18,
    context_id="controls_panel",
)
control_value_elements["octaves"] = gui_batched.add_text_element(
    texture_name="button",
    position=(-0.32, -0.10),
    scale=(0.10, 0.03),
    text=f"{int(planet_settings['octaves'])}",
    font_size=0.18,
    context_id="controls_panel",
)
gui_batched.add_text_button(
    texture_name="button",
    position=(-0.70, -0.10),
    scale=(0.10, 0.04),
    click_function=lambda: change_octaves(-1),
    text="Oct -",
    font_size=0.2,
    rotate_90_cw=True,
    context_id="controls_panel",
)
gui_batched.add_text_button(
    texture_name="button",
    position=(-0.50, -0.10),
    scale=(0.10, 0.04),
    click_function=lambda: change_octaves(1),
    text="Oct +",
    font_size=0.2,
    rotate_90_cw=True,
    context_id="controls_panel",
)

def set_view_normals():
    global VIEW_MODE
    VIEW_MODE = 6 if VIEW_MODE == 1 else 1
    gui_batched.switch_context_status("biome_key_overlay", VIEW_MODE == 4)
    names = {0: "Terrain", 1: "Normals", 2: "Heat", 3: "Rainfall", 4: "Biomes", 5: "Biome True", 6: "Heightmap"}
    print(f"View mode: {names[VIEW_MODE]}")

def set_view_heat():
    global VIEW_MODE
    VIEW_MODE = 6 if VIEW_MODE == 2 else 2
    gui_batched.switch_context_status("biome_key_overlay", VIEW_MODE == 4)
    names = {0: "Terrain", 1: "Normals", 2: "Heat", 3: "Rainfall", 4: "Biomes", 5: "Biome True", 6: "Heightmap"}
    print(f"View mode: {names[VIEW_MODE]}")

def set_view_rainfall():
    global VIEW_MODE
    VIEW_MODE = 6 if VIEW_MODE == 3 else 3
    gui_batched.switch_context_status("biome_key_overlay", VIEW_MODE == 4)
    names = {0: "Terrain", 1: "Normals", 2: "Heat", 3: "Rainfall", 4: "Biomes", 5: "Biome True", 6: "Heightmap"}
    print(f"View mode: {names[VIEW_MODE]}")

def set_view_biomes():
    global VIEW_MODE
    VIEW_MODE = 6 if VIEW_MODE == 4 else 4
    gui_batched.switch_context_status("biome_key_overlay", VIEW_MODE == 4)
    names = {0: "Terrain", 1: "Normals", 2: "Heat", 3: "Rainfall", 4: "Biomes", 5: "Biome True", 6: "Heightmap"}
    print(f"View mode: {names[VIEW_MODE]}")

def set_view_biome_true():
    global VIEW_MODE
    VIEW_MODE = 6 if VIEW_MODE == 5 else 5
    gui_batched.switch_context_status("biome_key_overlay", VIEW_MODE == 4)
    names = {0: "Terrain", 1: "Normals", 2: "Heat", 3: "Rainfall", 4: "Biomes", 5: "Biome True", 6: "Heightmap"}
    print(f"View mode: {names[VIEW_MODE]}")

def set_view_heightmap():
    global VIEW_MODE
    VIEW_MODE = 1 if VIEW_MODE == 6 else 6
    gui_batched.switch_context_status("biome_key_overlay", VIEW_MODE == 4)
    names = {0: "Terrain", 1: "Normals", 2: "Heat", 3: "Rainfall", 4: "Biomes", 5: "Biome True", 6: "Heightmap"}
    print(f"View mode: {names[VIEW_MODE]}")

gui_batched.add_text_button(
    texture_name="button",
    position=(0.85, 0.81),   # top_right: (1.0-0.15, 0.86-0.05)
    scale=(0.15, 0.05),
    click_function=set_view_normals,
    text="View Normal",
    font_size=0.2,
    rotate_90_cw=True,
    context_id="view_mode_buttons",
    context_status=False,    # starts hidden; revealed by View Mode button
)
gui_batched.add_text_button(
    texture_name="button",
    position=(0.85, 0.69),   # top_right: (1.0-0.15, 0.74-0.05)
    scale=(0.15, 0.05),
    click_function=set_view_heat,
    text="View Heat",
    font_size=0.2,
    rotate_90_cw=True,
    context_id="view_mode_buttons",
    context_status=False,    # starts hidden; revealed by View Mode button
)
gui_batched.add_text_button(
    texture_name="button",
    position=(0.85, 0.57),
    scale=(0.15, 0.05),
    click_function=set_view_rainfall,
    text="Rainfall",
    font_size=0.2,
    rotate_90_cw=True,
    context_id="view_mode_buttons",
    context_status=False,    # starts hidden; revealed by View Mode button
)
gui_batched.add_text_button(
    texture_name="button",
    position=(0.85, 0.45),
    scale=(0.15, 0.05),
    click_function=set_view_biomes,
    text="Biomes",
    font_size=0.2,
    rotate_90_cw=True,
    context_id="view_mode_buttons",
    context_status=False,    # starts hidden; revealed by View Mode button
)
gui_batched.add_text_button(
    texture_name="button",
    position=(0.85, 0.33),
    scale=(0.15, 0.05),
    click_function=set_view_biome_true,
    text="Biome True",
    font_size=0.2,
    rotate_90_cw=True,
    context_id="view_mode_buttons",
    context_status=False,    # starts hidden; revealed by View Mode button
)
gui_batched.add_text_button(
    texture_name="button",
    position=(0.85, 0.21),
    scale=(0.15, 0.05),
    click_function=set_view_heightmap,
    text="Heightmap",
    font_size=0.2,
    rotate_90_cw=True,
    context_id="view_mode_buttons",
    context_status=False,    # starts hidden; revealed by View Mode button
)

# Biome key overlay — shown only in Biome view mode (starts hidden)
gui_batched.add_texture_element(
    texture_id=biome_key_tex,
    position=(0.72, -0.55),
    scale=(0.125, 0.21),
    context_id="biome_key_overlay",
    context_status=False,
)

# ========== ATMOSPHERE COLOR SLIDERS ==========
for i, component_name in enumerate(['R', 'G', 'B']):
    y_pos = 0.85 - i * 0.10
    
    gui_batched.add_text_element(
        texture_name="button",
        position=(-0.90, y_pos),
        scale=(0.08, 0.03),
        text=f"Atm.{component_name}",
        font_size=0.16,
        context_id="atmos_color_sliders",
        context_status=False,
    )
    control_value_elements[f"atmos_color_{component_name.lower()}"] = gui_batched.add_text_element(
        texture_name="button",
        position=(-0.32, y_pos),
        scale=(0.10, 0.03),
        text=f"{sdf_atmosphere.color[i]:.2f}",
        font_size=0.16,
        context_id="atmos_color_sliders",
        context_status=False,
    )
    gui_batched.add_slider(
        bg_texture_name="slider_background",
        knob_texture_name="button",
        position=(-0.60, y_pos),
        scale=(0.18, 0.025),
        min_value=0.0,
        max_value=1.0,
        value=sdf_atmosphere.color[i],
        callback=lambda val, comp=i: set_atmosphere_color_component(val, comp),
        context_id="atmos_color_sliders",
        context_status=False,
    )

# ========== OCEAN COLOR SLIDERS ==========
for i, component_name in enumerate(['R', 'G', 'B']):
    y_pos = 0.45 - i * 0.10
    
    gui_batched.add_text_element(
        texture_name="button",
        position=(-0.90, y_pos),
        scale=(0.08, 0.03),
        text=f"Oce.{component_name}",
        font_size=0.16,
        context_id="ocean_color_sliders",
        context_status=False,
    )
    control_value_elements[f"ocean_color_{component_name.lower()}"] = gui_batched.add_text_element(
        texture_name="button",
        position=(-0.32, y_pos),
        scale=(0.10, 0.03),
        text=f"{sdf_ocean.color[i]:.2f}",
        font_size=0.16,
        context_id="ocean_color_sliders",
        context_status=False,
    )
    gui_batched.add_slider(
        bg_texture_name="slider_background",
        knob_texture_name="button",
        position=(-0.60, y_pos),
        scale=(0.18, 0.025),
        min_value=0.0,
        max_value=1.0,
        value=sdf_ocean.color[i],
        callback=lambda val, comp=i: set_ocean_color_component(val, comp),
        context_id="ocean_color_sliders",
        context_status=False,
    )


# decouple fps from camera movement with time delta
time_start = 0
time_end = 1 / 60
time_delta = 1.0

def update_planet_lights(shader):
    glUniform1f(glGetUniformLocation(shader, "shininess"), 4.0)
    glUniform3fv(glGetUniformLocation(shader, "object_color"), 1, [0.15, 0.2, 0.25])
    glUniform3fv(glGetUniformLocation(shader, "view_pos"), 1, list(active_camera.camera_pos))

     # Pass the SDF ocean radius so the terrain shader can discard submerged fragments.
    # When the planet is not Earth (no ocean), pass 0 to disable discarding.
    if planet_settings.get('planet_type', '') == 'Earth':
        glUniform1f(glGetUniformLocation(shader, "sdf_radius"), sdf_ocean.radius)
    else:
        glUniform1f(glGetUniformLocation(shader, "sdf_radius"), 0.0)

    glUniform1i(glGetUniformLocation(shader, "draw_hydrosphere"), planet_settings.get("draw_hydrosphere", 1))
    glUniform3fv(glGetUniformLocation(shader, "planet_axis"), 1, list(PLANET_AXIS))
    glUniform1i(glGetUniformLocation(shader, "view_mode"), VIEW_MODE)
    glUniform1f(glGetUniformLocation(shader, "global_temperature"), GLOBAL_TEMPERATURE)
    glUniform1f(glGetUniformLocation(shader, "global_rainfall_reduction"), GLOBAL_RAINFALL_REDUCTION)

    glUniform3fv(glGetUniformLocation(shader, "point_light.position"), 1, my_plc.get_pos())

    glUniform3fv(glGetUniformLocation(shader, "point_light.ambient"), 1, my_plc.get_ambient())
    glUniform3fv(glGetUniformLocation(shader, "point_light.diffuse"), 1, my_plc.get_diffuse())
    glUniform3fv(glGetUniformLocation(shader, "point_light.specular"), 1, my_plc.get_specular())
    glUniform1fv(glGetUniformLocation(shader, "point_light.constant"), 1, my_plc.get_constant())
    glUniform1fv(glGetUniformLocation(shader, "point_light.linear"), 1, my_plc.get_linear())
    glUniform1fv(glGetUniformLocation(shader, "point_light.quadratic"), 1, my_plc.get_quadratic())

    # Bind biome map texture to unit 2 (units 0/1 used by ocean/atmosphere passes)
    glActiveTexture(GL_TEXTURE2)
    glBindTexture(GL_TEXTURE_2D, biome_tex)
    glUniform1i(glGetUniformLocation(shader, "biome_map"), 2)
    # Bind photo-realistic biome map to unit 3
    glActiveTexture(GL_TEXTURE3)
    glBindTexture(GL_TEXTURE_2D, biome_true_tex)
    glUniform1i(glGetUniformLocation(shader, "biome_true_map"), 3)
    glActiveTexture(GL_TEXTURE0)


def update_lights(shader):
    glUseProgram(shader)
    view_pos_loc = glGetUniformLocation(shader, "view_pos")
    glUniform3fv(view_pos_loc, 1, list(active_camera.camera_pos))

    #TODO: loop over list of point lights
    glUniform1f(glGetUniformLocation(shader, "material.shininess"), 4.0)
    glUniform3fv(glGetUniformLocation(shader, "point_lights[0].position"), 1, light_cubes[0].get_pos())
    glUniform3fv(glGetUniformLocation(shader, "point_lights[0].diffuse"), 1, light_cubes[0].get_diffuse())
    glUniform3fv(glGetUniformLocation(shader, "point_lights[0].ambient"), 1, light_cubes[0].get_ambient())
    glUniform3fv(glGetUniformLocation(shader, "point_lights[0].specular"), 1, light_cubes[0].get_specular())
    glUniform1f(glGetUniformLocation(shader, "point_lights[0].constant"), light_cubes[0].get_constant())
    glUniform1f(glGetUniformLocation(shader, "point_lights[0].linear"), light_cubes[0].get_linear())
    glUniform1f(glGetUniformLocation(shader, "point_lights[0].quadratic"), light_cubes[0].get_quadratic())
    # second light for mesh viewing
    # glUniform3fv(glGetUniformLocation(shader, "point_lights[1].position"), 1, debug_plcs[1].get_pos())
    # glUniform3fv(glGetUniformLocation(shader, "point_lights[1].diffuse"), 1, debug_plcs[1].get_diffuse())
    # glUniform3fv(glGetUniformLocation(shader, "point_lights[1].ambient"), 1, debug_plcs[1].get_ambient())
    # glUniform3fv(glGetUniformLocation(shader, "point_lights[1].specular"), 1, debug_plcs[1].get_specular())
    # glUniform1f(glGetUniformLocation(shader, "point_lights[1].constant"), debug_plcs[1].get_constant())
    # glUniform1f(glGetUniformLocation(shader, "point_lights[1].linear"), debug_plcs[1].get_linear())
    # glUniform1f(glGetUniformLocation(shader, "point_lights[1].quadratic"), debug_plcs[1].get_quadratic())
    # spotlight
    glUniform3fv(glGetUniformLocation(shader, "spot_light.position"), 1, list(active_camera.camera_pos))
    glUniform3fv(glGetUniformLocation(shader, "spot_light.direction"), 1, list(active_camera.camera_front))
    glUniform3fv(glGetUniformLocation(shader, "spot_light.diffuse"), 1, [0.0] * 3)
    glUniform3fv(glGetUniformLocation(shader, "spot_light.ambient"), 1, [0.0] * 3)
    glUniform3fv(glGetUniformLocation(shader, "spot_light.specular"), 1, [0.0] * 3)
    glUniform1f(glGetUniformLocation(shader, "spot_light.cut_off"), cos(radians(12.5)))
    glUniform1f(glGetUniformLocation(shader, "spot_light.outer_cut_off"), cos(radians(45.0)))
    glUniform1f(glGetUniformLocation(shader, "spot_light.constant"), 1.0)
    glUniform1f(glGetUniformLocation(shader, "spot_light.linear"), 0.00003)
    glUniform1f(glGetUniformLocation(shader, "spot_light.quadratic"), 0.00007)

    #Direction
    glUniform3fv(glGetUniformLocation(shader, "dir_light.direction"), 1, [20.0, 50.0, 0.0])
    glUniform3fv(glGetUniformLocation(shader, "dir_light.ambient"), 1,   [0.0]*3)
    glUniform3fv(glGetUniformLocation(shader, "dir_light.diffuse"), 1,   [0.0]*3)
    glUniform3fv(glGetUniformLocation(shader, "dir_light.specular"), 1,  [0.0]*3)


#Experiment with new meshes/shaders






shaders_lighting = [planet.shader_program]



line_test = Line(shader_program=shader_line,
                 start=vec3([0.0, 0.0, 0.0]),
                 end=vec3([0.0, 1000.0, 0.0]),
                 projection=projection)


# ---- Offscreen FBO used only for the Earth ocean post-process ----------
# The SDF ocean shader needs scene colour + depth as textures so it can
# composite water on top.  Non-Earth planets render directly to the
# default framebuffer and never touch this FBO.
scene_fbo = glGenFramebuffers(1)
glBindFramebuffer(GL_FRAMEBUFFER, scene_fbo)

scene_color_tex = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, scene_color_tex)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, WIDTH, HEIGHT, 0,
             GL_RGBA, GL_FLOAT, None)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, scene_color_tex, 0)

scene_depth_tex = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, scene_depth_tex)
glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, WIDTH, HEIGHT, 0,
             GL_DEPTH_COMPONENT, GL_FLOAT, None)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, scene_depth_tex, 0)

if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
    raise RuntimeError("Scene FBO is not complete!")
glBindFramebuffer(GL_FRAMEBUFFER, 0)
glBindTexture(GL_TEXTURE_2D, 0)

# ---- Second FBO: captures ocean pass output so atmosphere can layer on top --
ocean_fbo = glGenFramebuffers(1)
glBindFramebuffer(GL_FRAMEBUFFER, ocean_fbo)

ocean_color_tex = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, ocean_color_tex)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, WIDTH, HEIGHT, 0,
             GL_RGBA, GL_FLOAT, None)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ocean_color_tex, 0)

# Reuse the same depth texture — the ocean shader doesn't write depth
glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, scene_depth_tex, 0)

if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
    raise RuntimeError("Ocean FBO is not complete!")
glBindFramebuffer(GL_FRAMEBUFFER, 0)
glBindTexture(GL_TEXTURE_2D, 0)

# -----------------------------------------------------------------------
def render_scene(view):
    """Draw skybox, planet, lights — used by both Earth and non-Earth paths."""
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glDepthFunc(GL_LEQUAL)
    skybox.draw(view=view, projection=projection)
    glDepthFunc(GL_LESS)

    for shader in shaders_lighting:
        update_lights(shader=shader)
    update_planet_lights(shader=shader_terrain_gpu)

    planet.draw(view_matrix=view)

    if DRAW_DEBUG_LINES:
        planet.draw_lines(line_shader_program=shader_line, view_matrix=view, camera_position=active_camera.camera_pos)

    for light in light_cubes:
        light.draw(view=view)

# -----------------------------------------------------------------------
def update_near_plane():
    """Adjust NEAR_PLANE based on camera distance to planet surface."""
    global NEAR_PLANE, projection
    cam = active_camera.camera_pos
    dist = np.linalg.norm(np.array([cam.x, cam.y, cam.z]) -
                          np.array([planet.position.x, planet.position.y, planet.position.z]))
    surface_dist = max(0.0, dist - planet.scale)

    # Smoothly interpolate: when within 2× radius, start decreasing near plane
    threshold = planet.scale * 2.0
    if surface_dist < threshold:
        t = surface_dist / threshold  # 0 at surface, 1 at threshold
        NEAR_PLANE = NEAR_PLANE_MIN + t * (NEAR_PLANE_MAX - NEAR_PLANE_MIN)
    else:
        NEAR_PLANE = NEAR_PLANE_MAX

    projection = pyrr.matrix44.create_perspective_projection_matrix(
        45, WIDTH / HEIGHT, NEAR_PLANE, DRAW_DISTANCE)
    planet.projection = projection
    sdf_ocean.near_plane = NEAR_PLANE
    sdf_atmosphere.near_plane = NEAR_PLANE

# -----------------------------------------------------------------------
while not glfw.window_should_close(window):

    glfw.poll_events()
    time_start = glfw.get_time()
    do_movement(speed=camera_speed * (time_delta))

    update_near_plane()

    view = active_camera.get_view_matrix()

    if use_sim_cam and DRAW_GUI:
        gui_batched.update(
            mouse_pos_pixels=glfw.get_cursor_pos(window),
            left_click=(glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS),
        )

    is_earth = planet_settings.get('planet_type', '') == 'Earth'

    if is_earth:
        # Pass 1: Render opaque scene into FBO
        glBindFramebuffer(GL_FRAMEBUFFER, scene_fbo)
        render_scene(view)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Pass 2: Composite ocean over scene -> ocean FBO
        glBindFramebuffer(GL_FRAMEBUFFER, ocean_fbo)
        glClear(GL_COLOR_BUFFER_BIT)
        
        # Set ocean transparency to 0 in Biome or Biome True view modes
        ocean_transparency_backup = sdf_ocean.transparency
        if VIEW_MODE == 4 or VIEW_MODE == 5:
            sdf_ocean.transparency = 0.0
        
        sdf_ocean.draw(
            view=view,
            projection=projection,
            camera_pos=active_camera.camera_pos,
            depth_texture=scene_depth_tex,
            scene_color_texture=scene_color_tex,
            screen_size=(WIDTH, HEIGHT),
        )
        
        # Restore ocean transparency
        sdf_ocean.transparency = ocean_transparency_backup
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Pass 3: Composite atmosphere over ocean result -> default framebuffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        sdf_atmosphere.draw(
            view=view,
            projection=projection,
            camera_pos=active_camera.camera_pos,
            depth_texture=scene_depth_tex,
            scene_color_texture=ocean_color_tex,
            screen_size=(WIDTH, HEIGHT),
        )
    else:
        # Non-Earth: render directly to the default framebuffer
        render_scene(view)

    #what is this?
    planet.update_lod(target_position=active_camera.camera_pos)



    if use_sim_cam and DRAW_GUI:
        gui_batched.draw()

    if WRITE_TO_GIF:
        write_fbo_to_gif(width=WIDTH, height=HEIGHT)

    glfw.swap_buffers(window)
    time_end = glfw.get_time()
    time_delta = time_end - time_start


if WRITE_TO_GIF:
    save_to_gif()
glfw.terminate()
