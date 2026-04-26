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
from engine.camera import Camera, SimulationCamera, RollableCamera
from engine.skybox import Skybox
from math import sin, cos
from glm import cos, radians
import logging
import engine.point_light_cube as plc
from engine.procedural_mesh import CubeMeshStatic
from engine.gui import GUI
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
# WIDTH, HEIGHT = 1728, 972
WIDTH, HEIGHT = 400, 200
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
    left_click = button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS
    right_click = button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS
    gui.button_update(position_mouse=glfw.get_cursor_pos(window), left_click=left_click, right_click=right_click)


def window_resize_clb(window, width, height):
    glViewport(0, 0, width, height)
    # projection = pyrr.matrix44.create_perspective_projection_matrix(45, width / height, 0.1, 2000)
    # glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    gui.set_screen_size(screen_size=(width, height))


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

texture_dictionary = {
    "button_atlas": textures[0],
    "font_atlas": textures[1],
    "banana": textures[2],
    "cloud": textures[3],
}

"""Shader Compilation"""
#New shader for position and normal only,with 3d camera
shader_program_pos_normal = create_shader(vertex_file='engine/shaders/pos_norm.vs', fragment_file='engine/shaders/pos_norm.fs')
# shader_point_light = create_shader(vertex_file='engine/shaders/pos_norm.vs', fragment_file='engine/shaders/pos_norm_lit.fs')
shader_point_light = create_shader(vertex_file='engine/shaders/pos_norm.vs', fragment_file='engine/shaders/terrain_height_coloring.fs')
shader_terrain_gpu = create_shader(vertex_file='engine/shaders/terrain_planet.vs', fragment_file='engine/shaders/terrain_coloring_gpu.fs')
shader_line = create_shader(vertex_file='engine/shaders/line.vs', fragment_file='engine/shaders/line.fs')
# shader_cloud = create_shader(vertex_file='engine/shaders/cloud_sphere.vs', fragment_file='engine/shaders/cloud_sphere.fs')


projection = pyrr.matrix44.create_perspective_projection_matrix(45, WIDTH / HEIGHT, NEAR_PLANE, DRAW_DISTANCE)

"""GUI CREATION"""
gui = GUI(screen_size=(WIDTH, HEIGHT))

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

def cycle_planet_type():
    global planet_settings
    planet_types = ["Moon", "Earth", "Mars"]
    current_index = planet_types.index(planet_settings["planet_type"])
    new_index = (current_index + 1) % len(planet_types)
    planet_settings["planet_type"] = planet_types[new_index]
    print(f"New planet type: {planet_settings['planet_type']}")
    planet.set_planet_type(planet_settings["planet_type"])

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

# create a skybox using banana texture on all six faces
#use the nebula textures for a more interesting skybox
skybox_paths = ["engine/textures/nebula/skybox_left.png", "engine/textures/nebula/skybox_right.png", "engine/textures/nebula/skybox_up.png",] \
                + ["engine/textures/nebula/skybox_down.png", "engine/textures/nebula/skybox_front.png", "engine/textures/nebula/skybox_back.png"]
skybox = Skybox(skybox_paths, scale=10000)



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


gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.10,0.05),
    position=(.85,-0.95),
    text="Change Planet Type",
    font_size=0.2,
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=cycle_planet_type,
)

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.10,0.05),
    position=(-0.85,-0.95),
    text="Octaves +",
    font_size=0.2,
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta = +1,
    setting = "octaves",
)
gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.10,0.05),
    position=(-0.85,-0.85),
    text="Octaves -",
    font_size=0.2,
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta = -1,
    setting = "octaves",
)

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.20,0.05),
    position=(-0.55,0.75),
    text="Lacunarity -",
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta = -0.05,
    setting = "lacunarity",
)

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.20,0.05),
    position=(-0.55,0.85),
    text="Lacunarity +",
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta = 0.05,
    setting = "lacunarity",
)

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.20,0.05),
    position=(-0.55,0.75),
    text="Lacunarity -",
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta = -0.05,
    setting = "lacunarity",
)

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.20,0.05),
    position=(-0.25,0.85),
    text="frequency +",
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta = +0.01,
    setting = "frequency",
)

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.20,0.05),
    position=(-0.25,0.75),
    text="frequency -",
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta = -0.01,
    setting = "frequency",
)
gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.20,0.05),
    position=(-0.05,0.85),
    text="amplitude +",
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta = +0.25,
    setting = "amplitude",
)

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.20,0.05),
    position=(-0.05,0.75),
    text="amplitude -",
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta = -0.25,
    setting = "amplitude",
)

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.20,0.05),
    position=(0.25,0.85),
    text="gain +",
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta = +0.05,
    setting = "gain",
)

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.20,0.05),
    position=(0.25,0.75),
    text="gain -",
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta = -0.05,
    setting = "gain",
)

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.15,0.03),
    position=(0.85,0.95),
    text="Next Seed",
    context_id="button_1",
    atlas_size=2,
    font_size = 0.20,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta=1,
    setting="seed",
)



def change_sdf_radius(delta=5.0):
    sdf_ocean.radius = max(1.0, sdf_ocean.radius + delta)
    print(f"SDF radius: {sdf_ocean.radius:.1f}")

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.15, 0.05),
    position=(-0.45, -0.75),
    text="SDF radius +",
    font_size=0.2,
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0, 0),
    click_function=change_sdf_radius,
    delta=0.125,
)
gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.15, 0.05),
    position=(-0.25, -0.75),
    text="SDF radius -",
    font_size=0.2,
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0, 0),
    click_function=change_sdf_radius,
    delta=-0.125,
)

def change_sdf_max_depth(delta=5.0):
    sdf_ocean.max_depth = max(1.0, sdf_ocean.max_depth + delta)
    print(f"SDF max_depth: {sdf_ocean.max_depth:.1f}")

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.15, 0.05),
    position=(-0.45, -0.85),
    text="max_depth +",
    font_size=0.2,
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0, 0),
    click_function=change_sdf_max_depth,
    delta=5.0,
)
gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.15, 0.05),
    position=(-0.25, -0.85),
    text="max_depth -",
    font_size=0.2,
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0, 0),
    click_function=change_sdf_max_depth,
    delta=-5.0,
)

def change_sdf_transparency(delta=0.05):
    sdf_ocean.transparency = max(0.0, min(1.0, sdf_ocean.transparency + delta))
    print(f"SDF transparency: {sdf_ocean.transparency:.2f}")

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.15, 0.05),
    position=(-0.45, -0.95),
    text="transparency +",
    font_size=0.2,
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0, 0),
    click_function=change_sdf_transparency,
    delta=0.05,
)
gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.15, 0.05),
    position=(-0.25, -0.95),
    text="transparency -",
    font_size=0.2,
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0, 0),
    click_function=change_sdf_transparency,
    delta=-0.05,
)

def toggle_hydrosphere():
    planet_settings["draw_hydrosphere"] = 1 - planet_settings["draw_hydrosphere"]
    state = "ON" if planet_settings["draw_hydrosphere"] else "OFF"
    print(f"Hydrosphere: {state}")

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.15, 0.05),
    position=(-0.05, -0.75),
    text="Hydrosphere",
    font_size=0.2,
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0, 0),
    click_function=toggle_hydrosphere,
)

def set_view_normals():
    global VIEW_MODE
    VIEW_MODE = 0 if VIEW_MODE == 1 else 1
    names = {0: "Terrain", 1: "Normals", 2: "Heat"}
    print(f"View mode: {names[VIEW_MODE]}")

def set_view_heat():
    global VIEW_MODE
    VIEW_MODE = 0 if VIEW_MODE == 2 else 2
    names = {0: "Terrain", 1: "Normals", 2: "Heat"}
    print(f"View mode: {names[VIEW_MODE]}")

def change_global_temperature(delta=0.05):
    global GLOBAL_TEMPERATURE
    GLOBAL_TEMPERATURE += delta
    print(f"Global temperature: {GLOBAL_TEMPERATURE:.2f}")

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.15, 0.05),
    position=(0.15, -0.75),
    text="View Normals",
    font_size=0.2,
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0, 0),
    click_function=set_view_normals,
)
gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.15, 0.05),
    position=(0.35, -0.75),
    text="View Heat",
    font_size=0.2,
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0, 0),
    click_function=set_view_heat,
)
gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.15, 0.05),
    position=(0.15, -0.85),
    text="Temp +",
    font_size=0.2,
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0, 0),
    click_function=change_global_temperature,
    delta=0.05,
)
gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.15, 0.05),
    position=(0.35, -0.85),
    text="Temp -",
    font_size=0.2,
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0, 0),
    click_function=change_global_temperature,
    delta=-0.05,
)

# must call as final setup of GUI
gui.build_elements_list()

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

    glUniform3fv(glGetUniformLocation(shader, "point_light.position"), 1, my_plc.get_pos())

    glUniform3fv(glGetUniformLocation(shader, "point_light.ambient"), 1, my_plc.get_ambient())
    glUniform3fv(glGetUniformLocation(shader, "point_light.diffuse"), 1, my_plc.get_diffuse())
    glUniform3fv(glGetUniformLocation(shader, "point_light.specular"), 1, my_plc.get_specular())
    glUniform1fv(glGetUniformLocation(shader, "point_light.constant"), 1, my_plc.get_constant())
    glUniform1fv(glGetUniformLocation(shader, "point_light.linear"), 1, my_plc.get_linear())
    glUniform1fv(glGetUniformLocation(shader, "point_light.quadratic"), 1, my_plc.get_quadratic())


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
    # max_depth=PLANET_RADIUS * 0.80,
    max_depth=PLANET_RADIUS * 0.40,
    near_plane=NEAR_PLANE,
    far_plane=DRAW_DISTANCE,
)

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

    is_earth = planet_settings.get('planet_type', '') == 'Earth'

    if is_earth:
        # Pass 1: Render opaque scene into FBO
        glBindFramebuffer(GL_FRAMEBUFFER, scene_fbo)
        render_scene(view)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Pass 2: Composite ocean over scene -> ocean FBO
        glBindFramebuffer(GL_FRAMEBUFFER, ocean_fbo)
        glClear(GL_COLOR_BUFFER_BIT)
        sdf_ocean.draw(
            view=view,
            projection=projection,
            camera_pos=active_camera.camera_pos,
            depth_texture=scene_depth_tex,
            scene_color_texture=scene_color_tex,
            screen_size=(WIDTH, HEIGHT),
        )
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
        gui.draw()

    if WRITE_TO_GIF:
        write_fbo_to_gif(width=WIDTH, height=HEIGHT)

    glfw.swap_buffers(window)
    time_end = glfw.get_time()
    time_delta = time_end - time_start


if WRITE_TO_GIF:
    save_to_gif()
glfw.terminate()
