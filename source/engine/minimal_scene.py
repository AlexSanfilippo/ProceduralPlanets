"""
A simple script that creates a window with OpenGL context and renders a basic scene.
It sets up camera controls, lighting, and a GUI overlay.
"""


import glfw
from OpenGL.GL import *
import pyrr
from texture_loader import load_texture
from camera import Camera, SimulationCamera
from math import sin, cos
from glm import cos, radians
import logging
import point_light_cube as plc
from procedural_mesh import CubeMeshStatic
from gui import GUI
from screen_capture import capture_screenshot, write_fbo_to_gif, save_to_gif

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
WINDOW_POSITION = (40, 40)
WRITE_TO_GIF = False
lastX, lastY = WIDTH / 2, HEIGHT / 2

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
        pause, up, down, wrote_to_gif, switch_view_mode

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


# def mouse_look_clb(window, xpos, ypos):
#     global first_mouse, WIDTH, HEIGHT
#     lastX, lastY = WIDTH / 2, HEIGHT / 2
#     if first_mouse:
#         lastX = xpos
#         lastY = ypos
#         first_mouse = False
#     xoffset = xpos - lastX
#     yoffset = lastY - ypos  # reversed since y-coordinates go from bottom to top
#     lastX = xpos
#     lastY = ypos
#     active_camera.process_mouse_movement(xoffset, yoffset)


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
        sim_cam.process_keyboard("YAW_CLOCKWISE", speed*2)
    if yaw_counterclockwise:
        sim_cam.process_keyboard("YAW_COUNTERCLOCKWISE", speed*2)
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
sim_cam = SimulationCamera(camera_pos=[6.0, 60.0, 42.0])
cam = Camera(camera_pos=[0.0, 20.0, 20.0], mouse_sensitivity=0.1)
use_sim_cam = False
glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
active_camera = cam

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
my_plc = plc.PointLightCube(pos=[45.0, 35.0, 0.0], ambient=[0.1] * 3, diffuse=[0.5] * 3,
                            specular=[0.95] * 3, linear=0.000014, quadratic=.0000007)
light_cubes = [my_plc]


textures = glGenTextures(12)
load_texture("textures/button_atlas_gradient.png", textures[0])
load_texture("fonts/my_font.png", textures[1])
load_texture("textures/banana.png", textures[2])

texture_dictionary = {
    "button_atlas": textures[0],
    "font_atlas": textures[1],
    "banana": textures[2],
}

projection = pyrr.matrix44.create_perspective_projection_matrix(45, WIDTH / HEIGHT, 0.1, 2000)
cube_default = CubeMeshStatic(
    diffuse=texture_dictionary["banana"],
    specular=texture_dictionary["banana"],
    shininess=32.0,
    dimensions=(10.0, 10.0, 10.0),
    position=(0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
    rotation_axis=(0.5, 0.25, 0.88),
    rotation_magnitude=(0, 0, 0),
    projection=projection,
)




"""GUI CREATION"""
gui = GUI(screen_size=(WIDTH, HEIGHT))

def speak():
    print("Hello World Button Pressed!")

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.15,0.15),
    position=(-0.85,0.85),
    text="Hello World",
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=speak,

)
# must call as final setup of GUI
gui.build_elements_list()

# decouple fps from camera movement with time delta
time_start = 0
time_end = 1 / 60
time_delta = 1.0


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



shaders_lighting = [cube_default.shader]

while not glfw.window_should_close(window):

    glfw.poll_events()
    time_start = glfw.get_time()
    do_movement(speed=25 * (time_delta))
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


    view = active_camera.get_view_matrix()

    #move the point light cube around
    old_pos = light_cubes[0].get_pos()
    new_pos = [20 * cos(glfw.get_time()), old_pos[1], 20 * sin(glfw.get_time())]
    light_cubes[0].set_pos(pos=new_pos)

    # draw the scene
    cube_default.draw(view=view)
    for shader in shaders_lighting:
        update_lights(shader=shader)


    for light in light_cubes:
        light.draw(view=view)
    gui.draw()

    if use_sim_cam:
        gui.draw()

    if WRITE_TO_GIF:
        write_fbo_to_gif(width=WIDTH, height=HEIGHT)

    glfw.swap_buffers(window)
    time_end = glfw.get_time()
    time_delta = time_end - time_start


if WRITE_TO_GIF:
    save_to_gif()
glfw.terminate()
