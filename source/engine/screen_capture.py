"""
Tools for capturing screenshots and screen recordings.
"""
from datetime import datetime
from OpenGL.GL import glReadPixels
from OpenGL.raw.GL.VERSION.GL_1_0 import GL_RGB, GL_UNSIGNED_BYTE
from PIL import ImageOps, Image


images = []

def capture_screenshot(width, height):
    filename = datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".jpg"
    print(f'saving screenshot as {filename}')
    fbo_array = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    screenshot = Image.frombytes("RGB", (width, height), fbo_array)
    screenshot = ImageOps.flip(screenshot)
    screenshot.save(filename)

def write_fbo_to_gif(width, height):
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (width, height), data)
    image = ImageOps.flip(image)
    images.append(image)

def save_to_gif(name=f'screen_recording.gif'):
    print(f'saving gif as {name}')
    images[0].save(
        name,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=20,
        loop=0
    )