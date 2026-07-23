"""
GUIBatched – minimal single-draw-call 2-D GUI layer.

Vertex layout (per vertex, 5 floats):
    x, y        – NDC position  (-1 … +1)
    u, v        – texture coords (0 … 1), derived from atlas_map pixel coords
    hover       – float 0.0 or 1.0  (set per-element each frame for shader hover)

Each quad = 4 vertices; indices = 6 per quad (two triangles).
Everything lives in one VAO/VBO/EBO; the buffer is rebuilt with
glBufferSubData whenever hover state changes or new elements are added.
"""

import ctypes
import numpy as np
from PIL import Image
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

# ── inline shaders ──────────────────────────────────────────────────────────────
_VERT_SRC = """
#version 330 core
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in float a_hover;

out vec2 v_uv;
out float v_hover;

void main() {
    v_uv    = a_uv;
    v_hover = a_hover;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

_FRAG_SRC = """
#version 330 core
in vec2  v_uv;
in float v_hover;

out vec4 frag_color;

uniform sampler2D u_atlas;

void main() {
    vec4 col = texture(u_atlas, v_uv);
    // Brighten slightly on hover
    col.rgb *= mix(1.0, 1.25, step(0.5, v_hover));
    frag_color = col;
}
"""

_FLOATS_PER_VERTEX = 5   # x, y, u, v, hover
_VERTS_PER_QUAD    = 4
_INDICES_PER_QUAD  = 6


class TextHandle:
    """
    A lightweight reference to a single text chunk inside a GUIBatched instance.

    Returned by add_text_element (and optionally add_text_button).
    Call update_text(new_text) at any time to replace the displayed string and
    immediately refresh the GPU text vertex buffer — the same live-update
    pattern used by gui.Element.update_text() in the old system.
    """

    def __init__(self, gui_instance, chunk_index: int):
        self._gui   = gui_instance
        self._index = chunk_index

    def update_text(self, text: str):
        """Replace the text string and rebuild the GPU text vertex buffer."""
        self._gui._update_text_chunk(self._index, text)


class GUIBatched:
    """
    Parameters
    ----------
    screen_size       : (width, height) in pixels – reserved for future hit-testing.
    texture_atlas     : OpenGL texture id (already uploaded).
    atlas_map         : dict[str, ((x0,y0),(x1,y1))]
                        Maps a name to top-left / bottom-right pixel coords inside
                        the atlas image (y=0 is the TOP of the image as stored on disk).
    atlas_image_path  : path to the atlas PNG – used only to read (width, height)
                        so UV coords can be normalised.
    """

    def __init__(self, screen_size, texture_atlas, atlas_map, atlas_image_path,
                 font_texture=None, font_fnt_path=None):
        self.screen_size   = screen_size
        self.texture_atlas = texture_atlas
        self.atlas_map     = atlas_map

        # Read atlas dimensions with PIL (no extra GL call needed)
        with Image.open(atlas_image_path) as img:
            self._atlas_w = img.width
            self._atlas_h = img.height

        # Compile the dedicated shader
        self._shader = compileProgram(
            compileShader(_VERT_SRC, GL_VERTEX_SHADER),
            compileShader(_FRAG_SRC, GL_FRAGMENT_SHADER),
        )
        self._u_atlas = glGetUniformLocation(self._shader, "u_atlas")

        # CPU-side storage (grown as elements are added)
        self._vertices = np.empty((0,), dtype=np.float32)  # flat array
        self._indices  = np.empty((0,), dtype=np.uint32)

        # Per-element hover state (index → bool); updated each frame
        self._element_count = 0
        self._hover         = []   # list[bool], one per element

        # Button records: list of dicts with element_index, bounds, click_function
        # bounds: (left, right, bottom, top) in NDC
        self._buttons = []

        # Slider records
        self._sliders = []

        # ── Context system (mirrors gui.GUI context on/off behaviour) ─────────
        # Every element, button, slider and text chunk carries a context_id.
        # switch_context_status / toggle_context_status show/hide whole groups
        # by rebuilding the EBO (quad draw call) and the text vertex buffer so
        # that hidden items are excluded from both rendering *and* hit-testing.
        self._element_visible  = []   # list[bool], one entry per quad element
        self._context_status   = {}   # context_id -> bool  (True = visible)
        self._context_elements = {}   # context_id -> [element_indices]
        self._text_chunks      = []   # [(context_id, np.ndarray)]  per text call

        # ── Font / text rendering ──────────────────────────────────────────────
        self._font_texture = font_texture
        self._font_chars   = {}
        if font_texture is not None and font_fnt_path is not None:
            self._font_chars = self._parse_fnt(font_fnt_path)

        # Text geometry: flat array of 5-float GL_TRIANGLES vertices (no index buf)
        # 6 vertices × 5 floats per character glyph quad.
        self._text_vertices  = np.empty((0,), dtype=np.float32)
        self._text_vao       = glGenVertexArrays(1)
        self._text_vbo       = glGenBuffers(1)
        self._setup_text_vao()
        self._text_dirty     = False

        # Standalone texture elements (each has its own texture, not from the atlas)
        self._texture_elements = []   # list of dicts: texture_id, vao, context_id, visible

        # GPU buffers
        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)
        self._ebo = glGenBuffers(1)

        self._setup_vao()
        self._dirty = False   # True when hover state has changed

    # ── public API ──────────────────────────────────────────────────────────────

    def add_element(self, texture_name: str, position: tuple, scale: tuple,
                    rotate_90_cw: bool = False,
                    context_id: str = 'default', context_status: bool = True):
        """
        Add a textured quad to the batch.

        Parameters
        ----------
        texture_name   : key into atlas_map
        position       : (cx, cy) in NDC, centre of the quad
        scale          : (half_w, half_h) in NDC  (e.g. (0.08, 0.05))
        rotate_90_cw   : if True, rotate the texture 90 degrees clockwise on the quad.
                         Achieved by cycling the UV assignments one step clockwise:
                         vertex  normal UVs       rotated-90-CW UVs
                         TL      (u0, v1)    →    (u1, v1)   ← was TR
                         BL      (u0, v0)    →    (u0, v1)   ← was TL
                         BR      (u1, v0)    →    (u0, v0)   ← was BL
                         TR      (u1, v1)    →    (u1, v0)   ← was BR
        context_id     : name of the visibility group this element belongs to
        context_status : initial visibility of the context (only applied when the
                         context is first created; ignored for existing contexts)
        """
        if texture_name not in self.atlas_map:
            raise KeyError(f"GUIBatched: texture name '{texture_name}' not in atlas_map")

        cx, cy = position
        hw, hh = scale
        (px0, py0), (px1, py1) = self.atlas_map[texture_name]

        # Normalise pixel coords to [0,1].
        # load_texture flips the image vertically (FLIP_TOP_BOTTOM), so
        # pixel y=0 (top of image on disk) → after flip → V = 1.0
        # pixel y=H (bottom of image on disk) → after flip → V = 0.0
        u0 = px0 / self._atlas_w
        u1 = px1 / self._atlas_w
        v0 = 1.0 - py1 / self._atlas_h   # atlas bottom pixel row  → lower V after flip
        v1 = 1.0 - py0 / self._atlas_h   # atlas top pixel row     → higher V after flip

        hover = 0.0

        if rotate_90_cw:
            # Each vertex's UV is the one that was 90° counter-clockwise of it
            # in the original layout, which visually rotates the texture 90° CW.
            # TL←TR, BL←TL, BR←BL, TR←BR
            quad_verts = np.array([
                cx - hw,  cy + hh,  u1, v1, hover,   # top-left     ← TR uv
                cx - hw,  cy - hh,  u0, v1, hover,   # bottom-left  ← TL uv
                cx + hw,  cy - hh,  u0, v0, hover,   # bottom-right ← BL uv
                cx + hw,  cy + hh,  u1, v0, hover,   # top-right    ← BR uv
            ], dtype=np.float32)
        else:
            # 4 vertices: top-left, bottom-left, bottom-right, top-right
            quad_verts = np.array([
                cx - hw,  cy + hh,  u0, v1, hover,   # top-left
                cx - hw,  cy - hh,  u0, v0, hover,   # bottom-left
                cx + hw,  cy - hh,  u1, v0, hover,   # bottom-right
                cx + hw,  cy + hh,  u1, v1, hover,   # top-right
            ], dtype=np.float32)

        self._vertices = np.concatenate([self._vertices, quad_verts])
        self._hover.append(False)
        self._element_visible.append(True)   # register before context call sets it
        self._element_count += 1
        self._register_to_context(context_id, context_status, self._element_count - 1)

        self._upload_vertex_buffer()
        self._rebuild_and_upload_indices()

    def set_hover(self, element_index: int, hovered: bool):
        """Update hover state for a single element; marks buffer dirty."""
        if self._hover[element_index] != hovered:
            self._hover[element_index] = hovered
            # Patch the hover float in-place in the CPU array for all 4 verts
            for v in range(_VERTS_PER_QUAD):
                idx = (element_index * _VERTS_PER_QUAD + v) * _FLOATS_PER_VERTEX + 4
                self._vertices[idx] = 1.0 if hovered else 0.0
            self._dirty = True

    def add_button(self, texture_name: str, position: tuple, scale: tuple,
                   click_function, rotate_90_cw: bool = False,
                   context_id: str = 'default', context_status: bool = True):
        """
        Add a clickable button quad to the batch.

        Identical to add_element but also registers a collision box and a
        click_function.  Call update() every frame with the current mouse
        position and click state to trigger hover highlighting and clicks.

        Parameters
        ----------
        texture_name   : key into atlas_map
        position       : (cx, cy) NDC centre of the quad
        scale          : (half_w, half_h) NDC half-extents
        click_function : callable – called with no arguments when the button
                         is left-clicked while the cursor is inside it
        rotate_90_cw   : rotate texture 90° clockwise (same as add_element)
        context_id     : visibility group this button belongs to
        context_status : initial visibility of the context (first creation only)
        """
        element_index = self._element_count   # index before add_element increments it
        self.add_element(texture_name, position, scale, rotate_90_cw=rotate_90_cw,
                         context_id=context_id, context_status=context_status)

        cx, cy = position
        hw, hh = scale
        self._buttons.append({
            "element_index": element_index,
            "context_id":    context_id,
            "left":   cx - hw,
            "right":  cx + hw,
            "bottom": cy - hh,
            "top":    cy + hh,
            "click_function": click_function,
            "was_down": False,   # tracks previous frame's left_click to detect press edge
        })

    def add_slider(self, bg_texture_name: str, knob_texture_name: str,
                   position: tuple, scale: tuple,
                   min_value: float, max_value: float, value: float,
                   callback, rotate_90_cw: bool = False,
                   context_id: str = 'default', context_status: bool = True):
        """
        Add a horizontal slider to the batch.

        Two quads are added: a background track and a draggable knob.
        The knob brightens on hover (same shader hover flag as buttons).
        Dragging the knob (or clicking anywhere on the track) updates the
        value and calls callback(new_value).

        Parameters
        ----------
        bg_texture_name   : atlas_map key for the track background quad
        knob_texture_name : atlas_map key for the knob quad
        position          : (cx, cy) NDC centre of the track
        scale             : (half_w, half_h) NDC half-extents of the track
        min_value         : minimum float value of the slider
        max_value         : maximum float value of the slider
        value             : initial value (clamped to [min_value, max_value])
        callback          : callable(float) – called whenever the value changes
        rotate_90_cw      : rotate both textures 90° CW
        context_id        : visibility group this slider belongs to
        context_status    : initial visibility of the context (first creation only)
        """
        cx, cy = position
        hw, hh = scale

        value = max(min_value, min(max_value, float(value)))

        # --- background track ---
        bg_index = self._element_count
        self.add_element(bg_texture_name, position, scale, rotate_90_cw=rotate_90_cw,
                         context_id=context_id, context_status=context_status)

        # --- knob: slightly taller than the track, same width as the track height ---
        knob_hw = hh * 1.2
        knob_hh = hh * 1.4
        t = (value - min_value) / (max_value - min_value)
        knob_cx = (cx - hw) + t * (hw * 2.0)

        knob_index = self._element_count
        self.add_element(knob_texture_name, (knob_cx, cy), (knob_hw, knob_hh),
                         rotate_90_cw=rotate_90_cw,
                         context_id=context_id, context_status=context_status)

        self._sliders.append({
            "bg_index":    bg_index,
            "knob_index":  knob_index,
            "context_id":  context_id,
            "cx": cx, "cy": cy,
            "left":   cx - hw,
            "right":  cx + hw,
            "bottom": cy - hh,
            "top":    cy + hh,
            "knob_hw": knob_hw,
            "knob_hh": knob_hh,
            "min_value": min_value,
            "max_value": max_value,
            "value":     value,
            "callback":  callback,
            "is_dragging": False,
            "was_down":    False,
        })

    def add_text_element(self, texture_name: str, position: tuple, scale: tuple,
                         text: str, font_size: float = 0.5,
                         rotate_90_cw: bool = False,
                         context_id: str = 'default', context_status: bool = True):
        """
        Add a textured quad with centred text rendered on top of it.

        Parameters
        ----------
        texture_name   : key into atlas_map (background quad texture)
        position       : (cx, cy) NDC centre
        scale          : (half_w, half_h) NDC half-extents
        text           : string to render
        font_size      : multiplier on glyph size  (default 0.5 works well for
                         typical button sizes of ~0.08–0.15 half-width)
        rotate_90_cw   : rotate background texture 90° CW
        context_id     : visibility group this element belongs to
        context_status : initial visibility of the context (first creation only)
        """
        self.add_element(texture_name, position, scale, rotate_90_cw=rotate_90_cw,
                         context_id=context_id, context_status=context_status)
        chunk_index = self._append_text(text, position, scale, font_size, context_id=context_id)
        return TextHandle(self, chunk_index) if chunk_index is not None else None

    def add_text_button(self, texture_name: str, position: tuple, scale: tuple,
                        click_function, text: str, font_size: float = 0.5,
                        rotate_90_cw: bool = False,
                        context_id: str = 'default', context_status: bool = True):
        """
        Add a clickable button with centred text rendered on top of it.
        Supports hover brightening on the background quad.

        Parameters
        ----------
        texture_name   : key into atlas_map (background quad texture)
        position       : (cx, cy) NDC centre
        scale          : (half_w, half_h) NDC half-extents
        click_function : callable() – called on leading-edge click
        text           : string to render centred on the button
        font_size      : multiplier on glyph size
        rotate_90_cw   : rotate background texture 90° CW
        context_id     : visibility group this button belongs to
        context_status : initial visibility of the context (first creation only)
        """
        self.add_button(texture_name, position, scale, click_function,
                        rotate_90_cw=rotate_90_cw,
                        context_id=context_id, context_status=context_status)
        self._append_text(text, position, scale, font_size, context_id=context_id)

    def add_texture_element(self, texture_id: int, position: tuple, scale: tuple,
                            context_id: str = 'default', context_status: bool = True):
        """
        Add a quad textured with an arbitrary OpenGL texture (not from the atlas).

        Unlike add_element / add_text_element these quads are rendered in a
        separate pass in draw() with the provided texture bound directly, so
        any GL texture handle can be used.

        Parameters
        ----------
        texture_id     : OpenGL texture handle to draw on the quad.
        position       : (cx, cy) NDC centre of the quad.
        scale          : (half_w, half_h) NDC half-extents.
        context_id     : visibility group this element belongs to.
        context_status : initial visibility of the context (first creation only).
        """
        cx, cy = position
        hw, hh = scale

        # Full UV coverage (0,0) → (1,1); hover channel always 0
        verts = np.array([
            cx - hw,  cy + hh,  0.0, 1.0, 0.0,   # top-left
            cx - hw,  cy - hh,  0.0, 0.0, 0.0,   # bottom-left
            cx + hw,  cy - hh,  1.0, 0.0, 0.0,   # bottom-right
            cx + hw,  cy + hh,  1.0, 1.0, 0.0,   # top-right
        ], dtype=np.float32)

        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        # Register context if not yet known
        if context_id not in self._context_status:
            self._context_status[context_id] = context_status
            self._context_elements[context_id] = []
        visible = self._context_status[context_id]

        # Build a dedicated VAO/VBO/EBO for this quad
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        ebo = glGenBuffers(1)

        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        stride = _FLOATS_PER_VERTEX * ctypes.sizeof(ctypes.c_float)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16))
        glEnableVertexAttribArray(2)
        glBindVertexArray(0)

        self._texture_elements.append({
            'texture_id': texture_id,
            'vao':        vao,
            'context_id': context_id,
            'visible':    visible,
        })

    def update(self, mouse_pos_pixels: tuple, left_click: bool):
        """
        Call once per frame to process hover highlighting, button clicks,
        and slider dragging.

        Parameters
        ----------
        mouse_pos_pixels : (x, y) cursor position in window pixels (top-left origin)
        left_click       : True while the left mouse button is held down
        """
        sw, sh = self.screen_size
        mx =  (mouse_pos_pixels[0] / sw) * 2.0 - 1.0
        my = -((mouse_pos_pixels[1] / sh) * 2.0 - 1.0)

        # ── buttons ──────────────────────────────────────────────────────────
        for btn in self._buttons:
            # Skip buttons whose context is currently hidden
            if not self._context_status.get(btn.get("context_id", "default"), True):
                btn["was_down"] = False
                continue
            inside = (btn["left"] <= mx <= btn["right"] and
                      btn["bottom"] <= my <= btn["top"])
            self.set_hover(btn["element_index"], inside)
            if inside and left_click and not btn["was_down"]:
                if btn["click_function"]:
                    btn["click_function"]()
            btn["was_down"] = left_click and inside

        # ── sliders ───────────────────────────────────────────────────────────
        for sld in self._sliders:
            # Skip sliders whose context is currently hidden
            if not self._context_status.get(sld.get("context_id", "default"), True):
                sld["was_down"] = False
                sld["is_dragging"] = False
                continue
            # current knob centre in NDC
            t = (sld["value"] - sld["min_value"]) / (sld["max_value"] - sld["min_value"])
            knob_cx = sld["left"] + t * (sld["right"] - sld["left"])

            over_knob = (knob_cx - sld["knob_hw"] <= mx <= knob_cx + sld["knob_hw"] and
                         sld["cy"] - sld["knob_hh"] <= my <= sld["cy"] + sld["knob_hh"])
            over_track = (sld["left"] <= mx <= sld["right"] and
                          sld["bottom"] <= my <= sld["top"])

            # hover highlight on knob only
            self.set_hover(sld["knob_index"], over_knob)

            # start drag on leading press edge over knob or track
            if left_click and not sld["was_down"] and (over_knob or over_track):
                sld["is_dragging"] = True

            # stop drag when mouse released
            if not left_click:
                sld["is_dragging"] = False

            # update value while dragging
            if sld["is_dragging"] and left_click:
                mx_clamped = max(sld["left"], min(sld["right"], mx))
                new_t = (mx_clamped - sld["left"]) / (sld["right"] - sld["left"])
                new_value = sld["min_value"] + new_t * (sld["max_value"] - sld["min_value"])
                if abs(new_value - sld["value"]) > 1e-9:
                    sld["value"] = new_value
                    new_knob_cx = sld["left"] + new_t * (sld["right"] - sld["left"])
                    self._patch_element_center_x(sld["knob_index"], new_knob_cx)
                    if sld["callback"]:
                        sld["callback"](new_value)

            sld["was_down"] = left_click and (over_knob or over_track or sld["is_dragging"])

    def draw(self):
        """Flush dirty vertex data and issue one draw call for quads, then one for text."""
        if self._element_count == 0:
            return

        if self._dirty:
            self._upload_vertices_only()
            self._dirty = False

        glUseProgram(self._shader)
        glEnable(GL_BLEND)
        glDisable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # ── pass 1: UI atlas quads ────────────────────────────────────────────
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_atlas)
        glUniform1i(self._u_atlas, 0)

        glBindVertexArray(self._vao)
        glDrawElements(GL_TRIANGLES, len(self._indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        # ── pass 2: font atlas text (only if any text was added) ─────────────
        if self._font_texture is not None and len(self._text_vertices) > 0:
            if self._text_dirty:
                self._upload_text_buffers()
                self._text_dirty = False

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self._font_texture)
            glUniform1i(self._u_atlas, 0)

            glBindVertexArray(self._text_vao)
            glDrawArrays(GL_TRIANGLES, 0,
                         len(self._text_vertices) // _FLOATS_PER_VERTEX)
            glBindVertexArray(0)

        # ── pass 3: standalone texture elements ──────────────────────────────
        for te in self._texture_elements:
            if not te['visible']:
                continue
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, te['texture_id'])
            glUniform1i(self._u_atlas, 0)
            glBindVertexArray(te['vao'])
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

    # ── context API (mirrors gui.GUI.switch/toggle_context_status) ──────────────

    def switch_context_status(self, context_id: str, status: bool):
        """
        Show (status=True) or hide (status=False) all elements, buttons, sliders
        and text that belong to *context_id*.

        Hidden elements are excluded from both the GPU draw call and from mouse
        hit-testing, exactly mirroring the behaviour of the old GUI system.
        """
        if context_id not in self._context_status:
            print(f"WARNING: GUIBatched: '{context_id}' is not a registered context.")
            return
        self._context_status[context_id] = status
        for ei in self._context_elements.get(context_id, []):
            self._element_visible[ei] = status
        # Also update standalone texture elements belonging to this context
        for te in self._texture_elements:
            if te['context_id'] == context_id:
                te['visible'] = status
        self._rebuild_and_upload_indices()
        self._rebuild_text_vertices()

    def toggle_context_status(self, context_id: str):
        """Toggle a context between visible and hidden."""
        if context_id not in self._context_status:
            print(f"WARNING: GUIBatched: '{context_id}' is not a registered context.")
            return
        self.switch_context_status(context_id, not self._context_status[context_id])

    # ── private helpers ─────────────────────────────────────────────────────────

    def _register_to_context(self, context_id: str, status: bool, element_index: int):
        """Register *element_index* to *context_id*, creating the context if new."""
        if context_id not in self._context_status:
            self._context_status[context_id] = status
            self._context_elements[context_id] = []
        self._context_elements[context_id].append(element_index)
        # Apply the context's *current* visibility to this element
        self._element_visible[element_index] = self._context_status[context_id]

    def _upload_vertex_buffer(self):
        """Upload only vertex data (VBO) to the GPU."""
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, self._vertices.nbytes, self._vertices,
                     GL_DYNAMIC_DRAW)
        glBindVertexArray(0)

    def _rebuild_and_upload_indices(self):
        """Rebuild the EBO to include only quads whose element is currently visible."""
        idx = []
        for i in range(self._element_count):
            if self._element_visible[i]:
                b = i * _VERTS_PER_QUAD
                idx += [b, b + 1, b + 2, b, b + 2, b + 3]
        self._indices = (np.array(idx, dtype=np.uint32)
                         if idx else np.empty((0,), dtype=np.uint32))
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     self._indices.nbytes, self._indices, GL_DYNAMIC_DRAW)
        glBindVertexArray(0)

    def _rebuild_text_vertices(self):
        """Rebuild the text vertex buffer from only the visible text chunks."""
        chunks = [c['verts'] for c in self._text_chunks
                  if self._context_status.get(c['context_id'], True)
                  and len(c['verts']) > 0]
        self._text_vertices = (np.concatenate(chunks) if chunks
                               else np.empty((0,), dtype=np.float32))
        self._upload_text_buffers()

    def _update_text_chunk(self, chunk_index: int, new_text: str):
        """
        Regenerate the vertex data for one text chunk and rebuild the full
        text vertex buffer.  Called by TextHandle.update_text().
        """
        chunk = self._text_chunks[chunk_index]
        chunk['text']  = new_text
        chunk['verts'] = self._build_text_verts(
            new_text, chunk['cx'], chunk['cy'],
            chunk['hw'], chunk['hh'], chunk['font_size'],
        )
        self._rebuild_text_vertices()

    def _parse_fnt(self, path: str) -> dict:
        """
        Parse a BMFont .fnt text file.

        Returns dict mapping each character → metric dict with keys:
        x, y, width, height, xoffset, yoffset, xadvance.
        """
        chars = {}
        try:
            with open(path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if not parts or parts[0] != 'char':
                        continue
                    kv = {}
                    for token in parts[1:]:
                        if '=' in token:
                            k, v = token.split('=', 1)
                            try:
                                kv[k] = int(v)
                            except ValueError:
                                pass
                    if 'id' in kv:
                        chars[chr(kv['id'])] = kv
        except FileNotFoundError:
            print(f"GUIBatched: font file not found: {path}")
        return chars

    def _build_text_verts(self, text: str, cx: float, cy: float,
                          hw: float, hh: float, font_size: float) -> np.ndarray:
        """
        Build a flat float32 array of 5-float GL_TRIANGLES vertices for
        centred text inside the box (cx±hw, cy±hh).

        Glyph UVs use the formula:  u = x_px/512,  v = 1 - y_px/512
        (matching load_texture FLIP_TOP_BOTTOM + the old TextBox convention).
        """
        if not self._font_chars:
            return np.empty((0,), dtype=np.float32)

        ATLAS_SIZE = 512.0

        verts = []   # List[float], 5 floats per vertex, 6 vertices per glyph

        # Start pen at extreme left of box; y at vertical center as baseline
        pen_x = cx - hw
        pen_y = cy  # will shift vertically after building

        for ch in text:
            if ch not in self._font_chars:
                continue
            m = self._font_chars[ch]
            gw  = m['width']   / ATLAS_SIZE * font_size
            gh  = m['height']  / ATLAS_SIZE * font_size
            ox  = m['xoffset'] / ATLAS_SIZE * font_size
            oy  = m['yoffset'] / ATLAS_SIZE * font_size
            adv = m['xadvance']/ ATLAS_SIZE * font_size

            # UV – top of glyph region (high V) and bottom (low V)
            u0 = m['x'] / ATLAS_SIZE
            u1 = (m['x'] + m['width']) / ATLAS_SIZE
            v_top = 1.0 - m['y'] / ATLAS_SIZE
            v_bot = 1.0 - (m['y'] + m['height']) / ATLAS_SIZE

            # Quad corners in NDC
            x0, x1 = pen_x + ox, pen_x + ox + gw
            y_hi = pen_y - oy        # top
            y_lo = pen_y - oy - gh   # bottom

            # Two triangles (TL, BL, TR) and (TR, BL, BR); hover = 0
            verts += [
                x0, y_hi, u0, v_top, 0.0,   # TL
                x0, y_lo, u0, v_bot, 0.0,   # BL
                x1, y_hi, u1, v_top, 0.0,   # TR
                x1, y_hi, u1, v_top, 0.0,   # TR
                x0, y_lo, u0, v_bot, 0.0,   # BL
                x1, y_lo, u1, v_bot, 0.0,   # BR
            ]
            pen_x += adv

        if not verts:
            return np.empty((0,), dtype=np.float32)

        arr = np.array(verts, dtype=np.float32)

        # ── horizontal centering: shift so text midpoint aligns with cx ──────
        xs = arr[0::5]           # every x coord
        text_mid_x = (xs.min() + xs.max()) / 2.0
        arr[0::5] += (cx - text_mid_x)

        # ── vertical centering: shift so text midpoint aligns with cy ────────
        ys = arr[1::5]
        text_mid_y = (ys.min() + ys.max()) / 2.0
        arr[1::5] += (cy - text_mid_y)

        return arr

    def _append_text(self, text: str, position: tuple, scale: tuple,
                     font_size: float, context_id: str = 'default'):
        """Build glyph verts for *text*, store the chunk as a dict (so it can
        be regenerated on update_text), and upload if the context is visible.

        Returns the chunk index (int) so callers can construct a TextHandle,
        or None if no font is loaded.
        """
        if not self._font_chars:
            return None
        cx, cy = position
        hw, hh = scale
        new_verts = self._build_text_verts(text, cx, cy, hw, hh, font_size)
        chunk_index = len(self._text_chunks)
        self._text_chunks.append({
            'context_id': context_id,
            'verts':      new_verts,
            'cx': cx, 'cy': cy,
            'hw': hw, 'hh': hh,
            'font_size':  font_size,
            'text':       text,
        })
        if len(new_verts) > 0 and self._context_status.get(context_id, True):
            self._text_vertices = np.concatenate([self._text_vertices, new_verts])
            self._upload_text_buffers()
            self._text_dirty = False
        return chunk_index

    def _setup_text_vao(self):
        """Set up a VAO/VBO for the text geometry (same attrib layout as main)."""
        glBindVertexArray(self._text_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._text_vbo)

        stride = _FLOATS_PER_VERTEX * ctypes.sizeof(ctypes.c_float)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16))
        glEnableVertexAttribArray(2)

        glBindVertexArray(0)

    def _upload_text_buffers(self):
        """Upload the text vertex array to the GPU."""
        glBindBuffer(GL_ARRAY_BUFFER, self._text_vbo)
        glBufferData(GL_ARRAY_BUFFER,
                     self._text_vertices.nbytes, self._text_vertices,
                     GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _patch_element_center_x(self, element_index: int, new_cx: float):
        """
        Move one element horizontally by patching its 4 vertices' x coords
        in the CPU array.  Reads the existing half-width from the stored verts
        so the quad size is preserved.  Marks the buffer dirty.
        """
        base = element_index * _VERTS_PER_QUAD * _FLOATS_PER_VERTEX
        # Layout: TL(0), BL(1), BR(2), TR(3) – each stride = _FLOATS_PER_VERTEX
        tl_x = float(self._vertices[base + 0 * _FLOATS_PER_VERTEX + 0])
        tr_x = float(self._vertices[base + 3 * _FLOATS_PER_VERTEX + 0])
        hw = (tr_x - tl_x) / 2.0
        self._vertices[base + 0 * _FLOATS_PER_VERTEX + 0] = new_cx - hw  # TL x
        self._vertices[base + 1 * _FLOATS_PER_VERTEX + 0] = new_cx - hw  # BL x
        self._vertices[base + 2 * _FLOATS_PER_VERTEX + 0] = new_cx + hw  # BR x
        self._vertices[base + 3 * _FLOATS_PER_VERTEX + 0] = new_cx + hw  # TR x
        self._dirty = True

    def _setup_vao(self):
        glBindVertexArray(self._vao)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)

        stride = _FLOATS_PER_VERTEX * ctypes.sizeof(ctypes.c_float)  # 20 bytes

        # a_pos  (location 0) : 2 floats at offset 0
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # a_uv   (location 1) : 2 floats at offset 8
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)

        # a_hover(location 2) : 1 float  at offset 16
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16))
        glEnableVertexAttribArray(2)

        glBindVertexArray(0)

    def _upload_buffers(self):
        """Re-upload both vertex and index buffers (called when adding elements)."""
        glBindVertexArray(self._vao)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER,
                     self._vertices.nbytes, self._vertices, GL_DYNAMIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     self._indices.nbytes, self._indices, GL_DYNAMIC_DRAW)

        glBindVertexArray(0)

    def _upload_vertices_only(self):
        """Cheap partial update – only vertex data changes (hover state)."""
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0,
                        self._vertices.nbytes, self._vertices)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

