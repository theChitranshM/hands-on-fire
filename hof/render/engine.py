"""GPU renderer: fullscreen projector window + operator HUD window, both GLFW + moderngl.

Layers, all in projector pixel space (origin top-left, y down, size = engine.size):
  direct  -> RGBA16F, cleared every frame. Flames, diyas, rangoli. No persistence (no lag smear).
  trail   -> RGBA16F, previous frame * trail_decay + this frame. Sparks and embers get motion trails.
Composite: (direct + trail) * brightness + multi-level bloom of the bright parts, hot values bleed to white,
clamped, upscaled to the projector.
"""
import ctypes
import time
import numpy as np
import glfw
import moderngl

VERT_FULLSCREEN = """
#version 410
in vec2 in_pos;
out vec2 uv;
void main() { uv = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

# Projector-pixel helpers every effect shader can paste in: px (top-left origin) -> clip space.
GLSL_PX_TO_CLIP = """
uniform vec2 u_res;
vec4 px_to_clip(vec2 p) { return vec4(p.x / u_res.x * 2.0 - 1.0, 1.0 - p.y / u_res.y * 2.0, 0.0, 1.0); }
"""

_COMPOSITE_FRAG = """
#version 410
uniform sampler2D direct_tex;
uniform sampler2D trail_tex;
uniform sampler2D bloom0, bloom1, bloom2, bloom3;
uniform float brightness;
uniform float trail_gain;
uniform float bloom_strength;
uniform vec4 bloom_w;
in vec2 uv;
out vec4 f;
void main() {
    vec3 c = (texture(direct_tex, uv).rgb + texture(trail_tex, uv).rgb * trail_gain) * brightness;
    c += (texture(bloom0, uv).rgb * bloom_w.x + texture(bloom1, uv).rgb * bloom_w.y
        + texture(bloom2, uv).rgb * bloom_w.z + texture(bloom3, uv).rgb * bloom_w.w) * bloom_strength;
    float m = max(c.r, max(c.g, c.b));
    c += vec3(max(m - 1.0, 0.0) * 0.6);     // white-hot cores instead of hue-clipped blobs
    f = vec4(clamp(c, 0.0, 1.0), 1.0);
}
"""

# bloom bright-pass: soft-knee threshold on the same (direct + trail) * brightness the composite uses, 4-tap box down to half res
_BRIGHT_FRAG = """
#version 410
uniform sampler2D direct_tex;
uniform sampler2D trail_tex;
uniform float brightness, trail_gain, threshold, knee;
uniform vec2 texel;
in vec2 uv;
out vec4 f;
vec3 src(vec2 u) { return (texture(direct_tex, u).rgb + texture(trail_tex, u).rgb * trail_gain) * brightness; }
void main() {
    vec3 c = (src(uv + texel * vec2(-0.5, -0.5)) + src(uv + texel * vec2(0.5, -0.5))
            + src(uv + texel * vec2(-0.5, 0.5)) + src(uv + texel * vec2(0.5, 0.5))) * 0.25;
    float m = max(c.r, max(c.g, c.b));
    float soft = clamp(m - threshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 1e-4);
    f = vec4(c * (max(soft, m - threshold) / max(m, 1e-4)), 1.0);
}
"""

# 9-tap gaussian along `step` (uv units); also serves as the downsample between levels
_BLUR_FRAG = """
#version 410
uniform sampler2D tex;
uniform vec2 step;
in vec2 uv;
out vec4 f;
void main() {
    vec3 c = texture(tex, uv).rgb * 0.2270;
    c += (texture(tex, uv + step).rgb + texture(tex, uv - step).rgb) * 0.1946;
    c += (texture(tex, uv + step * 2.0).rgb + texture(tex, uv - step * 2.0).rgb) * 0.1216;
    c += (texture(tex, uv + step * 3.0).rgb + texture(tex, uv - step * 3.0).rgb) * 0.0541;
    c += (texture(tex, uv + step * 4.0).rgb + texture(tex, uv - step * 4.0).rgb) * 0.0162;
    f = vec4(c, 1.0);
}
"""

_DECAY_FRAG = """
#version 410
uniform sampler2D tex;
uniform float decay;
in vec2 uv;
out vec4 f;
void main() { f = vec4(texture(tex, uv).rgb * decay, 1.0); }
"""

_BLIT_FRAG = """
#version 410
uniform sampler2D tex;
uniform int flip_y;
in vec2 uv;
out vec4 f;
void main() { vec2 u = flip_y == 1 ? vec2(uv.x, 1.0 - uv.y) : uv; f = vec4(texture(tex, u).rgb, 1.0); }
"""

_LINE_VERT = """
#version 410
in vec2 in_pos;
in vec4 in_color;
out vec4 v_color;
""" + GLSL_PX_TO_CLIP + """
void main() { v_color = in_color; gl_Position = px_to_clip(in_pos); }
"""
_LINE_FRAG = """
#version 410
in vec4 v_color;
out vec4 f;
void main() { f = v_color; }
"""

KEY_NAMES = {glfw.KEY_SPACE: "SPACE", glfw.KEY_ESCAPE: "ESCAPE", glfw.KEY_ENTER: "ENTER",
             glfw.KEY_LEFT: "LEFT", glfw.KEY_RIGHT: "RIGHT", glfw.KEY_UP: "UP", glfw.KEY_DOWN: "DOWN",
             glfw.KEY_TAB: "TAB", glfw.KEY_BACKSPACE: "BACKSPACE"}
KEY_NAMES.update({getattr(glfw, f"KEY_{c}"): c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"})

_glfw_ready = False


def init_glfw():
    global _glfw_ready
    if not _glfw_ready:
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        _glfw_ready = True


def fullscreen_quad(ctx, frag_src, vert_src=VERT_FULLSCREEN):
    """Return (program, vao) drawing a clip-space quad. Use vao.render(moderngl.TRIANGLE_STRIP)."""
    prog = ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
    vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
    return prog, ctx.vertex_array(prog, [(vbo, "2f", "in_pos")])


def _mid(m):
    return ctypes.cast(m, ctypes.c_void_p).value


def find_projector_monitor(spec):
    init_glfw()
    mons = glfw.get_monitors()
    primary = _mid(glfw.get_primary_monitor())
    if spec == "auto":
        others = [m for m in mons if _mid(m) != primary]
        return others[0] if others else None
    if isinstance(spec, int):
        return mons[spec] if spec < len(mons) else None
    for m in mons:
        if str(spec).lower() in glfw.get_monitor_name(m).decode(errors="ignore").lower():
            return m
    return None


def list_monitors():
    init_glfw()
    out = []
    for i, m in enumerate(glfw.get_monitors()):
        mode = glfw.get_video_mode(m)
        out.append({"index": i, "name": glfw.get_monitor_name(m).decode(errors="ignore"),
                    "size": tuple(mode.size), "refresh": mode.refresh_rate,
                    "primary": _mid(m) == _mid(glfw.get_primary_monitor()),
                    "modes": sorted({(v.size.width, v.size.height) for v in glfw.get_video_modes(m)})})
    return out


def _gl_hints(visible=True):
    glfw.default_window_hints()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.VISIBLE, visible)
    glfw.window_hint(glfw.AUTO_ICONIFY, False)   # don't minimise the projector when the HUD gets focus
    glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, False)


class Window:
    """A GLFW window with its own moderngl context and a shared key queue."""

    def __init__(self, title, size, monitor=None, vsync=True, key_queue=None, visible=True):
        init_glfw()
        _gl_hints(visible)
        if monitor is not None:
            mode = glfw.get_video_mode(monitor)
            glfw.window_hint(glfw.REFRESH_RATE, mode.refresh_rate)
            size = (mode.size.width, mode.size.height)
        self.win = glfw.create_window(int(size[0]), int(size[1]), title, monitor, None)
        if not self.win:
            raise RuntimeError(f"could not create window {title}")
        self.keys = key_queue if key_queue is not None else []
        glfw.set_key_callback(self.win, self._on_key)
        self.use()
        glfw.swap_interval(1 if vsync else 0)
        self.ctx = moderngl.create_context()
        self.fb_size = glfw.get_framebuffer_size(self.win)

    def _on_key(self, win, key, scancode, action, mods):
        if action == glfw.PRESS and key in KEY_NAMES:
            self.keys.append(KEY_NAMES[key])

    def use(self):
        glfw.make_context_current(self.win)

    def swap(self):
        glfw.swap_buffers(self.win)

    @property
    def should_close(self):
        return glfw.window_should_close(self.win)

    def close(self):
        glfw.destroy_window(self.win)


def poll_keys(key_queue):
    """Pump GLFW events; return and clear key names pressed since last call."""
    glfw.poll_events()
    keys = list(key_queue)
    key_queue.clear()
    return keys


class ImageWindow(Window):
    """Window that shows a numpy image stretched to fill it. Used for the HUD and by tools."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._prog, self._vao = fullscreen_quad(self.ctx, _BLIT_FRAG)
        self._tex = None

    def show(self, img, bgr=True):
        """img: HxWx3 uint8. Draws and swaps."""
        self.use()
        h, w = img.shape[:2]
        if self._tex is None or self._tex.size != (w, h):
            self._tex = self.ctx.texture((w, h), 3, dtype="f1")
            self._tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        data = img[:, :, ::-1] if bgr else img
        self._tex.write(np.ascontiguousarray(data).tobytes())
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, *glfw.get_framebuffer_size(self.win))
        self.ctx.clear(0, 0, 0)
        self._tex.use(0)
        self._prog["tex"] = 0
        self._prog["flip_y"] = 1
        self._vao.render(moderngl.TRIANGLE_STRIP)
        self.swap()


def open_projector(cfg, key_queue, title="Hands on Fire — projector", image_window=False):
    """Fullscreen window on the projector; falls back to a window on the Mac screen if none attached.
    Returns (window, render_size, is_real_projector)."""
    mon = find_projector_monitor(cfg.display.projector_monitor)
    real = mon is not None
    if not real and not cfg.display.fallback_windowed:
        raise RuntimeError("No projector monitor found. Check HDMI and set Displays to Extend, not Mirror.")
    cls = ImageWindow if image_window else Window
    win = cls(title, (960, 540), monitor=mon, vsync=cfg.display.vsync, key_queue=key_queue)
    rs = cfg.display.render_size
    if rs == "auto":
        mw, mh = cfg.display.max_render_size
        fw, fh = win.fb_size
        k = min(1.0, mw / fw, mh / fh)
        rs = (int(round(fw * k)), int(round(fh * k)))
    if real:
        glfw.set_input_mode(win.win, glfw.CURSOR, glfw.CURSOR_HIDDEN)
    return win, tuple(int(v) for v in rs), real


class Engine:
    """Owns the projector window and the layer FBOs. Effects draw into the layers in projector px."""

    def __init__(self, cfg, key_queue):
        self.cfg = cfg
        self.win, self.size, self.is_projector = open_projector(cfg, key_queue)
        self.ctx = self.win.ctx
        ctx = self.ctx
        self.direct_tex = ctx.texture(self.size, 4, dtype="f2")
        self.direct = ctx.framebuffer([self.direct_tex])
        self._trail_tex = [ctx.texture(self.size, 4, dtype="f2") for _ in range(2)]
        self._trail = [ctx.framebuffer([t]) for t in self._trail_tex]
        self._ti = 0
        for t in [self.direct_tex] + self._trail_tex:
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.final_tex = ctx.texture(self.size, 4, dtype="f1")
        self.final_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.final = ctx.framebuffer([self.final_tex])
        self._comp, self._comp_vao = fullscreen_quad(ctx, _COMPOSITE_FRAG)
        self._bright, self._bright_vao = fullscreen_quad(ctx, _BRIGHT_FRAG)
        self._blur, self._blur_vao = fullscreen_quad(ctx, _BLUR_FRAG)
        b = cfg.render.bloom
        self._bloom_on = float(b.strength) > 0
        self._bloom_sz = [(max(2, self.size[0] >> (k + 1)), max(2, self.size[1] >> (k + 1))) for k in range(4)]
        self._bloom_tex = [ctx.texture(sz, 4, dtype="f2") for sz in self._bloom_sz]       # blurred result per level
        self._bloom_tmp = [ctx.texture(sz, 4, dtype="f2") for sz in self._bloom_sz]       # horizontal pass scratch
        for t in self._bloom_tex + self._bloom_tmp:
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
        self._bloom_fb = [ctx.framebuffer([t]) for t in self._bloom_tex]
        self._bloom_tmp_fb = [ctx.framebuffer([t]) for t in self._bloom_tmp]
        self._bright_tex = ctx.texture(self._bloom_sz[0], 4, dtype="f2")
        self._bright_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._bright_fb = ctx.framebuffer([self._bright_tex])
        self.bloom_strength = float(b.strength)   # sequence/tools may animate this
        self._decay, self._decay_vao = fullscreen_quad(ctx, _DECAY_FRAG)
        self._blit, self._blit_vao = fullscreen_quad(ctx, _BLIT_FRAG)
        self._line_prog = ctx.program(vertex_shader=_LINE_VERT, fragment_shader=_LINE_FRAG)
        self._line_vbo = ctx.buffer(reserve=6 * 4 * 20000, dynamic=True)
        self._line_vao = ctx.vertex_array(self._line_prog, [(self._line_vbo, "2f 4f", "in_pos", "in_color")])
        self._lines = []
        self._small = {}
        self.brightness = float(cfg.render.brightness)
        self.blackout = False
        self.trail_gain = 1.0          # sequence fades the spark layer out for the quiet end
        self.fps = 0.0
        self._last = time.perf_counter()
        for fb in [self.direct] + self._trail:
            fb.clear(0, 0, 0, 1)

    # ---- per-frame API -------------------------------------------------------------------
    def begin_direct(self):
        """Bind the direct layer (cleared) with additive blending. Draw flames etc. after this."""
        self.win.use()
        self.direct.use()
        self.ctx.viewport = (0, 0, *self.size)
        self.direct.clear(0, 0, 0, 1)
        self._additive()

    def begin_trail(self):
        """Bind the trail layer: previous trail * decay is already in it. Draw sparks after this."""
        self.win.use()
        src, dst = self._trail[self._ti], self._trail[1 - self._ti]
        self._ti = 1 - self._ti
        dst.use()
        self.ctx.viewport = (0, 0, *self.size)
        self.ctx.disable(moderngl.BLEND)
        src.color_attachments[0].use(0)
        self._decay["tex"] = 0
        self._decay["decay"] = float(self.cfg.render.trail_decay)
        self._decay_vao.render(moderngl.TRIANGLE_STRIP)
        self._additive()

    def clear_trails(self):
        for fb in self._trail:
            fb.clear(0, 0, 0, 1)

    def line(self, pts, color=(0.3, 0.3, 0.3, 1.0), closed=False):
        """Queue a debug polyline in projector px; drawn on top of the final image this frame."""
        pts = [tuple(p) for p in pts]
        if closed and pts:
            pts.append(pts[0])
        for a, b in zip(pts[:-1], pts[1:]):
            self._lines.append((*a, *color))
            self._lines.append((*b, *color))

    def present(self):
        """Composite layers, draw debug lines, upscale to the projector and swap."""
        ctx = self.ctx
        ctx.disable(moderngl.BLEND)
        self.final.use()
        ctx.viewport = (0, 0, *self.size)
        if self.blackout:
            self.final.clear(0, 0, 0, 1)
        else:
            bs = self.bloom_strength if self._bloom_on else 0.0
            if bs > 0:
                self._bloom_pass()
                self.final.use()
                ctx.viewport = (0, 0, *self.size)
            self.direct_tex.use(0)
            self._trail[self._ti].color_attachments[0].use(1)
            for k in range(4):
                self._bloom_tex[k].use(2 + k)
                self._comp[f"bloom{k}"] = 2 + k
            self._comp["bloom_strength"] = bs
            self._comp["bloom_w"] = tuple(float(x) for x in self.cfg.render.bloom.weights)
            self._comp["direct_tex"] = 0
            self._comp["trail_tex"] = 1
            self._comp["brightness"] = self.brightness
            self._comp["trail_gain"] = self.trail_gain
            self._comp_vao.render(moderngl.TRIANGLE_STRIP)
        if self._lines and not self.blackout:
            data = np.array(self._lines[:20000], dtype="f4")
            self._line_vbo.write(data.tobytes())
            self._line_prog["u_res"] = self.size
            self._line_vao.render(moderngl.LINES, vertices=len(data))
        self._lines.clear()
        ctx.screen.use()
        ctx.viewport = (0, 0, *self.win.fb_size)
        self.final_tex.use(0)
        self._blit["tex"] = 0
        self._blit["flip_y"] = 0
        self._blit_vao.render(moderngl.TRIANGLE_STRIP)
        self.win.swap()
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        if dt > 0:
            self.fps = self.fps * 0.9 + (1.0 / dt) * 0.1 if self.fps else 1.0 / dt

    def grab(self, size):
        """Read back the final frame downscaled to size=(w,h): HxWx3 uint8 RGB, top row first."""
        self.win.use()
        if size not in self._small:
            t = self.ctx.texture(size, 3, dtype="f1")
            self._small[size] = self.ctx.framebuffer([t])
        fb = self._small[size]
        self.ctx.disable(moderngl.BLEND)
        fb.use()
        self.ctx.viewport = (0, 0, *size)
        self.final_tex.use(0)
        self._blit["tex"] = 0
        self._blit["flip_y"] = 1
        self._blit_vao.render(moderngl.TRIANGLE_STRIP)
        return np.frombuffer(fb.read(components=3), dtype=np.uint8).reshape(size[1], size[0], 3)

    def _bloom_pass(self):
        """Bright-pass at half res, then 4 blurred levels (1/2..1/16). Read back by the composite."""
        ctx, b = self.ctx, self.cfg.render.bloom
        ctx.disable(moderngl.BLEND)
        self._bright_fb.use()
        ctx.viewport = (0, 0, *self._bloom_sz[0])
        self.direct_tex.use(0)
        self._trail[self._ti].color_attachments[0].use(1)
        pr = self._bright
        pr["direct_tex"], pr["trail_tex"] = 0, 1
        pr["brightness"], pr["trail_gain"] = self.brightness, self.trail_gain
        pr["threshold"], pr["knee"] = float(b.threshold), float(b.knee)
        pr["texel"] = (1.0 / self.size[0], 1.0 / self.size[1])
        self._bright_vao.render(moderngl.TRIANGLE_STRIP)
        src = self._bright_tex
        for k in range(4):
            w, h = self._bloom_sz[k]
            r = float(b.radius)
            for dst, st in ((self._bloom_tmp_fb[k], (r / w, 0.0)), (self._bloom_fb[k], (0.0, r / h))):
                dst.use()
                ctx.viewport = (0, 0, w, h)
                src.use(0)
                self._blur["tex"] = 0
                self._blur["step"] = st
                self._blur_vao.render(moderngl.TRIANGLE_STRIP)
                src = self._bloom_tmp[k]
            src = self._bloom_tex[k]

    def _additive(self):
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        self.ctx.blend_equation = moderngl.FUNC_ADD
