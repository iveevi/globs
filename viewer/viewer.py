from pathlib import Path

import numpy as np
import slangpy as spy

from . import env
from . import gizmos
from .camera import FlyCamera
from .gltf_loader import load_scene

SHADER_DIR = Path(__file__).resolve().parent / "shaders"
FONT_DIR = Path(__file__).resolve().parent / "font"
ENV_PATH = Path(__file__).resolve().parent / "env" / "sky_1k.hdr"
COLOR_FORMAT = spy.Format.rgba8_unorm
DEPTH_FORMAT = spy.Format.d32_float
VERTEX_STRIDE = 32
GIZMO_STRIDE = 24
SHADOW_SIZE = 2048

SHADING_MODES = ["Lit", "Solid", "Albedo", "Geometric Normals", "Shading Normals",
                 "Metallic", "Roughness", "Occlusion"]

# Nerd Font (Font Awesome) glyphs, provided by the bundled CaskaydiaCove font.
ICON_MESH = "\uf1b2"
ICON_CAMERA = "\uf030"
ICON_LIGHT = "\uf0eb"
ICON_SUN = "\uf185"
ICON_RESET = "\uf021"
ICON_GEAR = "\uf013"


class Viewer:
    def __init__(self, glb_path):
        self.path = Path(glb_path)
        self.window = spy.Window(
            width=1600, height=900, title=f"GLB Viewer — {self.path.name}", resizable=True
        )
        self.device = spy.Device(
            enable_debug_layers=False,
            compiler_options={"include_paths": [SHADER_DIR]},
        )
        if not self.device.has_feature(spy.Feature.rasterization):
            raise RuntimeError("device has no rasterization support")

        self.surface = self.device.create_surface(self.window)
        self.surface.configure(width=self.window.width, height=self.window.height, vsync=True)
        self.ui = spy.ui.Context(self.device)
        self.ui.ini_filename = None
        font_path = str(FONT_DIR / "CaskaydiaCoveNerdFontMono-Regular.ttf")
        self.ui.add_font("ui", font_path, 16, is_default=True)
        # Same font file at a larger size, for the icon-only overlay buttons.
        self.ui.add_font("icon_large", font_path, 28)

        self.scene = load_scene(glb_path)
        self.mesh = self.scene.mesh
        self.upload_geometry()
        self.upload_materials()
        self.upload_lights()
        self.load_ibl()
        self.build_gizmos()
        self.build_pipelines()
        self.build_shadows()

        self.camera = FlyCamera()
        self.camera.frame(self.mesh.center, self.mesh.radius)

        self.shading_mode = 0
        self.wireframe = False
        self.alpha_cutout = True
        self.headlight = 0.9
        self.ambient = 0.25
        self.show_meshes = True
        self.show_cameras = True
        self.show_lights = True
        self.move_speed = 1.5

        self.color_texture = None
        self.depth_texture = None
        self.keys_held = set()
        self.dragging = {"rotate": False, "pan": False}
        self.last_mouse = None
        self.terminate = False

        self.window.on_keyboard_event = self.on_keyboard
        self.window.on_mouse_event = self.on_mouse
        self.window.on_resize = self.on_resize

        self.build_ui()

    # ---- GPU upload -------------------------------------------------------

    def upload_geometry(self):
        self.vertex_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource,
            label="vertices",
            data=self.mesh.vertices.reshape(-1),
        )
        self.index_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.index_buffer | spy.BufferUsage.shader_resource,
            label="indices",
            data=self.mesh.indices,
        )
        self.input_layout = self.device.create_input_layout(
            input_elements=[
                {"semantic_name": "POSITION", "semantic_index": 0,
                 "format": spy.Format.rgb32_float, "offset": 0},
                {"semantic_name": "NORMAL", "semantic_index": 0,
                 "format": spy.Format.rgb32_float, "offset": 12},
                {"semantic_name": "TEXCOORD", "semantic_index": 0,
                 "format": spy.Format.rg32_float, "offset": 24},
            ],
            vertex_streams=[{"stride": VERTEX_STRIDE}],
        )
        self.gizmo_layout = self.device.create_input_layout(
            input_elements=[
                {"semantic_name": "POSITION", "semantic_index": 0,
                 "format": spy.Format.rgb32_float, "offset": 0},
                {"semantic_name": "COLOR", "semantic_index": 0,
                 "format": spy.Format.rgb32_float, "offset": 12},
            ],
            vertex_streams=[{"stride": GIZMO_STRIDE}],
        )

    def white_texture(self):
        data = np.full((1, 1, 4), 255, dtype=np.uint8)
        return self.device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba8_unorm,
            width=1, height=1,
            usage=spy.TextureUsage.shader_resource,
            data=data, label="white",
        ).create_view({})

    def upload_materials(self):
        self.sampler = self.device.create_sampler(
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            mip_filter=spy.TextureFilteringMode.linear,
            address_u=spy.TextureAddressingMode.wrap,
            address_v=spy.TextureAddressingMode.wrap,
            label="material_sampler",
        )
        self.white_view = self.white_texture()
        self.material_views = []
        self.material_mr_views = []
        self.material_occ_views = []
        self.material_normal_views = []
        self.material_factors = []
        self.material_metallic = []
        self.material_roughness = []
        self.material_occ_strength = []
        for mat in self.mesh.materials:
            self.material_factors.append(mat.base_color_factor)
            self.material_metallic.append(mat.metallic_factor)
            self.material_roughness.append(mat.roughness_factor)
            self.material_occ_strength.append(mat.occlusion_strength)
            self.material_views.append(self.upload_image(mat.base_color_image, srgb=True))
            self.material_mr_views.append(self.upload_image(mat.metallic_roughness_image))
            self.material_occ_views.append(self.upload_image(mat.occlusion_image))
            self.material_normal_views.append(self.upload_image(mat.normal_image))

    def upload_image(self, image, srgb=False):
        if image is None:
            return None
        img = np.ascontiguousarray(image)
        tex = self.device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba8_unorm_srgb if srgb else spy.Format.rgba8_unorm,
            width=img.shape[1], height=img.shape[0],
            usage=spy.TextureUsage.shader_resource,
            data=img, label="material_texture",
        )
        return tex.create_view({})

    def upload_lights(self):
        dirs = [li for li in self.scene.lights if li.type == "directional"]
        points = [li for li in self.scene.lights if li.type in ("point", "spot")]
        self.num_dir = len(dirs)
        self.num_point = len(points)

        dir_arr = np.zeros((max(1, self.num_dir), 8), dtype=np.float32)
        for i, li in enumerate(dirs):
            dir_arr[i, 0:3] = li.direction
            dir_arr[i, 3] = li.intensity / 683.0 if li.intensity > 10 else li.intensity
            dir_arr[i, 4:7] = li.color
        point_arr = np.zeros((max(1, self.num_point), 8), dtype=np.float32)
        for i, li in enumerate(points):
            point_arr[i, 0:3] = li.position
            point_arr[i, 3] = li.range or 0.0
            point_arr[i, 4:7] = li.color
            point_arr[i, 7] = li.intensity / 683.0 if li.intensity > 10 else li.intensity

        self.dir_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            struct_size=32, element_count=dir_arr.shape[0],
            data=dir_arr.reshape(-1), label="dir_lights",
        )
        self.point_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            struct_size=32, element_count=point_arr.shape[0],
            data=point_arr.reshape(-1), label="point_lights",
        )

    def load_ibl(self):
        image = env.load_hdr(ENV_PATH)
        sh = env.project_sh(image)
        self.sh_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            struct_size=16, element_count=9,
            data=np.ascontiguousarray(sh).reshape(-1), label="sh_coeffs",
        )
        h, w = image.shape[:2]
        rgba = np.concatenate([image, np.ones((h, w, 1), np.float32)], axis=-1)
        self.env_texture = self.device.create_texture(
            type=spy.TextureType.texture_2d, format=spy.Format.rgba32_float,
            width=w, height=h, usage=spy.TextureUsage.shader_resource,
            data=np.ascontiguousarray(rgba), label="environment",
        )
        self.env_view = self.env_texture.create_view({})
        self.env_sampler = self.device.create_sampler(
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            address_u=spy.TextureAddressingMode.wrap,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
            label="env_sampler",
        )
        self.exposure = 1.0

    def make_gizmo_buffer(self, verts, label):
        if verts is None or len(verts) == 0:
            return None
        data = np.ascontiguousarray(verts.astype(np.float32))
        buf = self.device.create_buffer(
            usage=spy.BufferUsage.vertex_buffer,
            data=data.reshape(-1), label=label,
        )
        return (buf, data.shape[0])

    def build_gizmos(self):
        r = max(self.mesh.radius, 1e-3)
        center = self.mesh.center
        grid_y = float(self.mesh.bounds_min[1])
        grid_extent = r * 14.0
        cx, cz = float(center[0]), float(center[2])
        self.grid_fade_center = spy.float3(cx, grid_y, cz)
        self.grid_fade_radius = grid_extent * 0.85
        self.grid_cell = r * 0.35
        e = grid_extent
        ground = np.array([
            [cx - e, grid_y, cz - e], [cx + e, grid_y, cz - e], [cx + e, grid_y, cz + e],
            [cx - e, grid_y, cz - e], [cx + e, grid_y, cz + e], [cx - e, grid_y, cz + e],
        ], dtype=np.float32)
        self.grid_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.vertex_buffer, data=ground.reshape(-1),
            label="grid_ground",
        )
        self.line_gizmos = {
            "cameras": self.make_gizmo_buffer(
                gizmos.build_cameras(self.scene.cameras, r), "gizmo_cameras"),
            "lights": self.make_gizmo_buffer(
                gizmos.build_lights(self.scene.lights, r), "gizmo_lights"),
        }

    def build_shadows(self):
        # Render a depth shadow map for the first directional light. The scene
        # and lights are static, so this is done once.
        self.has_shadow = False
        self.light_vp = spy.float4x4.identity()
        self.shadow_view = self.white_view
        self.shadow_sampler = self.sampler
        dirs = [li for li in self.scene.lights if li.type == "directional"]
        if not dirs:
            return

        r = max(self.mesh.radius, 1e-3)
        center = np.asarray(self.mesh.center, dtype=np.float64)
        d = np.asarray(dirs[0].direction, dtype=np.float64)
        d = d / (np.linalg.norm(d) or 1.0)
        up = (0.0, 1.0, 0.0) if abs(d[1]) < 0.95 else (0.0, 0.0, 1.0)
        eye = center - d * r * 1.5
        view = spy.math.matrix_from_look_at(
            spy.float3(*eye), spy.float3(*center), spy.float3(*up))
        proj = spy.math.ortho(-r, r, -r, r, 0.01, 3.0 * r)
        self.light_vp = spy.math.mul(proj, view)

        shadow_tex = self.device.create_texture(
            format=DEPTH_FORMAT, width=SHADOW_SIZE, height=SHADOW_SIZE,
            usage=spy.TextureUsage.depth_stencil | spy.TextureUsage.shader_resource,
            label="shadow_map",
        )
        self.shadow_view = shadow_tex.create_view({})
        self.shadow_sampler = self.device.create_sampler(
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
            label="shadow_sampler",
        )
        shadow_program = self.device.load_program("shadow.slang", ["shadow_vs", "shadow_fs"])
        shadow_pipeline = self.device.create_render_pipeline(
            program=shadow_program, input_layout=self.input_layout,
            primitive_topology=spy.PrimitiveTopology.triangle_list,
            targets=[],
            depth_stencil={
                "format": DEPTH_FORMAT,
                "depth_test_enable": True,
                "depth_write_enable": True,
                "depth_func": spy.ComparisonFunc.less_equal,
            },
        )
        encoder = self.device.create_command_encoder()
        pass_desc = {
            "depth_stencil_attachment": {
                "view": self.shadow_view,
                "depth_load_op": spy.LoadOp.clear,
                "depth_store_op": spy.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        }
        with encoder.begin_render_pass(pass_desc) as rp:
            rp.set_render_state({
                "viewports": [spy.Viewport.from_size(SHADOW_SIZE, SHADOW_SIZE)],
                "scissor_rects": [spy.ScissorRect.from_size(SHADOW_SIZE, SHADOW_SIZE)],
                "vertex_buffers": [self.vertex_buffer],
                "index_buffer": self.index_buffer,
                "index_format": spy.IndexFormat.uint32,
            })
            obj = rp.bind_pipeline(shadow_pipeline)
            spy.ShaderCursor(obj).g_light_vp = self.light_vp
            rp.draw_indexed({"vertex_count": int(self.mesh.indices.shape[0]),
                             "start_index_location": 0})
        self.device.submit_command_buffer(encoder.finish())
        self.device.wait()
        self.has_shadow = True

    def build_pipelines(self):
        self.scene_program = self.device.load_program("scene.slang", ["vs_main", "fs_main"])
        self.scene_pipelines = {}
        self.gizmo_program = self.device.load_program(
            "gizmo.slang", ["gizmo_vs", "gizmo_fs"]
        )
        self.sky_program = self.device.load_program("sky.slang", ["sky_vs", "sky_fs"])
        self.sky_pipeline = self.device.create_render_pipeline(
            program=self.sky_program,
            input_layout=None,
            primitive_topology=spy.PrimitiveTopology.triangle_list,
            targets=[{"format": COLOR_FORMAT}],
            depth_stencil={
                "format": DEPTH_FORMAT,
                "depth_test_enable": False,
                "depth_write_enable": False,
            },
        )
        blend_target = {
            "format": COLOR_FORMAT,
            "enable_blend": True,
            "color": {
                "src_factor": spy.BlendFactor.src_alpha,
                "dst_factor": spy.BlendFactor.inv_src_alpha,
                "op": spy.BlendOp.add,
            },
            "alpha": {
                "src_factor": spy.BlendFactor.one,
                "dst_factor": spy.BlendFactor.inv_src_alpha,
                "op": spy.BlendOp.add,
            },
        }
        gizmo_depth = {
            "format": DEPTH_FORMAT,
            "depth_test_enable": True,
            "depth_write_enable": False,
            "depth_func": spy.ComparisonFunc.less_equal,
        }
        self.gizmo_pipeline = self.device.create_render_pipeline(
            program=self.gizmo_program,
            input_layout=self.gizmo_layout,
            primitive_topology=spy.PrimitiveTopology.line_list,
            targets=[blend_target],
            depth_stencil=gizmo_depth,
        )
        self.grid_program = self.device.load_program("grid.slang", ["grid_vs", "grid_fs"])
        self.grid_layout = self.device.create_input_layout(
            input_elements=[
                {"semantic_name": "POSITION", "semantic_index": 0,
                 "format": spy.Format.rgb32_float, "offset": 0},
            ],
            vertex_streams=[{"stride": 12}],
        )
        self.grid_pipeline = self.device.create_render_pipeline(
            program=self.grid_program,
            input_layout=self.grid_layout,
            primitive_topology=spy.PrimitiveTopology.triangle_list,
            targets=[blend_target],
            rasterizer={"cull_mode": spy.CullMode.none},
            depth_stencil=gizmo_depth,
        )

    def scene_pipeline(self):
        key = self.wireframe
        pipe = self.scene_pipelines.get(key)
        if pipe is not None:
            return pipe
        rasterizer = {"cull_mode": spy.CullMode.none}
        if self.wireframe:
            rasterizer["fill_mode"] = spy.FillMode.wireframe
        pipe = self.device.create_render_pipeline(
            program=self.scene_program,
            input_layout=self.input_layout,
            primitive_topology=spy.PrimitiveTopology.triangle_list,
            targets=[{"format": COLOR_FORMAT}],
            rasterizer=rasterizer,
            depth_stencil={
                "format": DEPTH_FORMAT,
                "depth_test_enable": True,
                "depth_write_enable": True,
                "depth_func": spy.ComparisonFunc.less,
            },
        )
        self.scene_pipelines[key] = pipe
        return pipe

    # ---- camera / matrices ------------------------------------------------

    def view_proj(self, width, height):
        eye = self.camera.position
        aspect = max(width / max(height, 1), 1e-3)
        view = spy.math.matrix_from_look_at(
            spy.float3(*eye), spy.float3(*self.camera.target()), spy.float3(0, 1, 0)
        )
        d = float(np.linalg.norm(eye - self.mesh.center))
        r = self.mesh.radius
        far = max(d + r * 2.5, r * 0.01)
        near = max(d - r * 2.5, far * 1e-4)
        proj = spy.math.perspective(self.camera.fov_y, aspect, near, far)
        return spy.math.mul(proj, view), spy.float3(*eye)

    # ---- targets ----------------------------------------------------------

    def ensure_targets(self, width, height):
        width = max(16, int(width))
        height = max(16, int(height))
        if (self.color_texture is not None
                and self.color_texture.width == width
                and self.color_texture.height == height):
            return
        self.device.wait()
        self.color_texture = self.device.create_texture(
            format=COLOR_FORMAT, width=width, height=height,
            usage=spy.TextureUsage.render_target | spy.TextureUsage.shader_resource
            | spy.TextureUsage.copy_source,
            label="scene_color",
        )
        self.depth_texture = self.device.create_texture(
            format=DEPTH_FORMAT, width=width, height=height,
            usage=spy.TextureUsage.depth_stencil, label="scene_depth",
        )

    # ---- render -----------------------------------------------------------

    def render(self, encoder):
        color = self.color_texture
        depth = self.depth_texture
        vp, eye = self.view_proj(color.width, color.height)
        headlight_dir = self.camera.forward()
        cam_right, cam_up = self.camera.basis()

        pass_desc = {
            "color_attachments": [{
                "view": color.create_view({}),
                "clear_value": [0.10, 0.11, 0.13, 1.0],
                "load_op": spy.LoadOp.clear,
                "store_op": spy.StoreOp.store,
            }],
            "depth_stencil_attachment": {
                "view": depth.create_view({}),
                "depth_load_op": spy.LoadOp.clear,
                "depth_store_op": spy.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        }
        viewport_full = {
            "viewports": [spy.Viewport.from_size(color.width, color.height)],
            "scissor_rects": [spy.ScissorRect.from_size(color.width, color.height)],
        }
        with encoder.begin_render_pass(pass_desc) as rp:
            # Environment skybox, only in Lit mode (other modes use the flat clear).
            if self.shading_mode == 0:
                rp.set_render_state(viewport_full)
                sky = rp.bind_pipeline(self.sky_pipeline)
                scur = spy.ShaderCursor(sky)
                scur.g_env = self.env_view
                scur.g_env_sampler = self.env_sampler
                scur.g_inv_view_proj = spy.math.inverse(vp)
                scur.g_eye = eye
                scur.g_exposure = self.exposure
                rp.draw({"vertex_count": 3})

            if self.show_meshes:
                rp.set_render_state({
                    "viewports": [spy.Viewport.from_size(color.width, color.height)],
                    "scissor_rects": [spy.ScissorRect.from_size(color.width, color.height)],
                    "vertex_buffers": [self.vertex_buffer],
                    "index_buffer": self.index_buffer,
                    "index_format": spy.IndexFormat.uint32,
                })
                obj = rp.bind_pipeline(self.scene_pipeline())
                cur = spy.ShaderCursor(obj)
                cur.g_camera = {"view_proj": vp, "eye": eye, "pad": 0.0}
                cur.g_dir_lights = self.dir_buffer
                cur.g_point_lights = self.point_buffer
                cur.g_sh = self.sh_buffer
                cur.g_num_dir = self.num_dir
                cur.g_num_point = self.num_point
                cur.g_shading_mode = self.shading_mode
                cur.g_alpha_cutout = 1 if self.alpha_cutout else 0
                cur.g_headlight_dir = spy.float3(*headlight_dir)
                cur.g_cam_right = spy.float3(*cam_right)
                cur.g_cam_up = spy.float3(*cam_up)
                cur.g_headlight_intensity = self.headlight
                cur.g_ambient = self.ambient
                cur.g_sampler = self.sampler
                cur.g_shadow_map = self.shadow_view
                cur.g_shadow_sampler = self.shadow_sampler
                cur.g_light_vp = self.light_vp
                cur.g_has_shadow = 1 if self.has_shadow else 0

                for prim in self.mesh.primitives:
                    factor = (1.0, 1.0, 1.0, 1.0)
                    base_view = mr_view = occ_view = nrm_view = None
                    metallic = roughness = occ_strength = 1.0
                    mi = prim.material_index
                    if 0 <= mi < len(self.material_factors):
                        factor = self.material_factors[mi]
                        base_view = self.material_views[mi]
                        mr_view = self.material_mr_views[mi]
                        occ_view = self.material_occ_views[mi]
                        nrm_view = self.material_normal_views[mi]
                        metallic = self.material_metallic[mi]
                        roughness = self.material_roughness[mi]
                        occ_strength = self.material_occ_strength[mi]
                    cur.g_base_color = base_view if base_view is not None else self.white_view
                    cur.g_mr_texture = mr_view if mr_view is not None else self.white_view
                    cur.g_occ_texture = occ_view if occ_view is not None else self.white_view
                    cur.g_normal_texture = nrm_view if nrm_view is not None else self.white_view
                    cur.g_draw = {
                        "base_color": spy.float4(*factor),
                        "has_texture": 1 if base_view is not None else 0,
                        "has_mr_texture": 1 if mr_view is not None else 0,
                        "has_occ_texture": 1 if occ_view is not None else 0,
                        "has_normal_texture": 1 if nrm_view is not None else 0,
                        "metallic": float(metallic),
                        "roughness": float(roughness),
                        "occlusion_strength": float(occ_strength),
                        "pad1": 0.0,
                    }
                    rp.draw_indexed({
                        "vertex_count": prim.index_count,
                        "start_index_location": prim.index_offset,
                    })

            full_state = {
                "viewports": [spy.Viewport.from_size(color.width, color.height)],
                "scissor_rects": [spy.ScissorRect.from_size(color.width, color.height)],
            }

            gobj = rp.bind_pipeline(self.grid_pipeline)
            gcur = spy.ShaderCursor(gobj)
            gcur.g_camera = {"view_proj": vp}
            gcur.g_cell = self.grid_cell
            gcur.g_center = self.grid_fade_center
            gcur.g_fade_radius = self.grid_fade_radius
            gcur.g_color = spy.float3(0.45, 0.47, 0.54)
            gcur.g_line_width = 0.9
            rp.set_render_state({**full_state, "vertex_buffers": [self.grid_buffer]})
            rp.draw({"vertex_count": 6})

            lines = []
            if self.show_cameras and self.line_gizmos["cameras"]:
                lines.append(self.line_gizmos["cameras"])
            if self.show_lights and self.line_gizmos["lights"]:
                lines.append(self.line_gizmos["lights"])
            if lines:
                gobj = rp.bind_pipeline(self.gizmo_pipeline)
                gcur = spy.ShaderCursor(gobj)
                gcur.g_camera = {"view_proj": vp}
                gcur.g_fade_enable = 0
                gcur.g_fade_center = spy.float3(0, 0, 0)
                gcur.g_fade_radius = 1.0
                for buf, count in lines:
                    rp.set_render_state({**full_state, "vertex_buffers": [buf]})
                    rp.draw({"vertex_count": count})

    # ---- input ------------------------------------------------------------

    MOVE_KEYS = None

    def move_keys(self):
        if Viewer.MOVE_KEYS is None:
            kc = spy.KeyCode
            Viewer.MOVE_KEYS = (kc.w, kc.a, kc.s, kc.d, kc.q, kc.e)
        return Viewer.MOVE_KEYS

    def on_keyboard(self, event):
        et = spy.KeyboardEventType
        consumed = self.ui.handle_keyboard_event(event)
        if event.type == et.key_release and event.key in self.move_keys():
            self.keys_held.discard(event.key)
        if consumed:
            return
        if event.type == et.key_press:
            if event.key == spy.KeyCode.escape:
                self.terminate = True
            elif event.key == spy.KeyCode.f:
                self.camera.frame(self.mesh.center, self.mesh.radius)
            elif event.key in self.move_keys():
                self.keys_held.add(event.key)

    def update_move_scale(self):
        # Speed adapts to scene size (floor) and distance to it, so the same
        # feel holds for a tiny prop or a whole city block.
        dist = float(np.linalg.norm(self.camera.position - self.mesh.center))
        self.camera.move_scale = max(self.mesh.radius * 0.1, dist)

    def apply_movement(self, dt):
        self.update_move_scale()
        kc = spy.KeyCode
        fwd = (kc.w in self.keys_held) - (kc.s in self.keys_held)
        rgt = (kc.d in self.keys_held) - (kc.a in self.keys_held)
        upd = (kc.e in self.keys_held) - (kc.q in self.keys_held)
        if not (fwd or rgt or upd):
            return
        step = self.camera.move_scale * self.move_speed * dt
        self.camera.move(fwd * step, rgt * step, upd * step)

    def on_mouse(self, event):
        et = spy.MouseEventType
        if event.type == et.button_down:
            consumed = self.ui.handle_mouse_event(event)
            if not consumed:
                self.last_mouse = (event.pos.x, event.pos.y)
                if event.button == spy.MouseButton.left:
                    self.dragging["rotate"] = True
                elif event.button == spy.MouseButton.middle:
                    self.dragging["pan"] = True
            return
        if event.type == et.button_up:
            self.dragging["rotate"] = False
            self.dragging["pan"] = False
            self.ui.handle_mouse_event(event)
            return
        if event.type == et.move:
            if any(self.dragging.values()) and self.last_mouse is not None:
                dx = event.pos.x - self.last_mouse[0]
                dy = event.pos.y - self.last_mouse[1]
                if self.dragging["rotate"]:
                    self.camera.rotate(dx, dy)
                if self.dragging["pan"]:
                    self.camera.pan(dx, dy)
                self.last_mouse = (event.pos.x, event.pos.y)
            else:
                self.ui.handle_mouse_event(event)
            return
        if event.type == et.scroll:
            if not self.ui.handle_mouse_event(event):
                self.camera.zoom(event.scroll.y)

    def on_resize(self, width, height):
        self.device.wait()
        if width > 0 and height > 0:
            self.surface.configure(width=width, height=height, vsync=True)
        else:
            self.surface.unconfigure()

    # ---- ui ---------------------------------------------------------------

    def build_ui(self):
        screen = self.ui.screen
        self.dock = spy.ui.DockSpace(screen)
        self.dock.passthru_central_node = True
        self.needs_layout = True
        self.layout_step = 0

        # Buttons / frames default to a translucent fill; make them opaque so the
        # chrome-less viewport overlay widgets don't show the scene through them.
        style = self.ui.style
        for col in (spy.ui.Col.button, spy.ui.Col.button_hovered, spy.ui.Col.button_active,
                    spy.ui.Col.frame_bg, spy.ui.Col.frame_bg_hovered, spy.ui.Col.frame_bg_active):
            c = style.get_color(col)
            style.set_color(col, spy.float4(c.x, c.y, c.z, 1.0))
        style.frame_padding = spy.float2(8.0, 4.0)
        style.button_text_align = spy.float2(0.5, 0.5)

        # Settings "popup": a floating window toggled by the gear overlay button.
        settings = spy.ui.Window(screen, "Settings", size=spy.float2(300, 430))
        settings.visible = False
        self.settings = settings

        spy.ui.CheckBox(settings, "Wireframe", value=self.wireframe,
                        callback=lambda v: setattr(self, "wireframe", bool(v)))
        spy.ui.CheckBox(settings, "Alpha Cutout", value=self.alpha_cutout,
                        callback=lambda v: setattr(self, "alpha_cutout", bool(v)))
        spy.ui.SliderFloat(settings, "Move Speed", value=self.move_speed, min=0.1, max=10.0,
                           callback=lambda v: setattr(self, "move_speed", float(v)))
        spy.ui.Separator(settings)

        spy.ui.Text(settings, "Show")
        spy.ui.CheckBox(settings, "Meshes", value=self.show_meshes,
                        callback=lambda v: setattr(self, "show_meshes", bool(v)))
        spy.ui.CheckBox(settings, "Cameras", value=self.show_cameras,
                        callback=lambda v: setattr(self, "show_cameras", bool(v)))
        spy.ui.CheckBox(settings, "Lights", value=self.show_lights,
                        callback=lambda v: setattr(self, "show_lights", bool(v)))

        if self.scene.cameras:
            spy.ui.Separator(settings)
            spy.ui.Text(settings, "Jump to Camera")
            for cam in self.scene.cameras:
                spy.ui.Button(settings, cam.name,
                              callback=lambda c=cam: self.camera.set_from_gltf(c))

        self.overlay = spy.ui.Window(screen, "overlay", overlay=True)
        self.overlay.font = "icon_large"
        spy.ui.Button(self.overlay, ICON_GEAR, callback=self.toggle_settings)

        self.shading_overlay = spy.ui.Window(screen, "shading_overlay", overlay=True)
        combo = spy.ui.ComboBox(self.shading_overlay, "##shading", value=self.shading_mode,
                                items=SHADING_MODES,
                                callback=lambda v: setattr(self, "shading_mode", int(v)))
        combo.width = 210.0

        self.info_window = spy.ui.Window(screen, "Info", size=spy.float2(300, 240))
        self.build_info(self.info_window)

        self.scene_window = spy.ui.Window(screen, "Scene", size=spy.float2(300, 360))
        self.build_tree(self.scene_window, self.scene.nodes)

    @staticmethod
    def format_size(n):
        size = float(n)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024.0 or unit == "GB":
                return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
            size /= 1024.0

    def build_info(self, window):
        bsize = self.mesh.bounds_max - self.mesh.bounds_min
        n_tex = sum(1 for v in self.material_views if v is not None)
        rows = [
            ("File", self.path.name),
            ("Size", self.format_size(self.path.stat().st_size)),
            ("Triangles", f"{self.mesh.triangle_count:,}"),
            ("Vertices", f"{self.mesh.vertex_count:,}"),
            ("Primitives", f"{len(self.mesh.primitives):,}"),
            ("Materials", str(len(self.mesh.materials))),
            ("Textures", str(n_tex)),
            ("Cameras", str(len(self.scene.cameras))),
            ("Lights", str(len(self.scene.lights))),
            ("Bounds", f"{bsize[0]:.1f} x {bsize[1]:.1f} x {bsize[2]:.1f}"),
        ]
        table = spy.ui.Table(window, "stats", columns=2, headers=["Property", "Value"])
        for key, value in rows:
            spy.ui.Text(table, key)
            spy.ui.Text(table, str(value))

    def toggle_settings(self):
        self.settings.visible = not self.settings.visible
        if self.settings.visible:
            # Position once on open so the window is freely movable afterwards.
            x = self.info_window.position.x + self.info_window.size.x
            self.settings.position = spy.float2(x + 6.0, 44.0)

    def apply_layout(self):
        if not self.needs_layout:
            return
        root = self.dock.dock_id
        if root == 0:
            return
        # Nodes must be realized before they can be split, so step over frames.
        if self.layout_step == 0:
            self.dock.request_split_horizontal(0.24)
            self.layout_step = 1
            return
        left = self.dock.left_dock_id
        if left == 0:
            return
        info_node, scene_node = self.dock.split_node(left, vertical=True, ratio=0.3)
        self.info_window.dock_id = info_node
        self.scene_window.dock_id = scene_node
        self.needs_layout = False

    def build_tree(self, parent, nodes):
        for n in nodes:
            label = n.name
            if n.camera:
                label = f"{ICON_CAMERA}  {n.name}"
            elif n.light:
                icon = ICON_SUN if n.light.type == "directional" else ICON_LIGHT
                label = f"{icon}  {n.name}"
            elif n.mesh_name:
                label = f"{ICON_MESH}  {n.name}"
            node = spy.ui.TreeNode(parent, label)
            if n.mesh_name:
                spy.ui.Text(node, f"{n.triangle_count:,} triangles"
                                  f"{' · UVs' if n.has_uvs else ''}")
            if n.light:
                spy.ui.Text(node, f"Color {tuple(round(c, 2) for c in n.light.color)} · "
                                  f"Intensity {n.light.intensity:g}")
            if n.camera:
                spy.ui.Button(node, "Look Through",
                              callback=lambda c=n.camera: self.camera.set_from_gltf(c))
            if n.children:
                self.build_tree(node, n.children)

    # ---- loop -------------------------------------------------------------

    def run(self, max_frames=0, screenshot=""):
        frame = 0
        timer = spy.Timer()
        while not self.window.should_close() and not self.terminate:
            self.window.process_events()
            dt = timer.elapsed_s()
            timer.reset()
            self.apply_movement(dt)

            if not self.surface.config or self.window.width == 0 or self.window.height == 0:
                continue
            surface_texture = self.surface.acquire_next_image()
            if not surface_texture:
                continue

            self.ensure_targets(surface_texture.width, surface_texture.height)
            encoder = self.device.create_command_encoder()
            self.render(encoder)
            encoder.blit(surface_texture, self.color_texture)

            self.ui.begin_frame(surface_texture.width, surface_texture.height)
            self.apply_layout()
            viewport_x = self.info_window.position.x + self.info_window.size.x
            self.overlay.position = spy.float2(viewport_x + 6.0, 6.0)
            self.shading_overlay.position = spy.float2(
                surface_texture.width - self.shading_overlay.size.x - 6.0, 6.0)
            self.ui.end_frame(surface_texture, encoder)

            self.device.submit_command_buffer(encoder.finish())
            frame += 1
            if max_frames and frame >= max_frames and screenshot:
                self.save_screenshot(surface_texture, screenshot)
            del surface_texture
            self.surface.present()
            if max_frames and frame >= max_frames:
                self.terminate = True

    def save_screenshot(self, texture, path):
        self.device.wait()
        bmp = texture.to_bitmap().convert(
            spy.Bitmap.PixelFormat.rgb, spy.Bitmap.ComponentType.uint8, srgb_gamma=False
        )
        arr = np.array(bmp)
        if "bgr" in str(texture.format).lower():
            arr = arr[..., ::-1]
        from PIL import Image
        Image.fromarray(np.ascontiguousarray(arr), "RGB").save(path)
