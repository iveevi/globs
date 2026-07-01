import logging
from pathlib import Path

import numpy as np
import slangpy as spy

log = logging.getLogger("glb.viewer")

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

PATHTRACE_MODE = 8

# Font Awesome glyphs (from the bundled Nerd Font) for the light markers.
ICON_SUN = "\uf185"    # directional (sun)
ICON_LIGHT = "\uf0eb"  # point (lightbulb)
ICON_SPOT = "\uf140"   # spot (bullseye / aimed)

# Shading dropdown rows -> shader mode index: render/lighting modes first, then
# the raw material-property inspectors.
SHADING_ITEMS = [
    ("Solid", 1),
    ("Shaded", 0),
    ("Path Traced", 8),
    ("Albedo", 2),
    ("Geometric Normals", 3),
    ("Shading Normals", 4),
    ("Metallic", 5),
    ("Roughness", 6),
    ("Occlusion", 7),
    ("Emission", 9),
]


class Viewer:
    def __init__(self, glb_path, headless=False):
        self.path = Path(glb_path)
        self.headless = headless
        self.window = None
        self.surface = None
        self.ui = None
        if not headless:
            self.window = spy.Window(
                width=1600, height=900, title=f"GLB Viewer — {self.path.name}", resizable=True
            )
        self.device = spy.Device(
            enable_debug_layers=False,
            compiler_options={"include_paths": [SHADER_DIR]},
        )
        if not self.device.has_feature(spy.Feature.rasterization):
            raise RuntimeError("device has no rasterization support")

        self.vsync = True
        if not headless:
            self.surface = self.device.create_surface(self.window)
            self.surface.configure(width=self.window.width, height=self.window.height,
                                   vsync=self.vsync)
            self.ui = spy.ui.Context(self.device)
            self.ui.ini_filename = None
            self.ui.add_font("ui", str(FONT_DIR / "FiraSansCondensed-Regular.ttf"),
                             16, is_default=True)

        self.scene = load_scene(glb_path)
        self.mesh = self.scene.mesh
        self.upload_geometry()
        self.upload_materials()
        self.upload_lights()
        self.load_ibl()
        self.build_gizmos()
        self.build_pipelines()
        self.build_shadows()
        self.build_pathtracer()

        self.camera = FlyCamera()
        self.camera.frame(self.mesh.center, self.mesh.radius)

        self.shading_mode = 0
        self.pt_bounces = 8
        self.show_env = True
        self.wireframe = False
        self.alpha_cutout = True
        self.headlight = 0.9
        self.ambient = 0.25
        self.show_meshes = True
        self.show_cameras = True
        self.show_lights = True
        self.move_speed = 1.0

        self.color_texture = None
        self.depth_texture = None
        self.keys_held = set()
        self.want_screenshot = False
        self.screenshot_index = 0
        self.dragging = {"rotate": False, "pan": False}
        self.last_mouse = None
        self.terminate = False

        if not headless:
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
        # Per-vertex material index (a second vertex stream). Vertices are not
        # shared across primitives, so each maps to exactly one material. This
        # lets the whole scene draw in one call with a bindless material table.
        default_index = len(self.mesh.materials)
        vmat = np.full(self.mesh.vertex_count, default_index, dtype=np.uint32)
        idx = self.mesh.indices
        for prim in self.mesh.primitives:
            mi = prim.material_index
            mi = mi if 0 <= mi < default_index else default_index
            span = idx[prim.index_offset:prim.index_offset + prim.index_count]
            vmat[span] = mi
        self.vmat_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.vertex_buffer, label="vertex_materials",
            data=vmat,
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
        self.scene_input_layout = self.device.create_input_layout(
            input_elements=[
                {"semantic_name": "POSITION", "semantic_index": 0,
                 "format": spy.Format.rgb32_float, "offset": 0},
                {"semantic_name": "NORMAL", "semantic_index": 0,
                 "format": spy.Format.rgb32_float, "offset": 12},
                {"semantic_name": "TEXCOORD", "semantic_index": 0,
                 "format": spy.Format.rg32_float, "offset": 24},
                {"semantic_name": "MATERIAL", "semantic_index": 0,
                 "format": spy.Format.r32_uint, "offset": 0, "buffer_slot_index": 1},
            ],
            vertex_streams=[{"stride": VERTEX_STRIDE}, {"stride": 4}],
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
        self.material_emissive_views = []
        self.material_factors = []
        self.material_metallic = []
        self.material_roughness = []
        self.material_occ_strength = []
        self.material_emissive = []
        for mat in self.mesh.materials:
            self.material_factors.append(mat.base_color_factor)
            self.material_metallic.append(mat.metallic_factor)
            self.material_roughness.append(mat.roughness_factor)
            self.material_occ_strength.append(mat.occlusion_strength)
            self.material_emissive.append(
                tuple(c * mat.emissive_strength for c in mat.emissive_factor))
            self.material_views.append(self.upload_image(mat.base_color_image, srgb=True))
            self.material_mr_views.append(self.upload_image(mat.metallic_roughness_image))
            self.material_occ_views.append(self.upload_image(mat.occlusion_image))
            self.material_normal_views.append(self.upload_image(mat.normal_image))
            self.material_emissive_views.append(
                self.upload_image(mat.emissive_image, srgb=True))

    def upload_image(self, image, srgb=False):
        if image is None:
            return None
        img = np.ascontiguousarray(image)
        w, h = int(img.shape[1]), int(img.shape[0])
        mips = int(max(w, h)).bit_length()
        # Full mip chain: minified material textures (esp. normal maps) alias
        # badly without it, producing speckled / inverted shading normals.
        tex = self.device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba8_unorm_srgb if srgb else spy.Format.rgba8_unorm,
            width=w, height=h, mip_count=mips,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.render_target
            | spy.TextureUsage.copy_destination | spy.TextureUsage.copy_source,
            data=img, label="material_texture",
        )
        encoder = self.device.create_command_encoder()
        encoder.generate_mips(tex)
        self.device.submit_command_buffer(encoder.finish())
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
        }

        # Light outlines: drawn as thick screen-space lines, so the segment
        # vertices live in a structured buffer the thick-line shader pulls from.
        light_verts = gizmos.build_lights(self.scene.lights, r)
        self.num_light_segments = 0
        self.light_line_buffer = None
        if light_verts is not None and len(light_verts):
            data = np.ascontiguousarray(light_verts.astype(np.float32))
            self.num_light_segments = data.shape[0] // 2
            self.light_line_buffer = self.device.create_buffer(
                usage=spy.BufferUsage.shader_resource, struct_size=24,
                element_count=data.shape[0], data=data.reshape(-1),
                label="light_lines")

        # Light markers: a screen-space billboard per light, sampling the Font
        # Awesome glyph atlas (type 0 = directional, 1 = point, 2 = spot).
        type_id = {"directional": 0.0, "point": 1.0, "spot": 2.0}
        icons = np.zeros((max(1, len(self.scene.lights)), 8), dtype=np.float32)
        for i, li in enumerate(self.scene.lights):
            icons[i, 0:3] = li.position
            icons[i, 3] = type_id.get(li.type, 1.0)
            icons[i, 4:7] = li.color
        self.num_light_icons = len(self.scene.lights)
        self.light_icon_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource, struct_size=32,
            element_count=icons.shape[0], data=icons.reshape(-1), label="light_icons")
        self.build_light_icon_atlas()

    def build_light_icon_atlas(self):
        from PIL import Image, ImageDraw, ImageFont
        tile = 64
        glyphs = [ICON_SUN, ICON_LIGHT, ICON_SPOT]
        atlas = Image.new("RGBA", (tile * 3, tile), (0, 0, 0, 0))
        draw = ImageDraw.Draw(atlas)
        font = ImageFont.truetype(
            str(FONT_DIR / "CaskaydiaCoveNerdFontMono-Regular.ttf"), 52)
        for i, g in enumerate(glyphs):
            bbox = draw.textbbox((0, 0), g, font=font)
            gw, gh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = i * tile + (tile - gw) // 2 - bbox[0]
            y = (tile - gh) // 2 - bbox[1]
            draw.text((x, y), g, font=font, fill=(255, 255, 255, 255))
        data = np.ascontiguousarray(np.array(atlas, dtype=np.uint8))
        tex = self.device.create_texture(
            type=spy.TextureType.texture_2d, format=spy.Format.rgba8_unorm,
            width=tile * 3, height=tile, usage=spy.TextureUsage.shader_resource,
            data=data, label="light_icon_atlas")
        self.icon_atlas_view = tex.create_view({})
        self.icon_sampler = self.device.create_sampler(
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
            label="icon_sampler")

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

    def build_accel(self):
        # BLAS over the single world-space vertex/index buffer, wrapped in a
        # one-instance TLAS. Geometry is static, so this is built once.
        tri_input = spy.AccelerationStructureBuildInputTriangles({
            "vertex_buffers": [self.vertex_buffer],
            "vertex_format": spy.Format.rgb32_float,
            "vertex_count": self.mesh.vertex_count,
            "vertex_stride": VERTEX_STRIDE,
            "index_buffer": self.index_buffer,
            "index_format": spy.IndexFormat.uint32,
            "index_count": int(self.mesh.indices.shape[0]),
            "flags": spy.AccelerationStructureGeometryFlags.none,
        })
        blas_desc = spy.AccelerationStructureBuildDesc({"inputs": [tri_input]})
        blas_sizes = self.device.get_acceleration_structure_sizes(blas_desc)
        blas_scratch = self.device.create_buffer(
            size=blas_sizes.scratch_size, usage=spy.BufferUsage.unordered_access,
            label="blas_scratch")
        self.blas = self.device.create_acceleration_structure(
            kind=spy.AccelerationStructureKind.bottom_level,
            size=blas_sizes.acceleration_structure_size, label="blas")
        encoder = self.device.create_command_encoder()
        encoder.build_acceleration_structure(
            desc=blas_desc, dst=self.blas, src=None, scratch_buffer=blas_scratch)
        self.device.submit_command_buffer(encoder.finish())
        self.device.wait()

        instances = self.device.create_acceleration_structure_instance_list(1)
        instances.write(0, {
            "transform": spy.float3x4.identity(),
            "instance_id": 0,
            "instance_mask": 0xFF,
            "instance_contribution_to_hit_group_index": 0,
            "flags": spy.AccelerationStructureInstanceFlags.none,
            "acceleration_structure": self.blas.handle,
        })
        tlas_desc = spy.AccelerationStructureBuildDesc(
            {"inputs": [instances.build_input_instances()]})
        tlas_sizes = self.device.get_acceleration_structure_sizes(tlas_desc)
        tlas_scratch = self.device.create_buffer(
            size=tlas_sizes.scratch_size, usage=spy.BufferUsage.unordered_access,
            label="tlas_scratch")
        self.tlas = self.device.create_acceleration_structure(
            kind=spy.AccelerationStructureKind.top_level,
            size=tlas_sizes.acceleration_structure_size, label="tlas")
        encoder = self.device.create_command_encoder()
        encoder.build_acceleration_structure(
            desc=tlas_desc, dst=self.tlas, src=None, scratch_buffer=tlas_scratch)
        self.device.submit_command_buffer(encoder.finish())
        self.device.wait()

    def build_pt_materials(self, layout):
        # Bindless material table: per-material factors plus descriptor handles
        # for the base-color and metallic-roughness textures.
        n = len(self.mesh.materials) + 1  # + default for prims with no material
        mat_type = layout.get_type_layout(
            layout.find_type_by_name("StructuredBuffer<PTMaterial>")).element_type_layout
        buf = self.device.create_buffer(
            size=n * mat_type.stride,
            usage=spy.BufferUsage.shader_resource, label="pt_materials")
        cur = spy.BufferCursor(mat_type, buf, load_before_write=False)
        samp = self.sampler.descriptor_handle
        white = self.white_view.descriptor_handle_ro
        for i in range(n):
            base_view = self.material_views[i] if i < len(self.material_views) else None
            mr_view = self.material_mr_views[i] if i < len(self.material_mr_views) else None
            emi_view = self.material_emissive_views[i] if i < len(self.material_emissive_views) else None
            factor = self.material_factors[i] if i < len(self.material_factors) else (1, 1, 1, 1)
            metallic = self.material_metallic[i] if i < len(self.material_metallic) else 1.0
            roughness = self.material_roughness[i] if i < len(self.material_roughness) else 1.0
            emissive = self.material_emissive[i] if i < len(self.material_emissive) else (0, 0, 0)
            cur[i].base_color = spy.float4(*factor)
            cur[i].metallic = float(metallic)
            cur[i].roughness = float(roughness)
            cur[i].has_base_tex = 1 if base_view is not None else 0
            cur[i].has_mr_tex = 1 if mr_view is not None else 0
            cur[i].emissive = spy.float3(*emissive)
            cur[i].has_emissive_tex = 1 if emi_view is not None else 0
            cur[i].base_tex = base_view.descriptor_handle_ro if base_view is not None else white
            cur[i].mr_tex = mr_view.descriptor_handle_ro if mr_view is not None else white
            cur[i].emissive_tex = emi_view.descriptor_handle_ro if emi_view is not None else white
            cur[i].samp = samp
        cur.apply()
        return buf

    def build_pathtracer(self):
        self.pt_ok = False
        for feat in (spy.Feature.ray_tracing, spy.Feature.acceleration_structure,
                     spy.Feature.bindless):
            if not self.device.has_feature(feat):
                return
        self.build_accel()

        tri_mat = np.zeros(self.mesh.triangle_count, dtype=np.uint32)
        default_index = len(self.mesh.materials)
        for prim in self.mesh.primitives:
            start = prim.index_offset // 3
            count = prim.index_count // 3
            mi = prim.material_index
            tri_mat[start:start + count] = mi if 0 <= mi < default_index else default_index
        self.pt_tri_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource, struct_size=4,
            element_count=int(tri_mat.shape[0]), data=tri_mat, label="pt_tri_material")

        self.pt_program = self.device.load_program(
            "pathtrace.slang",
            ["rt_raygen", "rt_miss", "rt_closesthit", "rt_anyhit"])
        self.pt_pipeline = self.device.create_ray_tracing_pipeline(
            program=self.pt_program,
            hit_groups=[spy.HitGroupDesc(
                hit_group_name="hit_group",
                closest_hit_entry_point="rt_closesthit",
                any_hit_entry_point="rt_anyhit")],
            max_recursion=1,
            max_ray_payload_size=48,
        )
        self.pt_shader_table = self.device.create_shader_table(
            program=self.pt_program,
            ray_gen_entry_points=["rt_raygen"],
            miss_entry_points=["rt_miss"],
            hit_group_names=["hit_group"],
        )
        self.pt_mat_buffer = self.build_pt_materials(self.pt_program.layout)
        self.pt_ok = True

    def build_scene_materials(self):
        # Bindless material table for the raster path (one entry per material,
        # plus a default at the end for primitives with no material).
        n = len(self.mesh.materials) + 1
        layout = self.scene_program.layout
        mat_type = layout.get_type_layout(
            layout.find_type_by_name("StructuredBuffer<Material>")).element_type_layout
        buf = self.device.create_buffer(
            size=n * mat_type.stride,
            usage=spy.BufferUsage.shader_resource, label="scene_materials")
        cur = spy.BufferCursor(mat_type, buf, load_before_write=False)
        white = self.white_view.descriptor_handle_ro

        def handle(v):
            return v.descriptor_handle_ro if v is not None else white

        for i in range(n):
            has = i < len(self.material_factors)
            base = self.material_views[i] if has else None
            mr = self.material_mr_views[i] if has else None
            occ = self.material_occ_views[i] if has else None
            nrm = self.material_normal_views[i] if has else None
            emi = self.material_emissive_views[i] if has else None
            cur[i].base_color = spy.float4(*(self.material_factors[i] if has else (1, 1, 1, 1)))
            cur[i].emissive = spy.float3(*(self.material_emissive[i] if has else (0, 0, 0)))
            cur[i].metallic = float(self.material_metallic[i] if has else 1.0)
            cur[i].roughness = float(self.material_roughness[i] if has else 1.0)
            cur[i].occlusion_strength = float(self.material_occ_strength[i] if has else 1.0)
            cur[i].has_texture = 1 if base is not None else 0
            cur[i].has_mr_texture = 1 if mr is not None else 0
            cur[i].has_occ_texture = 1 if occ is not None else 0
            cur[i].has_normal_texture = 1 if nrm is not None else 0
            cur[i].has_emissive = 1 if emi is not None else 0
            cur[i].base_tex = handle(base)
            cur[i].mr_tex = handle(mr)
            cur[i].occ_tex = handle(occ)
            cur[i].normal_tex = handle(nrm)
            cur[i].emissive_tex = handle(emi)
        cur.apply()
        self.scene_mat_buffer = buf

    def build_pipelines(self):
        self.scene_program = self.device.load_program("scene.slang", ["vs_main", "fs_main"])
        self.scene_pipelines = {}
        self.build_scene_materials()
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
        self.icon_program = self.device.load_program(
            "light_icons.slang", ["icon_vs", "icon_fs"])
        self.icon_pipeline = self.device.create_render_pipeline(
            program=self.icon_program,
            input_layout=None,
            primitive_topology=spy.PrimitiveTopology.triangle_list,
            targets=[blend_target],
            rasterizer={"cull_mode": spy.CullMode.none},
            depth_stencil={"format": DEPTH_FORMAT, "depth_test_enable": False,
                           "depth_write_enable": False},
        )
        self.thick_line_program = self.device.load_program(
            "thick_line.slang", ["line_vs", "line_fs"])
        self.thick_line_pipeline = self.device.create_render_pipeline(
            program=self.thick_line_program,
            input_layout=None,
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
            input_layout=self.scene_input_layout,
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
            | spy.TextureUsage.unordered_access | spy.TextureUsage.copy_source,
            label="scene_color",
        )
        self.depth_texture = self.device.create_texture(
            format=DEPTH_FORMAT, width=width, height=height,
            usage=spy.TextureUsage.depth_stencil, label="scene_depth",
        )
        self.accum_texture = self.device.create_texture(
            format=spy.Format.rgba32_float, width=width, height=height,
            usage=spy.TextureUsage.unordered_access | spy.TextureUsage.shader_resource,
            label="pt_accum",
        )
        self.pt_frame = 0
        self.pt_sig = None

    # ---- render -----------------------------------------------------------

    def render_pathtrace(self, encoder):
        color = self.color_texture
        vp, eye = self.view_proj(color.width, color.height)
        fwd = self.camera.forward()
        sig = (tuple(np.round(self.camera.position, 5)), tuple(np.round(fwd, 5)),
               round(float(self.exposure), 5), int(self.pt_bounces),
               bool(self.show_env), bool(self.alpha_cutout), color.width, color.height)
        if sig != self.pt_sig:
            self.pt_frame = 0
            self.pt_sig = sig
        reset = 1 if self.pt_frame == 0 else 0
        with encoder.begin_ray_tracing_pass() as rp:
            obj = rp.bind_pipeline(self.pt_pipeline, self.pt_shader_table)
            cur = spy.ShaderCursor(obj)
            cur.g_tlas = self.tlas
            cur.g_vertices = self.vertex_buffer
            cur.g_indices = self.index_buffer
            cur.g_tri_material = self.pt_tri_buffer
            cur.g_materials = self.pt_mat_buffer
            cur.g_dir_lights = self.dir_buffer
            cur.g_point_lights = self.point_buffer
            cur.g_env = self.env_view
            cur.g_env_sampler = self.env_sampler
            cur.g_accum = self.accum_texture
            cur.g_output = color
            cur.g_inv_view_proj = spy.math.inverse(vp)
            cur.g_eye = eye
            cur.g_frame = self.pt_frame
            cur.g_reset = reset
            cur.g_bounces = int(self.pt_bounces)
            cur.g_num_dir = self.num_dir
            cur.g_num_point = self.num_point
            cur.g_exposure = self.exposure
            cur.g_use_env = 1 if self.show_env else 0
            cur.g_alpha_cutout = 1 if self.alpha_cutout else 0
            rp.dispatch_rays(0, [color.width, color.height, 1])
        self.pt_frame += 1

    def render(self, encoder):
        if self.shading_mode == PATHTRACE_MODE and getattr(self, "pt_ok", False):
            self.render_pathtrace(encoder)
            return
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
            # Environment skybox, only in Shaded mode (other modes use the flat clear).
            if self.shading_mode == 0 and self.show_env:
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
                    "vertex_buffers": [self.vertex_buffer, self.vmat_buffer],
                    "index_buffer": self.index_buffer,
                    "index_format": spy.IndexFormat.uint32,
                })
                obj = rp.bind_pipeline(self.scene_pipeline())
                cur = spy.ShaderCursor(obj)
                cur.g_camera = {"view_proj": vp, "eye": eye, "pad": 0.0}
                cur.g_materials = self.scene_mat_buffer
                cur.g_dir_lights = self.dir_buffer
                cur.g_point_lights = self.point_buffer
                cur.g_sh = self.sh_buffer
                cur.g_num_dir = self.num_dir
                cur.g_num_point = self.num_point
                cur.g_shading_mode = self.shading_mode
                cur.g_use_env = 1 if self.show_env else 0
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

                # Bindless: whole scene in a single indexed draw.
                rp.draw_indexed({"vertex_count": int(self.mesh.indices.shape[0]),
                                 "start_index_location": 0})

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

            # Light outlines as thick screen-space lines.
            if self.show_lights and self.num_light_segments > 0:
                rp.set_render_state(full_state)
                lobj = rp.bind_pipeline(self.thick_line_pipeline)
                lcur = spy.ShaderCursor(lobj)
                lcur.g_verts = self.light_line_buffer
                lcur.g_view_proj = vp
                lcur.g_viewport = spy.float2(color.width, color.height)
                lcur.g_width = 2.5
                rp.draw({"vertex_count": self.num_light_segments * 6})

            # Screen-space light-marker billboards (Font Awesome glyph atlas).
            if self.show_lights and self.num_light_icons > 0:
                rp.set_render_state(full_state)
                iobj = rp.bind_pipeline(self.icon_pipeline)
                icur = spy.ShaderCursor(iobj)
                icur.g_lights = self.light_icon_buffer
                icur.g_atlas = self.icon_atlas_view
                icur.g_atlas_sampler = self.icon_sampler
                icur.g_view_proj = vp
                icur.g_viewport = spy.float2(color.width, color.height)
                icur.g_size = 28.0
                rp.draw({"vertex_count": self.num_light_icons * 6})

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
            elif event.key == spy.KeyCode.p:
                self.want_screenshot = True
            elif event.key in self.move_keys():
                self.keys_held.add(event.key)

    def apply_movement(self, dt):
        kc = spy.KeyCode
        fwd = (kc.w in self.keys_held) - (kc.s in self.keys_held)
        rgt = (kc.d in self.keys_held) - (kc.a in self.keys_held)
        upd = (kc.e in self.keys_held) - (kc.q in self.keys_held)
        if not (fwd or rgt or upd):
            return
        # Fixed scale from scene size (set in camera.frame), independent of distance.
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
            self.surface.configure(width=width, height=height, vsync=self.vsync)
        else:
            self.surface.unconfigure()

    def toggle_vsync(self):
        # Only flip state here; reconfiguring the surface mid-frame (from this UI
        # callback) crashes, so apply it at the top of the loop instead.
        self.vsync = not self.vsync
        self.vsync_item.checked = self.vsync

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

        # Settings: a docked panel (placed between Info and Scene by apply_layout).
        settings = spy.ui.Window(screen, "Settings", size=spy.float2(300, 200))
        self.settings = settings

        LABEL_W = 100.0

        def row(label, make):
            # Left-aligned label at a fixed column, then a control that fills to
            # the right edge (negative width = extend to right minus the margin).
            spy.ui.Text(settings, label)
            spy.ui.SameLine(settings).offset_x = LABEL_W
            w = make("##" + label)
            w.width = -8.0
            return w

        def checkbox(label, attr):
            row(label, lambda n: spy.ui.CheckBox(
                settings, n, value=getattr(self, attr),
                callback=lambda v, a=attr: setattr(self, a, bool(v))))

        cur_idx = next(i for i, (_, m) in enumerate(SHADING_ITEMS) if m == self.shading_mode)
        self.shading_combo = row("Shading", lambda n: spy.ui.ComboBox(
            settings, n, value=cur_idx, items=[label for label, _ in SHADING_ITEMS],
            callback=self.on_shading_select))
        checkbox("Wireframe", "wireframe")
        checkbox("Alpha Cutout", "alpha_cutout")
        checkbox("Environment", "show_env")
        row("Exposure", lambda n: spy.ui.SliderFloat(
            settings, n, value=self.exposure, min=0.05, max=8.0,
            callback=lambda v: setattr(self, "exposure", float(v))))
        row("Move Speed", lambda n: spy.ui.SliderFloat(
            settings, n, value=self.move_speed, min=0.1, max=10.0,
            callback=lambda v: setattr(self, "move_speed", float(v))))
        checkbox("Meshes", "show_meshes")
        checkbox("Cameras", "show_cameras")
        checkbox("Lights", "show_lights")

        if self.scene.cameras:
            spy.ui.Separator(settings)
            spy.ui.Text(settings, "Jump to Camera")
            for cam in self.scene.cameras:
                spy.ui.Button(settings, cam.name,
                              callback=lambda c=cam: self.camera.set_from_gltf(c))

        self.menu_window = spy.ui.Window(screen, "menubar", overlay=True)
        menu_bar = spy.ui.MenuBar(self.menu_window)
        menu = spy.ui.Menu(menu_bar, "Menu")
        self.vsync_item = spy.ui.MenuItem(menu, "VSync", callback=self.toggle_vsync)
        self.vsync_item.checked = self.vsync
        spy.ui.MenuItem(menu, "Screenshot",
                        callback=lambda: setattr(self, "want_screenshot", True))

        # Frame-rate readout, in its own UI context so its bright-green text
        # colour is isolated (the ImGui style is global per context).
        self.hud_ui = spy.ui.Context(self.device)
        self.hud_ui.ini_filename = None
        self.hud_ui.add_font("ui", str(FONT_DIR / "FiraSansCondensed-Regular.ttf"),
                             16, is_default=True)
        self.hud_ui.style.set_color(spy.ui.Col.text, spy.float4(0.2, 1.0, 0.2, 1.0))
        self.overlay = spy.ui.Window(self.hud_ui.screen, "overlay", overlay=True,
                                     size=spy.float2(90.0, 52.0))
        self.fps_text = spy.ui.Text(self.overlay, "")
        self.fps_ema = 0.0

        self.info_window = spy.ui.Window(screen, "Info", size=spy.float2(300, 240))
        self.build_info(self.info_window)

        self.scene_window = spy.ui.Window(screen, "Scene", size=spy.float2(300, 360))
        self.build_tree(self.scene_window, self.scene.nodes)

    def on_shading_select(self, v):
        self.shading_mode = SHADING_ITEMS[int(v)][1]

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
        # Left column stacked vertically: Info, Settings, Scene.
        info_node, rest = self.dock.split_node(left, vertical=True, ratio=0.28)
        settings_node, scene_node = self.dock.split_node(rest, vertical=True, ratio=0.42)
        self.info_window.dock_id = info_node
        self.settings.dock_id = settings_node
        self.scene_window.dock_id = scene_node
        self.needs_layout = False

    def build_tree(self, parent, nodes):
        for n in nodes:
            node = spy.ui.TreeNode(parent, n.name)
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
        applied_vsync = self.vsync
        timer = spy.Timer()
        while not self.window.should_close() and not self.terminate:
            self.window.process_events()
            dt = timer.elapsed_s()
            timer.reset()
            self.apply_movement(dt)

            if not self.surface.config or self.window.width == 0 or self.window.height == 0:
                continue
            # Apply a pending vsync change here, between frames, where no surface
            # texture is in flight (reconfiguring mid-frame crashes).
            if self.vsync != applied_vsync:
                self.device.wait()
                self.surface.configure(width=self.window.width, height=self.window.height,
                                       vsync=self.vsync)
                applied_vsync = self.vsync
                continue
            surface_texture = self.surface.acquire_next_image()
            if not surface_texture:
                continue

            self.ensure_targets(surface_texture.width, surface_texture.height)
            encoder = self.device.create_command_encoder()
            self.render(encoder)
            encoder.blit(surface_texture, self.color_texture)

            fps = 1.0 / dt if dt > 1e-6 else 0.0
            self.fps_ema = fps if self.fps_ema == 0.0 else self.fps_ema * 0.9 + fps * 0.1
            # Both figures are derived from the same smoothed rate so they agree.
            ms = 1000.0 / self.fps_ema if self.fps_ema > 1e-6 else 0.0
            self.fps_text.text = f"{self.fps_ema:.0f} FPS\n{ms:.1f} ms"

            self.ui.begin_frame(surface_texture.width, surface_texture.height)
            self.apply_layout()
            self.ui.end_frame(surface_texture, encoder)

            # Separate green-text HUD context for the frame-rate readout (top-right).
            self.hud_ui.begin_frame(surface_texture.width, surface_texture.height)
            self.overlay.size = spy.float2(74.0, 52.0)
            self.overlay.position = spy.float2(surface_texture.width - 78.0, 24.0)
            self.hud_ui.end_frame(surface_texture, encoder)

            self.device.submit_command_buffer(encoder.finish())
            frame += 1
            if max_frames and frame >= max_frames and screenshot:
                self.save_screenshot(surface_texture, screenshot)
            del surface_texture
            self.surface.present()
            if self.want_screenshot:
                self.want_screenshot = False
                self.save_viewport_screenshot()
            if max_frames and frame >= max_frames:
                self.terminate = True

    def render_array(self, width=1600, height=900):
        # Headless: render the current view to an RGB numpy array (no window).
        self.ensure_targets(width, height)
        encoder = self.device.create_command_encoder()
        self.render(encoder)
        self.device.submit_command_buffer(encoder.finish())
        self.device.wait()
        bmp = self.color_texture.to_bitmap().convert(
            spy.Bitmap.PixelFormat.rgb, spy.Bitmap.ComponentType.uint8, srgb_gamma=False
        )
        return np.ascontiguousarray(np.array(bmp))

    def save_viewport_screenshot(self):
        # Save the rendered viewport (color target, without the UI overlay).
        path = f"{self.path.stem}_{self.screenshot_index:04d}.png"
        self.screenshot_index += 1
        self.save_screenshot(self.color_texture, path)
        log.info("saved screenshot %s", path)

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
