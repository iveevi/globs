import logging
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image
from pygltflib import GLTF2

log = logging.getLogger("glb.loader")

COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


@dataclass
class Material:
    base_color_factor: tuple
    base_color_image: Optional[np.ndarray]
    metallic_factor: float = 1.0
    roughness_factor: float = 1.0
    occlusion_strength: float = 1.0
    metallic_roughness_image: Optional[np.ndarray] = None
    occlusion_image: Optional[np.ndarray] = None
    normal_image: Optional[np.ndarray] = None
    emissive_factor: tuple = (0.0, 0.0, 0.0)
    emissive_strength: float = 1.0
    emissive_image: Optional[np.ndarray] = None


@dataclass
class Primitive:
    index_offset: int
    index_count: int
    material_index: int


@dataclass
class Mesh:
    vertices: np.ndarray
    indices: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    primitives: list = field(default_factory=list)
    materials: list = field(default_factory=list)

    @property
    def center(self):
        return 0.5 * (self.bounds_min + self.bounds_max)

    @property
    def radius(self):
        return float(0.5 * np.linalg.norm(self.bounds_max - self.bounds_min))

    @property
    def vertex_count(self):
        return int(self.vertices.shape[0])

    @property
    def triangle_count(self):
        return int(self.indices.shape[0] // 3)


@dataclass
class CameraInfo:
    name: str
    world: np.ndarray
    yfov: float
    aspect: Optional[float]
    znear: float
    zfar: Optional[float]


@dataclass
class LightInfo:
    name: str
    type: str
    color: tuple
    intensity: float
    range: Optional[float]
    inner_cone_angle: Optional[float]
    outer_cone_angle: Optional[float]
    world: np.ndarray
    position: tuple
    direction: tuple


@dataclass
class SceneNode:
    name: str
    mesh_name: Optional[str]
    triangle_count: int
    has_uvs: bool
    light: Optional[LightInfo]
    camera: Optional[CameraInfo]
    children: list


@dataclass
class Scene:
    mesh: Mesh
    nodes: list
    cameras: list
    lights: list


def buffer_bytes(gltf, buffer_index):
    buffer = gltf.buffers[buffer_index]
    uri = buffer.uri
    if uri is None:
        blob = gltf.binary_blob()
        if blob is None:
            raise ValueError("GLB has no binary blob")
        return blob
    if uri.startswith("data:"):
        return gltf.get_data_from_buffer_uri(uri)
    raise NotImplementedError(f"external buffer uri not supported: {uri!r}")


def read_accessor(gltf, accessor_index):
    accessor = gltf.accessors[accessor_index]
    ncomp = TYPE_COMPONENTS[accessor.type]
    dtype = np.dtype(COMPONENT_DTYPE[accessor.componentType])
    count = accessor.count
    if accessor.bufferView is None:
        return np.zeros((count, ncomp), dtype=np.float32)

    view = gltf.bufferViews[accessor.bufferView]
    raw = buffer_bytes(gltf, view.buffer)
    base = (view.byteOffset or 0) + (accessor.byteOffset or 0)
    comp = dtype.itemsize
    stride = view.byteStride or (comp * ncomp)
    arr = np.ndarray(
        (count, ncomp), dtype=dtype, buffer=raw, offset=base, strides=(stride, comp)
    )
    data = arr.astype(np.float32)
    if accessor.normalized:
        if dtype == np.uint8:
            data /= 255.0
        elif dtype == np.uint16:
            data /= 65535.0
        elif dtype == np.int8:
            data = np.maximum(data / 127.0, -1.0)
        elif dtype == np.int16:
            data = np.maximum(data / 32767.0, -1.0)
    return data


def node_local_matrix(node):
    if node.matrix is not None:
        return np.array(node.matrix, dtype=np.float64).reshape(4, 4).T
    m = np.eye(4, dtype=np.float64)
    if node.scale is not None:
        s = np.array(node.scale, dtype=np.float64)
        m = np.diag([s[0], s[1], s[2], 1.0]) @ m
    if node.rotation is not None:
        x, y, z, w = node.rotation
        r = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        m = r @ m
    if node.translation is not None:
        t = np.array(node.translation, dtype=np.float64)
        tm = np.eye(4, dtype=np.float64)
        tm[:3, 3] = t
        m = tm @ m
    return m


def scene_roots(gltf):
    if gltf.scene is not None and gltf.scenes:
        return list(gltf.scenes[gltf.scene].nodes)
    if gltf.scenes:
        return list(gltf.scenes[0].nodes)
    return list(range(len(gltf.nodes)))


def iter_nodes(gltf):
    stack = [(n, np.eye(4, dtype=np.float64)) for n in reversed(scene_roots(gltf))]
    while stack:
        node_index, parent = stack.pop()
        node = gltf.nodes[node_index]
        world = parent @ node_local_matrix(node)
        yield node_index, node, world
        for child in reversed(node.children or []):
            stack.append((child, world))


def face_normals(positions, indices):
    normals = np.zeros_like(positions)
    tris = indices.reshape(-1, 3)
    a = positions[tris[:, 0]]
    b = positions[tris[:, 1]]
    c = positions[tris[:, 2]]
    fn = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(normals, tris[:, k], fn)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return normals / lengths


def primitive_arrays(gltf, prim, world, normal_mat, flip_winding):
    if prim.mode not in (None, 4):
        return None
    attrs = prim.attributes
    if attrs.POSITION is None:
        return None

    pos = read_accessor(gltf, attrs.POSITION)
    n_verts = pos.shape[0]
    nrm = (
        read_accessor(gltf, attrs.NORMAL)
        if attrs.NORMAL is not None
        else np.zeros((n_verts, 3), dtype=np.float32)
    )
    uv = (
        read_accessor(gltf, attrs.TEXCOORD_0)[:, :2]
        if attrs.TEXCOORD_0 is not None
        else np.zeros((n_verts, 2), dtype=np.float32)
    )
    idx = (
        read_accessor(gltf, prim.indices).astype(np.uint32).ravel()
        if prim.indices is not None
        else np.arange(n_verts, dtype=np.uint32)
    )
    if flip_winding:
        idx = np.ascontiguousarray(idx.reshape(-1, 3)[:, ::-1].ravel())

    world_pos = (pos @ world[:3, :3].T) + world[:3, 3]
    world_nrm = (
        face_normals(world_pos, idx) if attrs.NORMAL is None else nrm @ normal_mat.T
    )
    verts = np.concatenate(
        [
            world_pos.astype(np.float32),
            world_nrm.astype(np.float32),
            uv.astype(np.float32),
        ],
        axis=1,
    )
    return verts, idx


def build_mesh(gltf, path):
    log.info("geometry: decoding primitives")
    t0 = time.perf_counter()
    v_chunks = []
    i_chunks = []
    primitives = []
    vert_base = 0
    idx_base = 0
    for _, node, world in iter_nodes(gltf):
        if node.mesh is None:
            continue
        normal_mat = np.linalg.inv(world[:3, :3]).T
        flip_winding = np.linalg.det(world[:3, :3]) < 0.0
        for prim in gltf.meshes[node.mesh].primitives:
            result = primitive_arrays(gltf, prim, world, normal_mat, flip_winding)
            if result is None:
                continue
            verts, idx = result
            v_chunks.append(verts)
            i_chunks.append(idx + vert_base)
            primitives.append(
                Primitive(
                    index_offset=idx_base,
                    index_count=int(idx.size),
                    material_index=(int(prim.material) if prim.material is not None else -1),
                )
            )
            idx_base += int(idx.size)
            vert_base += verts.shape[0]

    if not v_chunks:
        raise ValueError(f"no triangle geometry found in {path}")
    log.info("geometry: %d primitives, %d vertices in %.2fs",
             len(primitives), vert_base, time.perf_counter() - t0)
    vertices = np.ascontiguousarray(np.concatenate(v_chunks, axis=0).astype(np.float32))
    indices = np.ascontiguousarray(np.concatenate(i_chunks, axis=0).astype(np.uint32))
    positions = vertices[:, :3]
    return Mesh(
        vertices=vertices,
        indices=indices,
        bounds_min=positions.min(axis=0),
        bounds_max=positions.max(axis=0),
        primitives=primitives,
    )


def decode_image(gltf, image_index):
    image = gltf.images[image_index]
    if image.bufferView is not None:
        view = gltf.bufferViews[image.bufferView]
        raw = buffer_bytes(gltf, view.buffer)
        start = view.byteOffset or 0
        data = bytes(raw[start : start + view.byteLength])
    elif image.uri and image.uri.startswith("data:"):
        data = gltf.get_data_from_buffer_uri(image.uri)
    else:
        return None
    try:
        img = Image.open(BytesIO(data)).convert("RGBA")
        return np.ascontiguousarray(np.array(img, dtype=np.uint8))
    except Exception:
        return None


def tex_source(gltf, tex_info):
    if tex_info is None:
        return None
    return gltf.textures[tex_info.index].source


def build_materials(gltf):
    mats = gltf.materials or []
    infos = []
    sources = set()
    for m in mats:
        pbr = m.pbrMetallicRoughness
        base_src = tex_source(gltf, pbr.baseColorTexture) if pbr else None
        mr_src = tex_source(gltf, pbr.metallicRoughnessTexture) if pbr else None
        occ_src = tex_source(gltf, m.occlusionTexture)
        nrm_src = tex_source(gltf, m.normalTexture)
        emi_src = tex_source(gltf, m.emissiveTexture)
        for s in (base_src, mr_src, occ_src, nrm_src, emi_src):
            if s is not None:
                sources.add(s)
        bf = (pbr.baseColorFactor if pbr and pbr.baseColorFactor else [1.0, 1.0, 1.0, 1.0])
        mf = 1.0 if not pbr or pbr.metallicFactor is None else pbr.metallicFactor
        rf = 1.0 if not pbr or pbr.roughnessFactor is None else pbr.roughnessFactor
        occ_s = (m.occlusionTexture.strength
                 if m.occlusionTexture and m.occlusionTexture.strength is not None else 1.0)
        ef = m.emissiveFactor if m.emissiveFactor else [0.0, 0.0, 0.0]
        me = m.extensions.get("KHR_materials_emissive_strength") if isinstance(
            m.extensions, dict) else None
        es = float(me.get("emissiveStrength", 1.0)) if isinstance(me, dict) else 1.0
        infos.append((base_src, mr_src, occ_src, nrm_src, emi_src, bf, mf, rf, occ_s, ef, es))

    log.info("textures: decoding %d images", len(sources))
    t0 = time.perf_counter()
    cache = {s: decode_image(gltf, s) for s in sources}
    log.info("textures: decoded %d images in %.2fs", len(sources),
             time.perf_counter() - t0)

    materials = []
    for (base_src, mr_src, occ_src, nrm_src, emi_src,
         bf, mf, rf, occ_s, ef, es) in infos:
        materials.append(Material(
            base_color_factor=(float(bf[0]), float(bf[1]), float(bf[2]), float(bf[3])),
            base_color_image=cache.get(base_src),
            metallic_factor=float(mf),
            roughness_factor=float(rf),
            occlusion_strength=float(occ_s),
            metallic_roughness_image=cache.get(mr_src),
            occlusion_image=cache.get(occ_src),
            normal_image=cache.get(nrm_src),
            emissive_factor=(float(ef[0]), float(ef[1]), float(ef[2])),
            emissive_strength=float(es),
            emissive_image=cache.get(emi_src),
        ))
    return materials


def light_table(gltf):
    doc_ext = gltf.extensions if isinstance(gltf.extensions, dict) else None
    if not doc_ext:
        return []
    ext = doc_ext.get("KHR_lights_punctual")
    if not ext:
        return []
    return ext.get("lights", []) or []


def light_for_node(gltf, node, world, defs):
    node_ext = node.extensions if isinstance(node.extensions, dict) else None
    ne = node_ext.get("KHR_lights_punctual") if node_ext else None
    if not ne:
        return None
    index = ne.get("light")
    if index is None or index >= len(defs):
        return None
    ld = defs[index]
    color = ld.get("color", [1.0, 1.0, 1.0])
    spot = ld.get("spot") or {}
    rot = world[:3, :3]
    direction = rot @ np.array([0.0, 0.0, -1.0])
    direction = direction / (np.linalg.norm(direction) or 1.0)
    position = world[:3, 3]
    return LightInfo(
        name=ld.get("name", f"light{index}"),
        type=ld.get("type", "directional"),
        color=(float(color[0]), float(color[1]), float(color[2])),
        intensity=float(ld.get("intensity", 1.0)),
        range=float(ld["range"]) if "range" in ld else None,
        inner_cone_angle=(
            float(spot["innerConeAngle"]) if "innerConeAngle" in spot else None
        ),
        outer_cone_angle=(
            float(spot["outerConeAngle"]) if "outerConeAngle" in spot else None
        ),
        world=world.copy(),
        position=(float(position[0]), float(position[1]), float(position[2])),
        direction=(float(direction[0]), float(direction[1]), float(direction[2])),
    )


def camera_for_node(gltf, node, world):
    if node.camera is None:
        return None
    cam = gltf.cameras[node.camera]
    persp = cam.perspective
    if persp is None:
        return None
    return CameraInfo(
        name=cam.name or f"camera{node.camera}",
        world=world.copy(),
        yfov=float(persp.yfov),
        aspect=float(persp.aspectRatio) if persp.aspectRatio else None,
        znear=float(persp.znear) if persp.znear else 0.05,
        zfar=float(persp.zfar) if persp.zfar else None,
    )


def build_graph(gltf, defs):
    log.info("scene graph: walking nodes")
    t0 = time.perf_counter()
    cameras = []
    lights = []
    node_cameras = {}
    node_lights = {}
    for ni, node, world in iter_nodes(gltf):
        cam = camera_for_node(gltf, node, world)
        if cam:
            node_cameras[ni] = cam
            cameras.append(cam)
        light = light_for_node(gltf, node, world, defs)
        if light:
            node_lights[ni] = light
            lights.append(light)

    def visit(node_index):
        node = gltf.nodes[node_index]
        mesh_name = None
        tri_count = 0
        has_uvs = False
        if node.mesh is not None:
            m = gltf.meshes[node.mesh]
            mesh_name = m.name or f"mesh{node.mesh}"
            for p in m.primitives:
                if p.attributes.POSITION is None:
                    continue
                if p.indices is not None:
                    tri_count += gltf.accessors[p.indices].count // 3
                else:
                    tri_count += gltf.accessors[p.attributes.POSITION].count // 3
                has_uvs = has_uvs or p.attributes.TEXCOORD_0 is not None
        children = [c for c in (visit(ci) for ci in (node.children or [])) if c]
        if node.mesh is None and node.camera is None and not children:
            node_ext = node.extensions if isinstance(node.extensions, dict) else None
            if not (node_ext and node_ext.get("KHR_lights_punctual")):
                return None
        return SceneNode(
            name=node.name or f"node{node_index}",
            mesh_name=mesh_name,
            triangle_count=tri_count,
            has_uvs=has_uvs,
            light=node_lights.get(node_index),
            camera=node_cameras.get(node_index),
            children=children,
        )

    nodes = [n for n in (visit(r) for r in scene_roots(gltf)) if n]
    log.info("scene graph: %d cameras, %d lights in %.2fs",
             len(cameras), len(lights), time.perf_counter() - t0)
    return nodes, cameras, lights


def load_scene(path):
    log.info("loading %s", path)
    t0 = time.perf_counter()
    gltf = GLTF2().load(str(path))
    defs = light_table(gltf)
    log.info("parsed glTF in %.2fs", time.perf_counter() - t0)

    mesh = build_mesh(gltf, path)
    mesh.materials = build_materials(gltf)
    nodes, cameras, lights = build_graph(gltf, defs)

    log.info("scene ready in %.2fs (%d tris, %d materials)",
             time.perf_counter() - t0, mesh.triangle_count, len(mesh.materials))
    return Scene(mesh=mesh, nodes=nodes, cameras=cameras, lights=lights)
