import math

import numpy as np


def segments(points_a, points_b, color):
    a = np.asarray(points_a, dtype=np.float32).reshape(-1, 3)
    b = np.asarray(points_b, dtype=np.float32).reshape(-1, 3)
    col = np.tile(np.asarray(color, dtype=np.float32), (a.shape[0], 1))
    va = np.concatenate([a, col], axis=1)
    vb = np.concatenate([b, col], axis=1)
    out = np.empty((a.shape[0] * 2, 6), dtype=np.float32)
    out[0::2] = va
    out[1::2] = vb
    return out


def line(a, b, color):
    return segments([a], [b], color)


def circle(center, axis, radius, color, n=32):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) or 1.0)
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u = u / (np.linalg.norm(u) or 1.0)
    v = np.cross(axis, u)
    t = np.linspace(0.0, 2.0 * math.pi, n + 1)
    pts = (
        np.asarray(center)
        + radius * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
    )
    return segments(pts[:-1], pts[1:], color)


def camera_frustum(cam, far_len, color=(0.2, 0.9, 0.95)):
    aspect = cam.aspect or 1.5
    near = max(cam.znear, far_len * 0.01)
    far = far_len
    tn = math.tan(cam.yfov * 0.5)

    def rect(dist):
        h = tn * dist
        w = h * aspect
        return [
            [-w, -h, -dist], [w, -h, -dist], [w, h, -dist], [-w, h, -dist],
        ]

    local = np.array(rect(near) + rect(far) + [[0.0, 0.0, 0.0]], dtype=np.float64)
    world = (local @ cam.world[:3, :3].T) + cam.world[:3, 3]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
        (8, 4), (8, 5), (8, 6), (8, 7),
    ]
    a = [world[i] for i, _ in edges]
    b = [world[j] for _, j in edges]
    return segments(a, b, color)


def directional_gizmo(pos, direction, length, color):
    pos = np.asarray(pos, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    d = d / (np.linalg.norm(d) or 1.0)
    tip = pos + d * length
    ref = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, ref)
    u = u / (np.linalg.norm(u) or 1.0)
    v = np.cross(d, u)
    h = length * 0.18
    back = tip - d * h
    parts = [line(pos, tip, color), circle(pos, d, length * 0.12, color, 48)]
    for vec in (u, v, -u, -v):
        parts.append(line(tip, back + vec * h, color))
    return np.concatenate(parts)


def point_gizmo(pos, radius, color):
    return np.concatenate(
        [
            circle(pos, [1, 0, 0], radius, color, 64),
            circle(pos, [0, 1, 0], radius, color, 64),
            circle(pos, [0, 0, 1], radius, color, 64),
        ]
    )


def spot_gizmo(pos, direction, outer_angle, length, color):
    pos = np.asarray(pos, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    d = d / (np.linalg.norm(d) or 1.0)
    radius = math.tan(min(outer_angle, math.radians(80.0))) * length
    radius = min(radius, length * 2.0)
    base = pos + d * length
    ref = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, ref)
    u = u / (np.linalg.norm(u) or 1.0)
    v = np.cross(d, u)
    rim = [base + radius * (math.cos(a) * u + math.sin(a) * v) for a in
           np.linspace(0, 2 * math.pi, 5)[:-1]]
    parts = [circle(base, d, radius, color, 96)]
    for r in rim:
        parts.append(line(pos, r, color))
    return np.concatenate(parts)


def build_lights(lights, scene_radius):
    if not lights:
        return None
    length = scene_radius * 0.25
    parts = []
    for li in lights:
        if li.type == "directional":
            parts.append(directional_gizmo(li.position, li.direction, length, li.color))
        elif li.type == "spot":
            outer = li.outer_cone_angle or math.radians(30.0)
            reach = scene_radius * 0.12
            parts.append(spot_gizmo(li.position, li.direction, outer, reach, li.color))
        else:
            parts.append(point_gizmo(li.position, scene_radius * 0.03, li.color))
    return np.concatenate(parts) if parts else None


def build_cameras(cameras, scene_radius):
    if not cameras:
        return None
    far = scene_radius * 0.35
    return np.concatenate([camera_frustum(c, far) for c in cameras])
