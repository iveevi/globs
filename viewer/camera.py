import math

import numpy as np


class FlyCamera:
    PITCH_LIMIT = math.pi * 0.5 - 0.02

    def __init__(self):
        self.position = np.zeros(3, dtype=np.float64)
        self.yaw = 0.0
        self.pitch = 0.3
        self.fov_y = math.radians(50.0)
        self.move_scale = 1.0
        self.cam_aspect = None

    def forward(self):
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        sy = math.sin(self.yaw)
        cy = math.cos(self.yaw)
        return np.array([-cp * sy, -sp, -cp * cy], dtype=np.float64)

    def basis(self):
        f = self.forward()
        right = np.array([-f[2], 0.0, f[0]], dtype=np.float64)
        n = np.linalg.norm(right)
        right = right / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        up = np.cross(right, f)
        return right, up

    def target(self):
        return self.position + self.forward()

    def rotate(self, dx, dy, speed=0.005):
        self.yaw -= dx * speed
        self.pitch += dy * speed
        self.pitch = max(-self.PITCH_LIMIT, min(self.PITCH_LIMIT, self.pitch))

    def pan(self, dx, dy):
        right, up = self.basis()
        speed = self.move_scale * 0.0015
        self.position += -right * dx * speed + up * dy * speed

    def move(self, forward, right_amt, up_amt):
        f = self.forward()
        right, _ = self.basis()
        self.position += f * forward + right * right_amt
        self.position[1] += up_amt

    def zoom(self, wheel):
        self.position += self.forward() * (self.move_scale * 0.1 * wheel)

    def frame(self, center, radius):
        r = max(float(radius), 1e-3)
        self.move_scale = r
        self.yaw = 0.0
        self.pitch = 0.3
        self.cam_aspect = None
        d = r / max(math.sin(self.fov_y * 0.5), 1e-3) * 1.1
        self.position = np.array(center, dtype=np.float64) - self.forward() * d

    def set_from_gltf(self, cam):
        world = cam.world
        self.position = world[:3, 3].astype(np.float64).copy()
        fwd = world[:3, :3] @ np.array([0.0, 0.0, -1.0])
        fwd = fwd / (np.linalg.norm(fwd) or 1.0)
        self.pitch = max(
            -self.PITCH_LIMIT, min(self.PITCH_LIMIT, math.asin(-fwd[1]))
        )
        self.yaw = math.atan2(-fwd[0], -fwd[2])
        self.fov_y = cam.yfov
        self.cam_aspect = cam.aspect

    def effective_fov_y(self, window_aspect):
        # Preserve the glTF camera's intended framing: if the window is
        # narrower than the authored aspect, widen the vertical fov so the
        # full intended horizontal extent stays visible (letterbox-style fit).
        if self.cam_aspect and window_aspect < self.cam_aspect:
            half_h = math.tan(self.fov_y * 0.5) * self.cam_aspect / window_aspect
            return 2.0 * math.atan(half_h)
        return self.fov_y
