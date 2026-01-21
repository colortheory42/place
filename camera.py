"""
Camera system.
Handles camera pose, smoothing, world-to-screen transforms, and projection.
"""

import math
from config import (
    NEAR, CAMERA_SMOOTHING, ROTATION_SMOOTHING,
    CAMERA_HEIGHT_STAND, HEAD_BOB_SPEED, CAMERA_SHAKE_AMOUNT,
    CEILING_HEIGHT_MULTIPLIER,
    get_scaled_head_bob_amount, get_scaled_head_bob_sway
)


class Camera:
    """Camera with smoothing and head bob effects."""

    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Smoothed camera values (what we actually render from)
        self.x_s = 200
        self.y_s = CAMERA_HEIGHT_STAND
        self.z_s = 200
        self.pitch_s = 0
        self.yaw_s = 0

        # Animation timers
        self.head_bob_time = 0
        self.camera_shake_time = 0

    def update(self, dt, player):
        """Update camera to follow player with smoothing and effects."""
        # Head bob (only when moving)
        if player.is_moving:
            self.head_bob_time += dt * HEAD_BOB_SPEED

        bob_y = 0
        bob_x = 0
        if player.is_moving:
            bob_y = math.sin(self.head_bob_time * 2 * math.pi) * get_scaled_head_bob_amount()
            bob_x = math.sin(self.head_bob_time * math.pi) * get_scaled_head_bob_sway()

        # Camera shake
        self.camera_shake_time += dt
        shake_x = math.sin(self.camera_shake_time * 13.7) * CAMERA_SHAKE_AMOUNT
        shake_y = math.cos(self.camera_shake_time * 11.3) * CAMERA_SHAKE_AMOUNT * CEILING_HEIGHT_MULTIPLIER

        # Effective position with effects
        effective_y = player.y + bob_y + shake_y
        effective_x = player.x + bob_x + shake_x

        # Position smoothing
        movement_smooth = CAMERA_SMOOTHING if player.is_moving else 1.0
        self.x_s += (effective_x - self.x_s) * movement_smooth
        self.y_s += (effective_y - self.y_s) * movement_smooth
        self.z_s += (player.z - self.z_s) * movement_smooth

        # Rotation smoothing
        rotation_smooth = ROTATION_SMOOTHING if player.is_rotating else 1.0
        self.pitch_s += (player.pitch - self.pitch_s) * rotation_smooth
        self.yaw_s += (player.yaw - self.yaw_s) * rotation_smooth

    def world_to_camera(self, x, y, z):
        """Transform world coordinates to camera space."""
        x -= self.x_s
        y -= self.y_s
        z -= self.z_s

        # Yaw rotation
        cy = math.cos(self.yaw_s)
        sy = math.sin(self.yaw_s)
        x1 = x * cy - z * sy
        z1 = x * sy + z * cy

        # Pitch rotation
        cp = math.cos(self.pitch_s)
        sp = math.sin(self.pitch_s)
        y2 = y * cp - z1 * sp
        z2 = y * sp + z1 * cp

        return (x1, y2, z2)

    def project(self, cam_point):
        """Project camera-space point to screen coordinates."""
        x, y, z = cam_point
        if z <= NEAR:
            return None

        aspect = self.height / self.width
        FOV_ANGLE = 90  # degrees
        focal_length = (self.width * 0.5) / math.tan(math.radians(FOV_ANGLE * 0.5))
        scale = focal_length / z

        sx = self.width * 0.5 + x * scale
        sy = self.height * 0.5 - y * scale * aspect

        if not (math.isfinite(sx) and math.isfinite(sy)):
            return None

        return (sx, sy)

    def clip_poly_near(self, poly):
        """Clip polygon against near plane."""
        if not poly or len(poly) < 3:
            return []

        def inside(p):
            return p[2] >= NEAR

        def intersect(a, b):
            ax, ay, az = a
            bx, by, bz = b

            dz = bz - az
            if abs(dz) < 0.00001:
                return None

            t = (NEAR - az) / dz

            if t < -0.001 or t > 1.001:
                return None

            t = max(0.0, min(1.0, t))

            return (ax + (bx - ax) * t, ay + (by - ay) * t, NEAR + 0.001)

        out = []
        prev = poly[-1]
        prev_in = inside(prev)

        for cur in poly:
            cur_in = inside(cur)

            if cur_in and prev_in:
                out.append(cur)
            elif cur_in and not prev_in:
                intersection = intersect(prev, cur)
                if intersection:
                    out.append(intersection)
                out.append(cur)
            elif (not cur_in) and prev_in:
                intersection = intersect(prev, cur)
                if intersection:
                    out.append(intersection)

            prev, prev_in = cur, cur_in

        if len(out) < 3:
            return []

        if any(not math.isfinite(p[2]) or p[2] < NEAR for p in out):
            return []

        return out

    def get_ray_direction(self):
        """Get ray direction from screen center (for targeting)."""
        import numpy as np
        from config import FOV

        mx, my = self.width // 2, self.height // 2

        ndc_x = (mx / self.width - 0.5) * 2
        ndc_y = (my / self.height - 0.5) * 2

        ray_dir_cam = np.array([
            ndc_x * self.width / FOV,
            ndc_y * self.height / FOV,
            1.0
        ])
        ray_dir_cam = ray_dir_cam / np.linalg.norm(ray_dir_cam)

        # Transform to world space (inverse camera rotation)
        cp = math.cos(-self.pitch_s)
        sp = math.sin(-self.pitch_s)
        x1 = ray_dir_cam[0]
        y1 = ray_dir_cam[1] * cp - ray_dir_cam[2] * sp
        z1 = ray_dir_cam[1] * sp + ray_dir_cam[2] * cp

        cy = math.cos(-self.yaw_s)
        sy = math.sin(-self.yaw_s)
        x2 = x1 * cy - z1 * sy
        z2 = x1 * sy + z1 * cy

        ray_dir = np.array([x2, y1, z2])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        ray_origin = np.array([self.x_s, self.y_s, self.z_s])

        return ray_origin, ray_dir
