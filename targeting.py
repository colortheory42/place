"""
Targeting system.
Handles raycasting from camera to find walls/pillars being looked at.
"""

import math
import numpy as np
from config import (
    PILLAR_SPACING, PILLAR_SIZE, WALL_THICKNESS,
    get_scaled_wall_height, get_scaled_floor_y
)
from raycasting import ray_intersects_triangle


def find_targeted_wall_or_pillar(camera, world):
    """Find the wall or pillar being looked at from camera center."""
    ray_origin, ray_dir = camera.get_ray_direction()
    max_distance = 100

    closest_hit = None
    closest_dist = float('inf')
    hit_type = None

    render_range = 200
    start_x = int((camera.x_s - render_range) // PILLAR_SPACING) * PILLAR_SPACING
    end_x = int((camera.x_s + render_range) // PILLAR_SPACING) * PILLAR_SPACING
    start_z = int((camera.z_s - render_range) // PILLAR_SPACING) * PILLAR_SPACING
    end_z = int((camera.z_s + render_range) // PILLAR_SPACING) * PILLAR_SPACING

    h = get_scaled_wall_height()
    floor_y = get_scaled_floor_y()

    # Check walls
    for px in range(start_x, end_x + PILLAR_SPACING, PILLAR_SPACING):
        for pz in range(start_z, end_z + PILLAR_SPACING, PILLAR_SPACING):
            # Horizontal walls
            if world.has_wall_between(px, pz, px + PILLAR_SPACING, pz):
                wall_key = tuple(sorted([(px, pz), (px + PILLAR_SPACING, pz)]))
                if not world.is_wall_destroyed(wall_key):
                    half_thick = WALL_THICKNESS / 2
                    z = pz
                    x1, x2 = px, px + PILLAR_SPACING

                    v0 = (x1, h, z - half_thick)
                    v1 = (x2, h, z - half_thick)
                    v2 = (x2, floor_y, z - half_thick)
                    v3 = (x1, floor_y, z - half_thick)

                    for tri in [(v0, v1, v2), (v0, v2, v3)]:
                        hit = ray_intersects_triangle(ray_origin, ray_dir, *tri)
                        if hit and hit[0] < max_distance and hit[0] < closest_dist:
                            closest_dist = hit[0]
                            closest_hit = wall_key
                            hit_type = 'wall'

            # Vertical walls
            if world.has_wall_between(px, pz, px, pz + PILLAR_SPACING):
                wall_key = tuple(sorted([(px, pz), (px, pz + PILLAR_SPACING)]))
                if not world.is_wall_destroyed(wall_key):
                    half_thick = WALL_THICKNESS / 2
                    x = px
                    z1, z2 = pz, pz + PILLAR_SPACING

                    v0 = (x - half_thick, h, z1)
                    v1 = (x - half_thick, h, z2)
                    v2 = (x - half_thick, floor_y, z2)
                    v3 = (x - half_thick, floor_y, z1)

                    for tri in [(v0, v1, v2), (v0, v2, v3)]:
                        hit = ray_intersects_triangle(ray_origin, ray_dir, *tri)
                        if hit and hit[0] < max_distance and hit[0] < closest_dist:
                            closest_dist = hit[0]
                            closest_hit = wall_key
                            hit_type = 'wall'

    # Check pillars
    offset = PILLAR_SPACING // 2
    for px in range(start_x, end_x + PILLAR_SPACING, PILLAR_SPACING):
        for pz in range(start_z, end_z + PILLAR_SPACING, PILLAR_SPACING):
            pillar_x = px + offset
            pillar_z = pz + offset

            if world.has_pillar_at(pillar_x, pillar_z):
                pillar_key = (pillar_x, pillar_z)
                if not world.is_pillar_destroyed(pillar_key):
                    s = PILLAR_SIZE

                    # Check all 4 faces
                    faces = [
                        [(pillar_x, h, pillar_z), (pillar_x + s, h, pillar_z),
                         (pillar_x + s, floor_y, pillar_z), (pillar_x, floor_y, pillar_z)],
                        [(pillar_x + s, h, pillar_z + s), (pillar_x, h, pillar_z + s),
                         (pillar_x, floor_y, pillar_z + s), (pillar_x + s, floor_y, pillar_z + s)],
                        [(pillar_x, h, pillar_z), (pillar_x, h, pillar_z + s),
                         (pillar_x, floor_y, pillar_z + s), (pillar_x, floor_y, pillar_z)],
                        [(pillar_x + s, h, pillar_z + s), (pillar_x + s, h, pillar_z),
                         (pillar_x + s, floor_y, pillar_z), (pillar_x + s, floor_y, pillar_z + s)]
                    ]

                    for face in faces:
                        v0, v1, v2, v3 = face
                        for tri in [(v0, v1, v2), (v0, v2, v3)]:
                            hit = ray_intersects_triangle(ray_origin, ray_dir, *tri)
                            if hit and hit[0] < max_distance and hit[0] < closest_dist:
                                closest_dist = hit[0]
                                closest_hit = pillar_key
                                hit_type = 'pillar'

    if closest_hit:
        return (hit_type, closest_hit)
    return None
