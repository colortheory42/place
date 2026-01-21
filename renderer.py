"""
Renderer system.
Handles 3D polygon rendering, debris, and visual effects.
"""

import math
import random
import pygame
from config import (
    RENDER_SCALE, NEAR, RENDER_DISTANCE, BLACK,
    PILLAR_SPACING, PILLAR_SIZE, WALL_THICKNESS, HALLWAY_WIDTH,
    FOG_ENABLED, FOG_START, FOG_END, FOG_COLOR,
    FLICKER_CHANCE, FLICKER_DURATION, FLICKER_BRIGHTNESS,
    get_scaled_wall_height, get_scaled_floor_y
)
from textures import (
    generate_carpet_texture, generate_ceiling_tile_texture,
    generate_wall_texture, generate_pillar_texture
)
from world import WallState
from ceiling_heights import get_ceiling_height_at_position, get_room_size_at_position


class Renderer:
    """Handles all rendering operations."""

    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Render scaling
        self.render_scale = RENDER_SCALE
        self.target_render_scale = RENDER_SCALE
        self.render_scale_transition_speed = 2.0
        self.render_surface = None
        self._update_render_surface()

        # Flickering
        self.flicker_timer = 0
        self.is_flickering = False
        self.flicker_brightness = 1.0

        # Generate textures and get average colors
        print("Generating procedural textures...")
        self.carpet_texture = generate_carpet_texture()
        self.ceiling_texture = generate_ceiling_tile_texture()
        self.wall_texture = generate_wall_texture()
        self.pillar_texture = generate_pillar_texture()

        self.carpet_avg = self._get_average_color(self.carpet_texture)
        self.ceiling_avg = self._get_average_color(self.ceiling_texture)
        self.wall_avg = self._get_average_color(self.wall_texture)
        self.pillar_avg = self._get_average_color(self.pillar_texture)
        print("Textures generated!")

    def _get_average_color(self, surface):
        """Extract average color from a surface."""
        arr = pygame.surfarray.array3d(surface)
        return tuple(int(arr[:, :, i].mean()) for i in range(3))

    # === RENDER SCALING ===

    def _update_render_surface(self):
        render_width = int(self.width * self.render_scale)
        render_height = int(self.height * self.render_scale)
        self.render_surface = pygame.Surface((render_width, render_height))

    def toggle_render_scale(self):
        """Toggle between full and half render scale."""
        if self.target_render_scale == 1.0:
            self.target_render_scale = 0.5
            print("Render scale transitioning to: 0.5x")
        else:
            self.target_render_scale = 1.0
            print("Render scale transitioning to: 1.0x")

    def update_render_scale(self, dt):
        """Smoothly transition render scale."""
        if abs(self.render_scale - self.target_render_scale) > 0.01:
            if self.render_scale < self.target_render_scale:
                self.render_scale = min(self.target_render_scale,
                                        self.render_scale + self.render_scale_transition_speed * dt)
            else:
                self.render_scale = max(self.target_render_scale,
                                        self.render_scale - self.render_scale_transition_speed * dt)
            self._update_render_surface()
        else:
            if self.render_scale != self.target_render_scale:
                self.render_scale = self.target_render_scale
                self._update_render_surface()

    # === FLICKERING ===

    def update_flicker(self, dt):
        """Update light flicker effect."""
        if self.is_flickering:
            self.flicker_timer += dt
            if self.flicker_timer >= FLICKER_DURATION:
                self.is_flickering = False
                self.flicker_brightness = 1.0
        else:
            if random.random() < FLICKER_CHANCE:
                self.is_flickering = True
                self.flicker_timer = 0
                self.flicker_brightness = 1.0 - FLICKER_BRIGHTNESS

    # === VISUAL EFFECTS ===

    def apply_fog(self, color, distance):
        """Apply fog effect based on distance."""
        if not FOG_ENABLED:
            return tuple(int(c * self.flicker_brightness) for c in color)

        if distance < FOG_START:
            return tuple(int(c * self.flicker_brightness) for c in color)
        if distance > FOG_END:
            fog_color = tuple(int(c * self.flicker_brightness) for c in FOG_COLOR)
            return fog_color

        fog_amount = (distance - FOG_START) / (FOG_END - FOG_START)
        adjusted_color = tuple(int(c * self.flicker_brightness) for c in color)
        fog_color = tuple(int(c * self.flicker_brightness) for c in FOG_COLOR)

        return tuple(
            int(adjusted_color[i] * (1 - fog_amount) + fog_color[i] * fog_amount)
            for i in range(3)
        )

    def apply_surface_noise(self, color, x, z):
        """Apply subtle surface noise for texture variation."""
        noise = ((int(x) * 13 + int(z) * 17) % 5) - 2
        return tuple(max(0, min(255, c + noise)) for c in color)

    def apply_zone_tint(self, color, world, zone_x, zone_z):
        """Apply zone-specific color tint."""
        props = world.get_zone_properties(zone_x, zone_z)
        tint = props['color_tint']
        return tuple(int(min(255, c * tint[i])) for i, c in enumerate(color))

    # === POLYGON RENDERING ===

    def draw_world_poly(self, surface, camera, world, world_pts, color,
                        width_edges=0, edge_color=None,
                        is_wall=False, is_floor=False, is_ceiling=False):
        """Draw a 3D polygon with all effects applied."""
        cam_pts = [camera.world_to_camera(*p) for p in world_pts]

        behind_count = sum(1 for p in cam_pts if p[2] < NEAR)
        if behind_count == len(cam_pts):
            return

        distances = [math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) for p in cam_pts]
        avg_dist = sum(distances) / len(distances) if distances else 0

        if avg_dist > RENDER_DISTANCE * 1.5:
            return

        # Get world position for effects
        avg_x = sum(p[0] for p in world_pts) / len(world_pts)
        avg_z = sum(p[2] for p in world_pts) / len(world_pts)
        avg_y = sum(p[1] for p in world_pts) / len(world_pts)

        zone = world.get_zone_at(avg_x, avg_z)
        tinted_color = self.apply_zone_tint(color, world, *zone)
        noisy_color = self.apply_surface_noise(tinted_color, avg_x, avg_z)

        # Ambient occlusion for walls
        ao_factor = 1.0
        if is_wall:
            if avg_y < get_scaled_floor_y() + 20:
                ao_factor = 0.7
            elif avg_y > get_scaled_wall_height() - 20:
                ao_factor = 0.8

        ao_color = tuple(int(c * ao_factor) for c in noisy_color)
        fogged_color = self.apply_fog(ao_color, avg_dist)

        # Clip and project
        cam_pts = camera.clip_poly_near(cam_pts)
        if len(cam_pts) < 3:
            return

        screen_pts = [camera.project(p) for p in cam_pts]
        if any(p is None for p in screen_pts):
            return

        # Frustum culling
        min_x = min(p[0] for p in screen_pts)
        max_x = max(p[0] for p in screen_pts)
        min_y = min(p[1] for p in screen_pts)
        max_y = max(p[1] for p in screen_pts)

        margin = 500
        if (max_x < -margin or min_x > camera.width + margin or
                max_y < -margin or min_y > camera.height + margin):
            return

        # Skip tiny polygons
        if (max_x - min_x) < 0.5 and (max_y - min_y) < 0.5:
            return

        try:
            pygame.draw.polygon(surface, fogged_color, screen_pts)
        except:
            return

        # Draw edges
        if width_edges > 0 and edge_color is not None:
            tinted_edge = self.apply_zone_tint(edge_color, world, *zone)
            noisy_edge = self.apply_surface_noise(tinted_edge, avg_x, avg_z)
            fogged_edge = self.apply_fog(noisy_edge, avg_dist)
            try:
                for i in range(len(screen_pts)):
                    pygame.draw.line(surface, fogged_edge, screen_pts[i],
                                     screen_pts[(i + 1) % len(screen_pts)], width_edges)
            except:
                pass

    # === GEOMETRY GENERATION ===

    def _get_floor_tiles(self, camera, world):
        """Generate floor tile render queue."""
        render_items = []
        render_range = RENDER_DISTANCE
        tile_size = PILLAR_SPACING

        start_x = int((camera.x_s - render_range) // tile_size) * tile_size
        end_x = int((camera.x_s + render_range) // tile_size) * tile_size
        start_z = int((camera.z_s - render_range) // tile_size) * tile_size
        end_z = int((camera.z_s + render_range) // tile_size) * tile_size

        floor_y = get_scaled_floor_y()

        for px in range(start_x, end_x, tile_size):
            for pz in range(start_z, end_z, tile_size):
                tile_center_x = px + tile_size / 2
                tile_center_z = pz + tile_size / 2

                dist = math.sqrt((tile_center_x - camera.x_s) ** 2 +
                                 (tile_center_z - camera.z_s) ** 2)

                if dist > render_range + tile_size:
                    continue

                def make_draw_func(px=px, pz=pz, floor_y=floor_y, tile_size=tile_size):
                    return lambda surface, camera=camera, world=world: self.draw_world_poly(
                        surface, camera, world,
                        [(px, floor_y, pz), (px + tile_size, floor_y, pz),
                         (px + tile_size, floor_y, pz + tile_size),
                         (px, floor_y, pz + tile_size)],
                        self.carpet_avg,
                        width_edges=0,
                        edge_color=None,
                        is_floor=True
                    )

                render_items.append((dist, make_draw_func()))

        return render_items

    def _get_ceiling_tiles(self, camera, world):
        """Generate ceiling tile render queue with VARIABLE HEIGHTS."""
        render_items = []
        render_range = RENDER_DISTANCE
        tile_size = PILLAR_SPACING

        start_x = int((camera.x_s - render_range) // tile_size) * tile_size
        end_x = int((camera.x_s + render_range) // tile_size) * tile_size
        start_z = int((camera.z_s - render_range) // tile_size) * tile_size
        end_z = int((camera.z_s + render_range) // tile_size) * tile_size

        for px in range(start_x, end_x, tile_size):
            for pz in range(start_z, end_z, tile_size):
                tile_center_x = px + tile_size / 2
                tile_center_z = pz + tile_size / 2

                dist = math.sqrt((tile_center_x - camera.x_s) ** 2 +
                                 (tile_center_z - camera.z_s) ** 2)

                if dist > render_range + tile_size:
                    continue

                # MEGA-ARCHITECTURE: Get ceiling height for this tile
                ceiling_y = get_ceiling_height_at_position(tile_center_x, tile_center_z, world)

                def make_draw_func(px=px, pz=pz, ceiling_y=ceiling_y, tile_size=tile_size):
                    return lambda surface, camera=camera, world=world: self.draw_world_poly(
                        surface, camera, world,
                        [(px, ceiling_y, pz), (px + tile_size, ceiling_y, pz),
                         (px + tile_size, ceiling_y, pz + tile_size),
                         (px, ceiling_y, pz + tile_size)],
                        self.ceiling_avg,
                        width_edges=0,
                        edge_color=None,
                        is_ceiling=True
                    )

                render_items.append((dist, make_draw_func()))

        return render_items

    def _get_pillars(self, camera, world, drawing_system=None):
        """Generate pillar render queue."""
        render_items = []
        render_range = RENDER_DISTANCE

        offset = PILLAR_SPACING // 2

        start_x = int((camera.x_s - render_range) // PILLAR_SPACING) * PILLAR_SPACING
        end_x = int((camera.x_s + render_range) // PILLAR_SPACING) * PILLAR_SPACING
        start_z = int((camera.z_s - render_range) // PILLAR_SPACING) * PILLAR_SPACING
        end_z = int((camera.z_s + render_range) // PILLAR_SPACING) * PILLAR_SPACING

        for px in range(start_x, end_x + PILLAR_SPACING, PILLAR_SPACING):
            for pz in range(start_z, end_z + PILLAR_SPACING, PILLAR_SPACING):
                pillar_x = px + offset
                pillar_z = pz + offset

                pillar_key = (pillar_x, pillar_z)

                if world.is_pillar_destroyed(pillar_key):
                    continue

                if world.has_pillar_at(pillar_x, pillar_z):
                    dist = math.sqrt((pillar_x - camera.x_s) ** 2 + (pillar_z - camera.z_s) ** 2)
                    if dist < RENDER_DISTANCE:
                        def make_draw_func(pillar_x=pillar_x, pillar_z=pillar_z, ds=drawing_system):
                            return lambda surface, camera=camera, world=world: self._draw_single_pillar(
                                surface, camera, world, pillar_x, pillar_z, ds)

                        render_items.append((dist, make_draw_func()))

        return render_items

    def _draw_single_pillar(self, surface, camera, world, px, pz, drawing_system=None):
        """Draw a single pillar."""
        s = PILLAR_SIZE
        # MEGA-ARCHITECTURE: Get height for this specific pillar
        h = get_ceiling_height_at_position(px + s/2, pz + s/2, world)
        floor_y = get_scaled_floor_y()
        edge_color = (220, 200, 70)
        pillar_key = (px, pz)

        # Front face
        self.draw_world_poly(
            surface, camera, world,
            [(px, h, pz), (px + s, h, pz), (px + s, floor_y, pz), (px, floor_y, pz)],
            self.pillar_avg,
            width_edges=1,
            edge_color=edge_color
        )
        if drawing_system:
            self._draw_pillar_face_drawings(surface, camera, world, pillar_key, 'front', 
                                           px, pz, s, h, floor_y, drawing_system)

        # Back face
        self.draw_world_poly(
            surface, camera, world,
            [(px + s, h, pz + s), (px, h, pz + s), (px, floor_y, pz + s), (px + s, floor_y, pz + s)],
            self.pillar_avg,
            width_edges=1,
            edge_color=edge_color
        )
        if drawing_system:
            self._draw_pillar_face_drawings(surface, camera, world, pillar_key, 'back',
                                           px, pz, s, h, floor_y, drawing_system)

        # Left face
        self.draw_world_poly(
            surface, camera, world,
            [(px, h, pz), (px, h, pz + s), (px, floor_y, pz + s), (px, floor_y, pz)],
            self.pillar_avg,
            width_edges=1,
            edge_color=edge_color
        )
        if drawing_system:
            self._draw_pillar_face_drawings(surface, camera, world, pillar_key, 'left',
                                           px, pz, s, h, floor_y, drawing_system)

        # Right face
        self.draw_world_poly(
            surface, camera, world,
            [(px + s, h, pz + s), (px + s, h, pz), (px + s, floor_y, pz), (px + s, floor_y, pz + s)],
            self.pillar_avg,
            width_edges=1,
            edge_color=edge_color
        )
        if drawing_system:
            self._draw_pillar_face_drawings(surface, camera, world, pillar_key, 'right',
                                           px, pz, s, h, floor_y, drawing_system)

    def _get_walls(self, camera, world, drawing_system=None):
        """Generate wall render queue."""
        render_items = []
        render_range = RENDER_DISTANCE

        start_x = int((camera.x_s - render_range) // PILLAR_SPACING) * PILLAR_SPACING
        end_x = int((camera.x_s + render_range) // PILLAR_SPACING) * PILLAR_SPACING
        start_z = int((camera.z_s - render_range) // PILLAR_SPACING) * PILLAR_SPACING
        end_z = int((camera.z_s + render_range) // PILLAR_SPACING) * PILLAR_SPACING

        for px in range(start_x, end_x + PILLAR_SPACING, PILLAR_SPACING):
            for pz in range(start_z, end_z + PILLAR_SPACING, PILLAR_SPACING):
                # Horizontal walls
                wall_key_h = tuple(sorted([(px, pz), (px + PILLAR_SPACING, pz)]))
                if world.has_wall_between(px, pz, px + PILLAR_SPACING, pz) and not world.is_wall_destroyed(wall_key_h):
                    wall_center_x = px + PILLAR_SPACING / 2
                    wall_center_z = pz
                    dist = math.sqrt((wall_center_x - camera.x_s) ** 2 + (wall_center_z - camera.z_s) ** 2)

                    def make_draw_func(px=px, pz=pz, ds=drawing_system):
                        return lambda surface, camera=camera, world=world: self._draw_connecting_wall(
                            surface, camera, world, px, pz, px + PILLAR_SPACING, pz, ds)

                    render_items.append((dist, make_draw_func()))

                # Vertical walls
                wall_key_v = tuple(sorted([(px, pz), (px, pz + PILLAR_SPACING)]))
                if world.has_wall_between(px, pz, px, pz + PILLAR_SPACING) and not world.is_wall_destroyed(wall_key_v):
                    wall_center_x = px
                    wall_center_z = pz + PILLAR_SPACING / 2
                    dist = math.sqrt((wall_center_x - camera.x_s) ** 2 + (wall_center_z - camera.z_s) ** 2)

                    def make_draw_func(px=px, pz=pz, ds=drawing_system):
                        return lambda surface, camera=camera, world=world: self._draw_connecting_wall(
                            surface, camera, world, px, pz, px, pz + PILLAR_SPACING, ds)

                    render_items.append((dist, make_draw_func()))

        return render_items

    def _draw_connecting_wall(self, surface, camera, world, x1, z1, x2, z2, drawing_system=None):
        """Draw a connecting wall with doorways/hallways and damage."""
        wall_key = tuple(sorted([(x1, z1), (x2, z2)]))

        # Check wall state
        wall_state = world.get_wall_state(wall_key)
        
        if wall_state == WallState.DESTROYED:
            world.spawn_rubble_pile(x1, z1, x2, z2)
            return

        # Also check legacy damage system for pre-damaged walls
        damage_state = world.get_wall_damage(wall_key)
        if damage_state < 0.2:
            world.spawn_rubble_pile(x1, z1, x2, z2)
            return

        # MEGA-ARCHITECTURE: Get height for this specific wall
        wall_center_x = (x1 + x2) / 2
        wall_center_z = (z1 + z2) / 2
        h = get_ceiling_height_at_position(wall_center_x, wall_center_z, world)
        floor_y = get_scaled_floor_y()

        # Color based on damage state
        if wall_state == WallState.FRACTURED or damage_state < 0.5:
            # Heavy damage - dark, dirty
            edge_color = (160, 140, 35)
            baseboard_color = (150, 130, 40)
            wall_color_mod = 0.75
        elif wall_state == WallState.CRACKED or damage_state < 0.8:
            # Cracked - slightly darkened
            edge_color = (190, 170, 42)
            baseboard_color = (180, 160, 50)
            wall_color_mod = 0.88
        else:
            # Intact
            edge_color = (220, 190, 50)
            baseboard_color = (210, 190, 60)
            wall_color_mod = 1.0

        baseboard_height = 8

        opening_type = world.get_doorway_type(x1, z1, x2, z2)

        if opening_type is None:
            self._draw_thick_wall_segment(surface, camera, world, x1, z1, x2, z2, h, floor_y,
                                          edge_color, baseboard_color, baseboard_height,
                                          wall_color_mod=wall_color_mod)
            # Draw cracks on top
            self._draw_wall_cracks(surface, camera, world, wall_key, x1, z1, x2, z2, h, floor_y)
        else:
            opening_width = HALLWAY_WIDTH if opening_type == "hallway" else 60

            if x1 == x2:  # Vertical wall
                wall_length = abs(z2 - z1)
                opening_start = min(z1, z2) + (wall_length - opening_width) / 2
                opening_end = opening_start + opening_width

                if opening_start > min(z1, z2):
                    self._draw_thick_wall_segment(surface, camera, world, x1, min(z1, z2), x2, opening_start,
                                                  h, floor_y, edge_color, baseboard_color, baseboard_height,
                                                  wall_color_mod=wall_color_mod)

                if opening_end < max(z1, z2):
                    self._draw_thick_wall_segment(surface, camera, world, x1, opening_end, x2, max(z1, z2),
                                                  h, floor_y, edge_color, baseboard_color, baseboard_height,
                                                  wall_color_mod=wall_color_mod)
            else:  # Horizontal wall
                wall_length = abs(x2 - x1)
                opening_start = min(x1, x2) + (wall_length - opening_width) / 2
                opening_end = opening_start + opening_width

                if opening_start > min(x1, x2):
                    self._draw_thick_wall_segment(surface, camera, world, min(x1, x2), z1, opening_start, z2,
                                                  h, floor_y, edge_color, baseboard_color, baseboard_height,
                                                  wall_color_mod=wall_color_mod)

                if opening_end < max(x1, x2):
                    self._draw_thick_wall_segment(surface, camera, world, opening_end, z1, max(x1, x2), z2,
                                                  h, floor_y, edge_color, baseboard_color, baseboard_height,
                                                  wall_color_mod=wall_color_mod)
        
        # Render drawings on this wall
        if drawing_system:
            self._draw_wall_drawings(surface, camera, world, wall_key, x1, z1, x2, z2, h, floor_y, drawing_system)

    def _draw_wall_cracks(self, surface, camera, world, wall_key, x1, z1, x2, z2, h, floor_y):
        """Draw crack lines on a damaged wall."""
        cracks = world.get_wall_cracks(wall_key)
        if not cracks:
            return

        half_thick = WALL_THICKNESS / 2
        wall_height = h - floor_y

        for u, v, angle, length in cracks:
            # Convert UV to world position
            if x1 == x2:  # Vertical wall
                wall_len = abs(z2 - z1)
                crack_x = x1 - half_thick  # Front face
                crack_z = min(z1, z2) + u * wall_len
                crack_y = floor_y + v * wall_height

                # Crack endpoints
                dz = math.cos(angle) * length * wall_len * 0.3
                dy = math.sin(angle) * length * wall_height * 0.3
                p1 = (crack_x, crack_y - dy, crack_z - dz)
                p2 = (crack_x, crack_y + dy, crack_z + dz)
            else:  # Horizontal wall
                wall_len = abs(x2 - x1)
                crack_x = min(x1, x2) + u * wall_len
                crack_z = z1 - half_thick  # Front face
                crack_y = floor_y + v * wall_height

                dx = math.cos(angle) * length * wall_len * 0.3
                dy = math.sin(angle) * length * wall_height * 0.3
                p1 = (crack_x - dx, crack_y - dy, crack_z)
                p2 = (crack_x + dx, crack_y + dy, crack_z)

            # Project and draw
            cam1 = camera.world_to_camera(*p1)
            cam2 = camera.world_to_camera(*p2)

            if cam1[2] > NEAR and cam2[2] > NEAR:
                screen1 = camera.project(cam1)
                screen2 = camera.project(cam2)

                if screen1 and screen2:
                    # Dark crack color
                    crack_color = (60, 50, 30)
                    try:
                        pygame.draw.line(surface, crack_color, screen1, screen2, 2)
                        # Slightly offset second line for thickness
                        pygame.draw.line(surface, (40, 35, 20),
                                        (screen1[0]+1, screen1[1]),
                                        (screen2[0]+1, screen2[1]), 1)
                    except:
                        pass

    def _draw_thick_wall_segment(self, surface, camera, world, x1, z1, x2, z2, h, floor_y,
                                 edge_color, baseboard_color, baseboard_height, wall_color_mod=1.0):
        """Draw a thick wall segment with baseboard."""
        half_thick = WALL_THICKNESS / 2
        
        # Apply damage modifier to colors
        wall_color = tuple(int(c * wall_color_mod) for c in self.wall_avg)
        wall_side_color = tuple(int(c * wall_color_mod) for c in (230, 210, 70))

        if x1 == x2:  # Vertical wall
            x = x1

            # Front face
            self.draw_world_poly(
                surface, camera, world,
                [(x - half_thick, h, z1), (x - half_thick, h, z2),
                 (x - half_thick, floor_y + baseboard_height, z2), (x - half_thick, floor_y + baseboard_height, z1)],
                wall_color, width_edges=1, edge_color=edge_color, is_wall=True
            )
            # Front baseboard
            self.draw_world_poly(
                surface, camera, world,
                [(x - half_thick, floor_y + baseboard_height, z1), (x - half_thick, floor_y + baseboard_height, z2),
                 (x - half_thick, floor_y, z2), (x - half_thick, floor_y, z1)],
                baseboard_color, width_edges=0, is_wall=True
            )

            # Back face
            self.draw_world_poly(
                surface, camera, world,
                [(x + half_thick, h, z2), (x + half_thick, h, z1),
                 (x + half_thick, floor_y + baseboard_height, z1), (x + half_thick, floor_y + baseboard_height, z2)],
                wall_color, width_edges=1, edge_color=edge_color, is_wall=True
            )
            # Back baseboard
            self.draw_world_poly(
                surface, camera, world,
                [(x + half_thick, floor_y + baseboard_height, z2), (x + half_thick, floor_y + baseboard_height, z1),
                 (x + half_thick, floor_y, z1), (x + half_thick, floor_y, z2)],
                baseboard_color, width_edges=0, is_wall=True
            )

            # End caps
            self.draw_world_poly(
                surface, camera, world,
                [(x - half_thick, h, z1), (x + half_thick, h, z1),
                 (x + half_thick, floor_y, z1), (x - half_thick, floor_y, z1)],
                wall_side_color, width_edges=1, edge_color=edge_color, is_wall=True
            )
            self.draw_world_poly(
                surface, camera, world,
                [(x + half_thick, h, z2), (x - half_thick, h, z2),
                 (x - half_thick, floor_y, z2), (x + half_thick, floor_y, z2)],
                wall_side_color, width_edges=1, edge_color=edge_color, is_wall=True
            )
        else:  # Horizontal wall
            z = z1

            # Front face
            self.draw_world_poly(
                surface, camera, world,
                [(x1, h, z - half_thick), (x2, h, z - half_thick),
                 (x2, floor_y + baseboard_height, z - half_thick), (x1, floor_y + baseboard_height, z - half_thick)],
                wall_color, width_edges=1, edge_color=edge_color, is_wall=True
            )
            # Front baseboard
            self.draw_world_poly(
                surface, camera, world,
                [(x1, floor_y + baseboard_height, z - half_thick), (x2, floor_y + baseboard_height, z - half_thick),
                 (x2, floor_y, z - half_thick), (x1, floor_y, z - half_thick)],
                baseboard_color, width_edges=0, is_wall=True
            )

            # Back face
            self.draw_world_poly(
                surface, camera, world,
                [(x2, h, z + half_thick), (x1, h, z + half_thick),
                 (x1, floor_y + baseboard_height, z + half_thick), (x2, floor_y + baseboard_height, z + half_thick)],
                wall_color, width_edges=1, edge_color=edge_color, is_wall=True
            )
            # Back baseboard
            self.draw_world_poly(
                surface, camera, world,
                [(x2, floor_y + baseboard_height, z + half_thick), (x1, floor_y + baseboard_height, z + half_thick),
                 (x1, floor_y, z + half_thick), (x2, floor_y, z + half_thick)],
                baseboard_color, width_edges=0, is_wall=True
            )

            # End caps
            self.draw_world_poly(
                surface, camera, world,
                [(x1, h, z + half_thick), (x1, h, z - half_thick),
                 (x1, floor_y, z - half_thick), (x1, floor_y, z + half_thick)],
                wall_side_color, width_edges=1, edge_color=edge_color, is_wall=True
            )
            self.draw_world_poly(
                surface, camera, world,
                [(x2, h, z - half_thick), (x2, h, z + half_thick),
                 (x2, floor_y, z + half_thick), (x2, floor_y, z - half_thick)],
                wall_side_color, width_edges=1, edge_color=edge_color, is_wall=True
            )

    # === DEBRIS RENDERING ===
    
    def _draw_pillar_face_drawings(self, surface, camera, world, pillar_key, face, px, pz, s, h, floor_y, drawing_system):
        """Draw graffiti on a pillar face."""
        strokes = drawing_system.get_drawings_for_pillar(pillar_key, face)
        
        # Include current stroke if drawing on this pillar face
        all_strokes = list(strokes)
        if (drawing_system.drawing_active and
            drawing_system.current_stroke and
            drawing_system.current_pillar == pillar_key and
            drawing_system.current_pillar_face == face and
            len(drawing_system.current_stroke) > 0):
            all_strokes.append(drawing_system.current_stroke)
        
        if not all_strokes:
            return
        
        pillar_height = h - floor_y
        
        for stroke in all_strokes:
            if len(stroke) < 1:
                continue
            
            # Convert UV to 3D world coordinates based on face
            world_points = []
            for u, v in stroke:
                if face == 'front':
                    world_x = px + u * s
                    world_z = pz
                    world_y = floor_y + v * pillar_height
                elif face == 'back':
                    world_x = px + (1 - u) * s
                    world_z = pz + s
                    world_y = floor_y + v * pillar_height
                elif face == 'left':
                    world_x = px
                    world_z = pz + u * s
                    world_y = floor_y + v * pillar_height
                elif face == 'right':
                    world_x = px + s
                    world_z = pz + (1 - u) * s
                    world_y = floor_y + v * pillar_height
                else:
                    continue
                
                world_points.append((world_x, world_y, world_z))
            
            # Project and draw
            screen_points = []
            for wx, wy, wz in world_points:
                cam_pt = camera.world_to_camera(wx, wy, wz)
                if cam_pt[2] > NEAR:
                    screen_pt = camera.project(cam_pt)
                    if screen_pt:
                        screen_points.append(screen_pt)
            
            # Draw the stroke
            if len(screen_points) == 1:
                sx, sy = screen_points[0]
                pygame.draw.circle(surface, (0, 0, 0), (int(sx), int(sy)), 3)
            elif len(screen_points) > 1:
                for i in range(len(screen_points) - 1):
                    try:
                        pygame.draw.line(surface, (0, 0, 0),
                                       (int(screen_points[i][0]), int(screen_points[i][1])),
                                       (int(screen_points[i+1][0]), int(screen_points[i+1][1])),
                                       3)
                    except:
                        pass
    
    def _draw_wall_drawings(self, surface, camera, world, wall_key, x1, z1, x2, z2, h, floor_y, drawing_system):
        """Draw graffiti/drawings on a wall."""
        strokes = drawing_system.get_drawings_for_wall(wall_key)
        
        # Also include current stroke if we're drawing on this wall
        all_strokes = list(strokes)
        if (drawing_system.drawing_active and 
            drawing_system.current_stroke and 
            drawing_system.current_wall == wall_key and
            len(drawing_system.current_stroke) > 0):
            # Add the current stroke being drawn for real-time preview
            all_strokes.append(drawing_system.current_stroke)
        
        if not all_strokes:
            return
        
        half_thick = WALL_THICKNESS / 2
        wall_height = h - floor_y
        
        # Determine wall orientation
        is_horizontal = abs(z1 - z2) < 0.1
        
        # Draw all strokes (including current one if active)
        for stroke in all_strokes:
            if len(stroke) < 1:
                continue
            
            # Convert stroke points to 3D world coordinates
            world_points = []
            for u, v in stroke:
                if is_horizontal:
                    # Horizontal wall (along X axis)
                    wall_length = abs(x2 - x1)
                    world_x = min(x1, x2) + u * wall_length
                    world_z = z1 - half_thick  # Front face
                    world_y = floor_y + v * wall_height
                else:
                    # Vertical wall (along Z axis)
                    wall_length = abs(z2 - z1)
                    world_x = x1 - half_thick  # Front face
                    world_z = min(z1, z2) + u * wall_length
                    world_y = floor_y + v * wall_height
                
                world_points.append((world_x, world_y, world_z))
            
            # Project to screen and draw
            screen_points = []
            for wx, wy, wz in world_points:
                cam_pt = camera.world_to_camera(wx, wy, wz)
                if cam_pt[2] > NEAR:
                    screen_pt = camera.project(cam_pt)
                    if screen_pt:
                        screen_points.append(screen_pt)
            
            # Draw the stroke
            if len(screen_points) == 1:
                # Single point - draw circle
                sx, sy = screen_points[0]
                pygame.draw.circle(surface, (0, 0, 0), (int(sx), int(sy)), 3)
            elif len(screen_points) > 1:
                # Multiple points - draw connected lines
                for i in range(len(screen_points) - 1):
                    try:
                        pygame.draw.line(surface, (0, 0, 0), 
                                       (int(screen_points[i][0]), int(screen_points[i][1])),
                                       (int(screen_points[i+1][0]), int(screen_points[i+1][1])), 
                                       3)
                    except:
                        pass

    # === DEBRIS RENDERING ===

    def _render_debris(self, surface, camera, world):
        """Render all debris particles."""
        DEBRIS_RENDER_DIST = 600.0
        px, pz = camera.x_s, camera.z_s

        debris_to_render = []
        for d in world.debris_pieces:
            if not d.active:
                continue

            dx = d.cx - px
            dz = d.cz - pz
            dist_sq = dx * dx + dz * dz
            if dist_sq > DEBRIS_RENDER_DIST * DEBRIS_RENDER_DIST:
                continue

            cam_pos = camera.world_to_camera(d.cx, d.cy, d.cz)
            if cam_pos[2] <= NEAR:
                continue

            screen_pos = camera.project(cam_pos)
            if screen_pos is None:
                continue

            sx, sy = screen_pos
            if 0 <= sx < camera.width and 0 <= sy < camera.height:
                dist = math.sqrt(dist_sq)
                size = max(1, int(3 * (1.0 - dist / DEBRIS_RENDER_DIST)))
                debris_to_render.append((cam_pos[2], sx, sy, size, d.color))

        # Sort back-to-front
        debris_to_render.sort(key=lambda x: x[0], reverse=True)

        for _, sx, sy, size, color in debris_to_render:
            if size == 1:
                try:
                    surface.set_at((int(sx), int(sy)), color)
                except:
                    pass
            else:
                pygame.draw.circle(surface, color, (int(sx), int(sy)), size)
    
    def _render_current_drawing_stroke(self, surface, camera, drawing_system):
        """Render the stroke currently being drawn (in progress)."""
        if not drawing_system or not drawing_system.drawing_active:
            return
        
        if not drawing_system.current_stroke or len(drawing_system.current_stroke) < 1:
            return
        
        # We need to get the wall info to project properly
        # For now, render it as a screen-space preview by projecting a point slightly in front of camera
        # This is a simplified approach - we'll render it in 2D screen space
        
        # Actually, let's do it properly by storing the wall key when we start drawing
        # For now, just render as a simple crosshair extension
        pass

    # === MAIN RENDER ===

    def render(self, surface, camera, world, player, drawing_system=None):
        """Main render method."""
        target_surface = self.render_surface
        target_surface.fill(BLACK)

        # Temporarily adjust camera dimensions for scaled rendering
        original_width, original_height = camera.width, camera.height
        camera.width = target_surface.get_width()
        camera.height = target_surface.get_height()

        # Build render queue
        render_queue = []
        render_queue.extend(self._get_floor_tiles(camera, world))
        render_queue.extend(self._get_ceiling_tiles(camera, world))
        render_queue.extend(self._get_pillars(camera, world, drawing_system))
        render_queue.extend(self._get_walls(camera, world, drawing_system))

        # Sort back-to-front
        render_queue.sort(key=lambda item: item[0], reverse=True)

        # Draw geometry
        for depth, draw_func in render_queue:
            draw_func(target_surface)

        # Draw debris
        self._render_debris(target_surface, camera, world)

        # Restore camera dimensions
        camera.width, camera.height = original_width, original_height

        # Scale up if needed
        if self.render_scale < 1.0:
            final_surface = pygame.Surface((self.width, self.height))
            pygame.transform.smoothscale(target_surface, (self.width, self.height), final_surface)
        else:
            final_surface = target_surface.copy()

        surface.blit(final_surface, (0, 0))

        # Crosshair
        cx, cy = self.width // 2, self.height // 2
        pygame.draw.circle(surface, (255, 255, 100), (cx, cy), 3, 1)
