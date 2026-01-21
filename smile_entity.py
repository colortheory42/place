"""
Smile Entity - Collision-Bound Anomaly
=======================================
A rolling sphere with a static smile that follows the player
only as permitted by geometry. No intelligence, no cheating.

Core principles:
- Geometry is authority
- Movement is intent, collision decides outcome
- The smile never tilts (billboard effect)
- Rolling is visual justification, not physics
- No pathfinding, no navigation, no teleporting
"""

import math
import numpy as np
import pygame
from config import NEAR, get_scaled_floor_y


class SmileEntity:
    """
    Collision-bound anomaly that rolls toward player.
    
    Movement philosophy:
    - Weak directional bias toward player
    - Collision system is the final authority
    - No navigation logic, no pathfinding
    - Stops when blocked, slides when angled
    """
    
    def __init__(self, spawn_x, spawn_z, radius=20.0, speed=15.0):
        # Position (always on floor)
        self.x = spawn_x
        self.z = spawn_z
        self.radius = radius

        # Movement
        self.speed = speed  # Very slow, weak bias
        self.vx = 0.0  # Current velocity
        self.vz = 0.0

        # Rolling visualization (NOT physics)
        self.rotation_x = 0.0  # Roll around X axis
        self.rotation_z = 0.0  # Roll around Z axis

        # Zone constraint
        self.spawn_x = spawn_x
        self.spawn_z = spawn_z
        self.max_roam_distance = 800.0  # Stays within local area

        # Behavior flags
        self.active = True
        self.centered_in_view = False  # Set by external observer

        # Collision history (for stuck detection)
        self.last_moved_time = 0.0
        self.stuck_threshold = 3.0  # Seconds without movement = stuck

    def update(self, dt, player_pos, collision_system, camera_looking_at_me=False):
        """
        Update entity position with collision-bound following.

        Args:
            dt: Delta time
            player_pos: (x, z) tuple of player position
            collision_system: CollisionSystem instance for movement validation
            camera_looking_at_me: True if player is looking directly at entity
        """
        if not self.active:
            return

        # Check zone constraint - slow down near edge
        distance_from_spawn = math.sqrt(
            (self.x - self.spawn_x)**2 +
            (self.z - self.spawn_z)**2
        )

        if distance_from_spawn > self.max_roam_distance:
            # Beyond boundary - stop following, drift back
            self._drift_toward_spawn(dt, collision_system)
            return

        # When centered in camera view, often become motionless
        if camera_looking_at_me:
            if np.random.random() < 0.85:  # 85% chance to freeze when observed
                return

        # Calculate intent vector toward player
        player_x, player_z = player_pos
        dx = player_x - self.x
        dz = player_z - self.z
        distance = math.sqrt(dx*dx + dz*dz)

        # Stop following if too close (personal space)
        if distance < self.radius * 3:
            return

        # Normalize direction
        if distance > 0.001:
            dx /= distance
            dz /= distance
        else:
            return

        # Apply weak bias (this is intent, not guaranteed movement)
        move_amount = self.speed * dt

        # Slow down near zone edge
        edge_factor = 1.0 - (distance_from_spawn / self.max_roam_distance)
        edge_factor = max(0.3, edge_factor)  # Never completely stop following
        move_amount *= edge_factor

        intent_x = dx * move_amount
        intent_z = dz * move_amount

        # Apply movement through collision system
        from_pos = (self.x, self.z)
        to_pos = (self.x + intent_x, self.z + intent_z)

        final_x, final_z, collided = collision_system.resolve_collision(from_pos, to_pos)

        # Calculate actual movement (what collision allowed)
        actual_dx = final_x - self.x
        actual_dz = final_z - self.z
        actual_distance = math.sqrt(actual_dx*actual_dx + actual_dz*actual_dz)

        # Update position
        old_x, old_z = self.x, self.z
        self.x = final_x
        self.z = final_z

        # Update rolling rotation (visual only)
        if actual_distance > 0.001:
            # Roll amount based on actual movement
            roll_amount = actual_distance / self.radius

            # Determine roll axes based on movement direction
            if abs(actual_dz) > 0.001:
                self.rotation_x += roll_amount * (actual_dz / actual_distance)
            if abs(actual_dx) > 0.001:
                self.rotation_z += roll_amount * (actual_dx / actual_distance)

            # Track movement for stuck detection
            self.last_moved_time = 0.0
        else:
            # Didn't move - might be stuck
            self.last_moved_time += dt

    def _drift_toward_spawn(self, dt, collision_system):
        """Gentle drift back toward spawn when outside zone."""
        dx = self.spawn_x - self.x
        dz = self.spawn_z - self.z
        distance = math.sqrt(dx*dx + dz*dz)

        if distance > 0.001:
            dx /= distance
            dz /= distance

            # Very weak drift
            drift_amount = self.speed * 0.3 * dt
            intent_x = dx * drift_amount
            intent_z = dz * drift_amount

            from_pos = (self.x, self.z)
            to_pos = (self.x + intent_x, self.z + intent_z)

            final_x, final_z, _ = collision_system.resolve_collision(from_pos, to_pos)
            self.x = final_x
            self.z = final_z

    def get_world_position(self):
        """Get entity center position in world space."""
        floor_y = get_scaled_floor_y()
        return (self.x, floor_y + self.radius, self.z)

    def is_visible_from(self, camera, world, max_distance=3000.0):
        """Check if entity should be rendered from camera perspective."""
        cam_x, cam_y, cam_z = camera.x_s, camera.y_s, camera.z_s

        # Distance check
        dx = self.x - cam_x
        dz = self.z - cam_z
        distance = math.sqrt(dx*dx + dz*dz)

        if distance > max_distance:
            return False

        # Frustum check (simple - just check if in front of camera)
        world_x, world_y, world_z = self.get_world_position()
        cam_space = camera.world_to_camera(world_x, world_y, world_z)

        if cam_space[2] <= NEAR:
            return False

        # Simple occlusion check - only check direct line
        if not self._check_line_of_sight_simple(camera, world):
            return False

        return True

    def _check_line_of_sight_simple(self, camera, world):
        """
        Simplified line of sight check.
        Checks all walls/pillars between camera and entity.
        More tolerant to avoid disappearing at edges.
        """
        from raycasting import ray_intersects_triangle
        from config import PILLAR_SPACING, WALL_THICKNESS, PILLAR_SIZE, get_scaled_wall_height, get_scaled_floor_y

        # Ray from camera to entity
        cam_x, cam_y, cam_z = camera.x_s, camera.y_s, camera.z_s
        entity_x, entity_y, entity_z = self.get_world_position()

        # Direction vector
        dx = entity_x - cam_x
        dy = entity_y - cam_y
        dz = entity_z - cam_z
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)

        if distance < 0.001:
            return True

        # Normalized direction
        ray_dir = np.array([dx/distance, dy/distance, dz/distance])
        ray_origin = np.array([cam_x, cam_y, cam_z])

        h = get_scaled_wall_height()
        floor_y = get_scaled_floor_y()

        # Check ALL geometry between camera and entity
        min_x = min(cam_x, entity_x)
        max_x = max(cam_x, entity_x)
        min_z = min(cam_z, entity_z)
        max_z = max(cam_z, entity_z)

        # Expand check range slightly
        check_range = PILLAR_SPACING
        start_x = int((min_x - check_range) // PILLAR_SPACING) * PILLAR_SPACING
        end_x = int((max_x + check_range) // PILLAR_SPACING) * PILLAR_SPACING
        start_z = int((min_z - check_range) // PILLAR_SPACING) * PILLAR_SPACING
        end_z = int((max_z + check_range) // PILLAR_SPACING) * PILLAR_SPACING

        half_thick = WALL_THICKNESS / 2

        # Very generous tolerance - entity needs to be well behind the wall
        occlusion_tolerance = self.radius * 3.5

        # Check walls
        for px in range(start_x, end_x + PILLAR_SPACING, PILLAR_SPACING):
            for pz in range(start_z, end_z + PILLAR_SPACING, PILLAR_SPACING):

                # Horizontal walls
                if world.has_wall_between(px, pz, px + PILLAR_SPACING, pz):
                    wall_key = tuple(sorted([(px, pz), (px + PILLAR_SPACING, pz)]))
                    if not world.is_wall_destroyed(wall_key):
                        z = pz
                        x1, x2 = px, px + PILLAR_SPACING

                        # Check both faces
                        faces = [
                            [(x1, h, z - half_thick), (x2, h, z - half_thick),
                             (x2, floor_y, z - half_thick), (x1, floor_y, z - half_thick)],
                            [(x1, h, z + half_thick), (x2, h, z + half_thick),
                             (x2, floor_y, z + half_thick), (x1, floor_y, z + half_thick)]
                        ]

                        for face in faces:
                            v0, v1, v2, v3 = face
                            for tri in [(v0, v1, v2), (v0, v2, v3)]:
                                hit = ray_intersects_triangle(ray_origin, ray_dir, *tri)
                                if hit and hit[0] < distance - occlusion_tolerance:
                                    return False

                # Vertical walls
                if world.has_wall_between(px, pz, px, pz + PILLAR_SPACING):
                    wall_key = tuple(sorted([(px, pz), (px, pz + PILLAR_SPACING)]))
                    if not world.is_wall_destroyed(wall_key):
                        x = px
                        z1, z2 = pz, pz + PILLAR_SPACING

                        # Check both faces
                        faces = [
                            [(x - half_thick, h, z1), (x - half_thick, h, z2),
                             (x - half_thick, floor_y, z2), (x - half_thick, floor_y, z1)],
                            [(x + half_thick, h, z1), (x + half_thick, h, z2),
                             (x + half_thick, floor_y, z2), (x + half_thick, floor_y, z1)]
                        ]

                        for face in faces:
                            v0, v1, v2, v3 = face
                            for tri in [(v0, v1, v2), (v0, v2, v3)]:
                                hit = ray_intersects_triangle(ray_origin, ray_dir, *tri)
                                if hit and hit[0] < distance - occlusion_tolerance:
                                    return False

                # Check pillars
                offset = PILLAR_SPACING // 2
                pillar_x = px + offset
                pillar_z = pz + offset

                if world.has_pillar_at(pillar_x, pillar_z):
                    pillar_key = (pillar_x, pillar_z)
                    if not world.is_pillar_destroyed(pillar_key):
                        s = PILLAR_SIZE

                        # Check all 4 faces of pillar
                        pillar_faces = [
                            # Front face
                            [(pillar_x, h, pillar_z), (pillar_x + s, h, pillar_z),
                             (pillar_x + s, floor_y, pillar_z), (pillar_x, floor_y, pillar_z)],
                            # Back face
                            [(pillar_x + s, h, pillar_z + s), (pillar_x, h, pillar_z + s),
                             (pillar_x, floor_y, pillar_z + s), (pillar_x + s, floor_y, pillar_z + s)],
                            # Left face
                            [(pillar_x, h, pillar_z), (pillar_x, h, pillar_z + s),
                             (pillar_x, floor_y, pillar_z + s), (pillar_x, floor_y, pillar_z)],
                            # Right face
                            [(pillar_x + s, h, pillar_z + s), (pillar_x + s, h, pillar_z),
                             (pillar_x + s, floor_y, pillar_z), (pillar_x + s, floor_y, pillar_z + s)]
                        ]

                        for face in pillar_faces:
                            v0, v1, v2, v3 = face
                            for tri in [(v0, v1, v2), (v0, v2, v3)]:
                                hit = ray_intersects_triangle(ray_origin, ray_dir, *tri)
                                if hit and hit[0] < distance - occlusion_tolerance:
                                    return False

        return True

    def _check_line_of_sight(self, camera, world):
        """
        Check if there's a clear line of sight from camera to entity.
        Returns False if walls or pillars block the view.
        """
        from raycasting import ray_intersects_triangle
        from config import PILLAR_SPACING, WALL_THICKNESS, PILLAR_SIZE, get_scaled_wall_height, get_scaled_floor_y

        # Ray from camera to entity
        cam_x, cam_y, cam_z = camera.x_s, camera.y_s, camera.z_s
        entity_x, entity_y, entity_z = self.get_world_position()

        # Direction vector
        dx = entity_x - cam_x
        dy = entity_y - cam_y
        dz = entity_z - cam_z
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)

        if distance < 0.001:
            return True

        # Normalized direction
        ray_dir = np.array([dx/distance, dy/distance, dz/distance])
        ray_origin = np.array([cam_x, cam_y, cam_z])

        # Check walls and pillars in between
        h = get_scaled_wall_height()
        floor_y = get_scaled_floor_y()

        # Get range to check
        min_x = min(cam_x, entity_x)
        max_x = max(cam_x, entity_x)
        min_z = min(cam_z, entity_z)
        max_z = max(cam_z, entity_z)

        # Expand slightly
        check_range = PILLAR_SPACING * 2
        start_x = int((min_x - check_range) // PILLAR_SPACING) * PILLAR_SPACING
        end_x = int((max_x + check_range) // PILLAR_SPACING) * PILLAR_SPACING
        start_z = int((min_z - check_range) // PILLAR_SPACING) * PILLAR_SPACING
        end_z = int((max_z + check_range) // PILLAR_SPACING) * PILLAR_SPACING

        half_thick = WALL_THICKNESS / 2

        # Check walls
        for px in range(start_x, end_x + PILLAR_SPACING, PILLAR_SPACING):
            for pz in range(start_z, end_z + PILLAR_SPACING, PILLAR_SPACING):

                # Horizontal walls
                if world.has_wall_between(px, pz, px + PILLAR_SPACING, pz):
                    wall_key = tuple(sorted([(px, pz), (px + PILLAR_SPACING, pz)]))
                    if not world.is_wall_destroyed(wall_key):
                        z = pz
                        x1, x2 = px, px + PILLAR_SPACING

                        # Simple check - both faces
                        faces = [
                            [(x1, h, z - half_thick), (x2, h, z - half_thick),
                             (x2, floor_y, z - half_thick), (x1, floor_y, z - half_thick)],
                            [(x1, h, z + half_thick), (x2, h, z + half_thick),
                             (x2, floor_y, z + half_thick), (x1, floor_y, z + half_thick)]
                        ]

                        for face in faces:
                            v0, v1, v2, v3 = face
                            for tri in [(v0, v1, v2), (v0, v2, v3)]:
                                hit = ray_intersects_triangle(ray_origin, ray_dir, *tri)
                                # Only occlude if wall is significantly in front (tolerance for edge cases)
                                if hit and hit[0] < distance - self.radius * 2:
                                    return False

                # Vertical walls
                if world.has_wall_between(px, pz, px, pz + PILLAR_SPACING):
                    wall_key = tuple(sorted([(px, pz), (px, pz + PILLAR_SPACING)]))
                    if not world.is_wall_destroyed(wall_key):
                        x = px
                        z1, z2 = pz, pz + PILLAR_SPACING

                        faces = [
                            [(x - half_thick, h, z1), (x - half_thick, h, z2),
                             (x - half_thick, floor_y, z2), (x - half_thick, floor_y, z1)],
                            [(x + half_thick, h, z1), (x + half_thick, h, z2),
                             (x + half_thick, floor_y, z2), (x + half_thick, floor_y, z1)]
                        ]

                        for face in faces:
                            v0, v1, v2, v3 = face
                            for tri in [(v0, v1, v2), (v0, v2, v3)]:
                                hit = ray_intersects_triangle(ray_origin, ray_dir, *tri)
                                # Only occlude if wall is significantly in front (tolerance for edge cases)
                                if hit and hit[0] < distance - self.radius * 2:
                                    return False

                # Check pillars
                offset = PILLAR_SPACING // 2
                pillar_x = px + offset
                pillar_z = pz + offset

                if world.has_pillar_at(pillar_x, pillar_z):
                    pillar_key = (pillar_x, pillar_z)
                    if not world.is_pillar_destroyed(pillar_key):
                        s = PILLAR_SIZE

                        # All 4 faces of pillar
                        pillar_faces = [
                            # Front face
                            [(pillar_x, h, pillar_z), (pillar_x + s, h, pillar_z),
                             (pillar_x + s, floor_y, pillar_z), (pillar_x, floor_y, pillar_z)],
                            # Back face
                            [(pillar_x + s, h, pillar_z + s), (pillar_x, h, pillar_z + s),
                             (pillar_x, floor_y, pillar_z + s), (pillar_x + s, floor_y, pillar_z + s)],
                            # Left face
                            [(pillar_x, h, pillar_z), (pillar_x, h, pillar_z + s),
                             (pillar_x, floor_y, pillar_z + s), (pillar_x, floor_y, pillar_z)],
                            # Right face
                            [(pillar_x + s, h, pillar_z + s), (pillar_x + s, h, pillar_z),
                             (pillar_x + s, floor_y, pillar_z), (pillar_x + s, floor_y, pillar_z + s)]
                        ]

                        for face in pillar_faces:
                            v0, v1, v2, v3 = face
                            for tri in [(v0, v1, v2), (v0, v2, v3)]:
                                hit = ray_intersects_triangle(ray_origin, ray_dir, *tri)
                                # Only occlude if pillar is significantly in front (tolerance for edge cases)
                                if hit and hit[0] < distance - self.radius * 2:
                                    return False

        return True

    def render(self, surface, camera):
        """
        Render the smile entity.

        The sphere rolls, but the smile face always stays upright.
        """
        if not self.active:
            return

        # Get position
        world_x, world_y, world_z = self.get_world_position()

        # Transform to camera space
        cam_pos = camera.world_to_camera(world_x, world_y, world_z)
        if cam_pos[2] <= NEAR:
            return

        # Project to screen
        screen_pos = camera.project(cam_pos)
        if not screen_pos:
            return

        sx, sy = screen_pos

        # Calculate screen radius based on distance
        distance = cam_pos[2]
        screen_radius = int((self.radius / distance) * camera.width * 0.5)

        if screen_radius < 2:
            return

        # Draw perfect sphere (bright yellow circle)
        sphere_color = (255, 240, 100)
        pygame.draw.circle(surface, sphere_color, (int(sx), int(sy)), screen_radius)

        # Draw smile (ALWAYS UPRIGHT - billboard effect)
        self._render_smile(surface, int(sx), int(sy), screen_radius)

        # Optional: Draw debug info
        if False:  # Set to True for debugging
            debug_color = (255, 0, 0)
            font = pygame.font.SysFont("monospace", 10)
            debug_text = f"d:{distance:.0f}"
            text_surf = font.render(debug_text, True, debug_color)
            surface.blit(text_surf, (int(sx - 20), int(sy - screen_radius - 15)))

    def _render_smile(self, surface, center_x, center_y, radius):
        """
        Render the smile face.
        Face is always upright regardless of sphere rotation.
        """
        if radius < 5:
            return

        # Smile color (dark, contrasting with yellow)
        smile_color = (40, 40, 40)

        # Eyes
        eye_radius = max(2, radius // 6)
        eye_offset_x = radius // 3
        eye_offset_y = -radius // 4

        # Left eye
        pygame.draw.circle(
            surface,
            smile_color,
            (center_x - eye_offset_x, center_y + eye_offset_y),
            eye_radius
        )

        # Right eye
        pygame.draw.circle(
            surface,
            smile_color,
            (center_x + eye_offset_x, center_y + eye_offset_y),
            eye_radius
        )

        # Smile (arc)
        if radius > 8:
            smile_width = max(2, radius // 10)
            smile_rect = pygame.Rect(
                center_x - radius // 2,
                center_y - radius // 4,
                radius,
                radius
            )

            # Draw arc for smile (upward curve)
            pygame.draw.arc(
                surface,
                smile_color,
                smile_rect,
                math.pi + 0.2,  # Start angle (radians) - bottom left
                2 * math.pi - 0.2,  # End angle - bottom right
                smile_width
            )

    def is_stuck(self):
        """Check if entity hasn't moved in a while."""
        return self.last_moved_time > self.stuck_threshold

    def get_state_for_save(self):
        """Get state for saving."""
        return {
            'x': self.x,
            'z': self.z,
            'radius': self.radius,
            'speed': self.speed,
            'spawn_x': self.spawn_x,
            'spawn_z': self.spawn_z,
            'rotation_x': self.rotation_x,
            'rotation_z': self.rotation_z,
            'active': self.active
        }

    def load_state(self, data):
        """Load state from save data."""
        self.x = data.get('x', self.x)
        self.z = data.get('z', self.z)
        self.radius = data.get('radius', self.radius)
        self.speed = data.get('speed', self.speed)
        self.spawn_x = data.get('spawn_x', self.spawn_x)
        self.spawn_z = data.get('spawn_z', self.spawn_z)
        self.rotation_x = data.get('rotation_x', 0.0)
        self.rotation_z = data.get('rotation_z', 0.0)
        self.active = data.get('active', True)
        self.last_moved_time = 0.0


class SmileManager:
    """
    Manages multiple Smile entities.
    Spawns them in specific zones, updates them, renders them.
    """

    def __init__(self):
        self.entities = []
        self.spawn_zones = []  # List of (x, z, radius) spawn zones

    def add_spawn_zone(self, zone_x, zone_z, count=1):
        """Add a zone where Smiles can spawn."""
        import random

        for _ in range(count):
            # Random offset within zone
            offset_x = random.uniform(-200, 200)
            offset_z = random.uniform(-200, 200)

            spawn_x = zone_x + offset_x
            spawn_z = zone_z + offset_z

            # Create entity
            smile = SmileEntity(
                spawn_x,
                spawn_z,
                radius=20.0,
                speed=random.uniform(100.0, 125.0)  # Slight speed variation
            )

            self.entities.append(smile)

    def update(self, dt, player_pos, collision_system, camera):
        """Update all entities."""
        for smile in self.entities:
            if not smile.active:
                continue

            # Check if player is looking at this entity
            looking_at = self._is_camera_centered_on(smile, camera)

            smile.update(dt, player_pos, collision_system, looking_at)

    def render(self, surface, camera, world):
        """
        Render all visible entities.

        Args:
            surface: pygame surface to draw on
            camera: Camera instance
            world: World instance (for occlusion checking)
        """
        # Sort by distance (far to near) for proper occlusion
        visible = [
            (smile, self._distance_to_camera(smile, camera))
            for smile in self.entities
            if smile.is_visible_from(camera, world)
        ]

        visible.sort(key=lambda x: x[1], reverse=True)

        for smile, _ in visible:
            smile.render(surface, camera)

    def _is_camera_centered_on(self, smile, camera):
        """Check if camera is looking directly at entity (center of screen)."""
        world_pos = smile.get_world_position()
        cam_pos = camera.world_to_camera(*world_pos)

        if cam_pos[2] <= NEAR:
            return False

        screen_pos = camera.project(cam_pos)
        if not screen_pos:
            return False

        sx, sy = screen_pos
        center_x = camera.width // 2
        center_y = camera.height // 2

        # Check if within center region (20% of screen)
        threshold_x = camera.width * 0.1
        threshold_y = camera.height * 0.1

        return (abs(sx - center_x) < threshold_x and
                abs(sy - center_y) < threshold_y)

    def _distance_to_camera(self, smile, camera):
        """Calculate distance from camera to entity."""
        dx = smile.x - camera.x_s
        dz = smile.z - camera.z_s
        return math.sqrt(dx*dx + dz*dz)

    def get_state_for_save(self):
        """Get all entities state for saving."""
        return {
            'entities': [smile.get_state_for_save() for smile in self.entities]
        }

    def load_state(self, data):
        """Load entities from save data."""
        self.entities.clear()

        for entity_data in data.get('entities', []):
            smile = SmileEntity(0, 0)  # Temporary init
            smile.load_state(entity_data)
            self.entities.append(smile)

    def spawn_near_player(self, player_x, player_z, distance=500.0):
        """Spawn a new entity at a distance from player."""
        import random

        angle = random.uniform(0, 2 * math.pi)
        spawn_x = player_x + math.cos(angle) * distance
        spawn_z = player_z + math.sin(angle) * distance

        smile = SmileEntity(spawn_x, spawn_z)
        self.entities.append(smile)

        return smile