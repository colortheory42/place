"""
World system.
Manages zones, walls, pillars, destruction state, and collision queries.
"""

import math
import random
from enum import Enum, auto
from config import (
    PILLAR_SPACING, PILLAR_SIZE, PILLAR_MODE, WALL_THICKNESS,
    ZONE_SIZE, HALLWAY_WIDTH, WALL_COLOR, PILLAR_COLOR,
    get_scaled_wall_height, get_scaled_floor_y
)
from procedural import ProceduralZone
from debris import Debris
from events import event_bus, EventType
from ceiling_heights import get_ceiling_height_at_position, get_room_size_at_position


class WallState(Enum):
    """Progressive damage states for walls."""
    INTACT = auto()      # Full health, no visible damage
    CRACKED = auto()     # Hairline cracks, still solid
    FRACTURED = auto()   # Major cracks, chunks missing
    BREAKING = auto()    # Actively falling/crumbling
    DESTROYED = auto()   # Gone, only debris remains


class World:
    """World state and procedural queries."""

    def __init__(self, world_seed=None):
        self.world_seed = world_seed if world_seed is not None else random.randint(0, 999999)

        # Caches
        self.pillar_cache = {}
        self.wall_cache = {}
        self.zone_cache = {}
        self.lamp_cache = {}
        self.trap_cache = {}

        # Destruction state
        self.destroyed_walls = set()
        self.destroyed_pillars = set()
        self.destroyed_lamps = set()
        self.triggered_traps = set()  # Traps that have been triggered
        self.pre_damaged_walls = {}  # wall_key -> damage_state (0.0-1.0)

        # Progressive wall damage system
        self.wall_states = {}   # wall_key -> WallState
        self.wall_health = {}   # wall_key -> float (0.0 to 1.0)
        self.wall_cracks = {}   # wall_key -> list of (u, v, angle, length) tuples

        # Debris
        self.debris_pieces = []
        self._spawned_rubble = set()

        print(f"World seed: {self.world_seed}")

    # === ZONE SYSTEM ===

    def get_zone_at(self, x, z):
        """Get zone coordinates for a world position."""
        zone_x = int(x // ZONE_SIZE)
        zone_z = int(z // ZONE_SIZE)
        return (zone_x, zone_z)

    def get_zone_properties(self, zone_x, zone_z):
        """Get cached zone properties."""
        key = (zone_x, zone_z)
        if key not in self.zone_cache:
            self.zone_cache[key] = ProceduralZone.get_zone_properties(zone_x, zone_z, self.world_seed)
        return self.zone_cache[key]

    # === PILLAR QUERIES ===

    def has_pillar_at(self, px, pz):
        """Check if there's a pillar at this position."""
        key = (px, pz)
        if key in self.pillar_cache:
            return self.pillar_cache[key]

        if PILLAR_MODE == "none":
            self.pillar_cache[key] = False
            return False

        offset = PILLAR_SPACING // 2
        is_on_pillar_grid = (px % PILLAR_SPACING == offset) and (pz % PILLAR_SPACING == offset)

        if not is_on_pillar_grid:
            self.pillar_cache[key] = False
            return False

        if PILLAR_MODE == "all":
            self.pillar_cache[key] = True
            return True

        # Deterministic random based on position
        seed = hash((px, pz, self.world_seed)) % 100000
        rng = random.Random(seed)

        probability_map = {
            "sparse": 0.10,
            "normal": 0.30,
            "dense": 0.60,
        }

        probability = probability_map.get(PILLAR_MODE, 0.0)
        has_pillar = rng.random() < probability

        self.pillar_cache[key] = has_pillar
        return has_pillar

    def is_pillar_destroyed(self, pillar_key):
        """Check if a pillar has been destroyed."""
        return pillar_key in self.destroyed_pillars

    # === WALL QUERIES ===

    def has_wall_between(self, x1, z1, x2, z2):
        """Check if there's a wall between two grid points - ZONE-AWARE VERSION."""
        key = tuple(sorted([(x1, z1), (x2, z2)]))

        if key in self.wall_cache:
            return self.wall_cache[key]

        # === MEGA-ARCHITECTURE: Zone-aware grid ===
        center_x = (x1 + x2) / 2
        center_z = (z1 + z2) / 2
        room_size = get_room_size_at_position(center_x, center_z, self)
        
        is_horizontal = (z1 == z2)
        is_vertical = (x1 == x2)

        if not (is_horizontal or is_vertical):
            self.wall_cache[key] = False
            return False
        
        # Check if wall aligns with this zone's grid
        tolerance = 1.0  # Allow small floating point errors
        
        if is_horizontal:
            # Wall along X axis - check if Z coordinate is on grid
            z_on_grid = (abs(z1 % room_size) < tolerance or 
                         abs(z1 % room_size - room_size) < tolerance)
            if not z_on_grid:
                self.wall_cache[key] = False
                return False
            
            # Check if X span matches room size
            x_span = abs(x2 - x1)
            if abs(x_span - room_size) > tolerance:
                self.wall_cache[key] = False
                return False
                
        elif is_vertical:
            # Wall along Z axis - check if X coordinate is on grid
            x_on_grid = (abs(x1 % room_size) < tolerance or 
                         abs(x1 % room_size - room_size) < tolerance)
            if not x_on_grid:
                self.wall_cache[key] = False
                return False
            
            # Check if Z span matches room size
            z_span = abs(z2 - z1)
            if abs(z_span - room_size) > tolerance:
                self.wall_cache[key] = False
                return False
        # === END MEGA-ARCHITECTURE CHANGES ===

        # Check for pre-existing damage
        if key not in self.pre_damaged_walls:
            zone = self.get_zone_at((x1 + x2) / 2, (z1 + z2) / 2)
            props = self.get_zone_properties(*zone)

            # Deterministic decay check
            decay_seed = int(x1 * 7919 + z1 * 6577 + x2 * 4993 + z2 * 3571 + self.world_seed * 9973)
            rng = random.Random(decay_seed)

            if rng.random() < props['decay_chance']:
                damage = rng.uniform(0.0, 0.5)
                self.pre_damaged_walls[key] = damage

                # Fully destroyed
                if damage < 0.2:
                    self.destroyed_walls.add(key)

        has_wall = True
        self.wall_cache[key] = has_wall
        return has_wall

    def is_wall_destroyed(self, wall_key):
        """Check if a wall has been destroyed."""
        return wall_key in self.destroyed_walls

    def get_wall_damage(self, wall_key):
        """Get damage state for a wall (1.0 = intact, 0.0 = rubble)."""
        return self.pre_damaged_walls.get(wall_key, 1.0)

    def get_doorway_type(self, x1, z1, x2, z2):
        """Determine if a wall has a doorway or hallway."""
        is_horizontal = (z1 == z2)

        if is_horizontal:
            door_seed = int(z1 * 3571 + ((x1 + x2) // 2) * 2897 + self.world_seed * 9973)
        else:
            door_seed = int(x1 * 3571 + ((z1 + z2) // 2) * 2897 + self.world_seed * 9973)

        rng = random.Random(door_seed)
        roll = rng.random()

        if roll < 0.3:
            return "hallway"
        elif roll < 0.5:
            return "doorway"
        else:
            return None

    # === PROGRESSIVE WALL DAMAGE ===

    def get_wall_state(self, wall_key):
        """Get current damage state of a wall."""
        if wall_key in self.destroyed_walls:
            return WallState.DESTROYED
        return self.wall_states.get(wall_key, WallState.INTACT)

    def get_wall_health(self, wall_key):
        """Get wall health (1.0 = full, 0.0 = destroyed)."""
        if wall_key in self.destroyed_walls:
            return 0.0
        return self.wall_health.get(wall_key, 1.0)

    def get_wall_cracks(self, wall_key):
        """Get crack data for rendering."""
        return self.wall_cracks.get(wall_key, [])

    def _get_wall_center(self, wall_key):
        """Get world position of wall center."""
        (x1, z1), (x2, z2) = wall_key
        h = get_scaled_wall_height()
        floor_y = get_scaled_floor_y()
        return (
            (x1 + x2) / 2,
            (floor_y + h) / 2,
            (z1 + z2) / 2
        )

    def _add_crack(self, wall_key, hit_u=None, hit_v=None):
        """Add a crack at hit position (or random if not specified)."""
        if wall_key not in self.wall_cracks:
            self.wall_cracks[wall_key] = []

        u = hit_u if hit_u is not None else random.uniform(0.1, 0.9)
        v = hit_v if hit_v is not None else random.uniform(0.1, 0.9)
        angle = random.uniform(0, math.pi)
        length = random.uniform(0.1, 0.4)

        self.wall_cracks[wall_key].append((u, v, angle, length))

    def _spawn_impact_debris(self, wall_key, count=50):
        """Spawn small debris burst on impact."""
        (x1, z1), (x2, z2) = wall_key
        h = get_scaled_wall_height()
        floor_y = get_scaled_floor_y()
        half_thick = WALL_THICKNESS / 2

        if x1 == x2:
            cx, cz = x1, (z1 + z2) / 2
        else:
            cx, cz = (x1 + x2) / 2, z1

        cy = (floor_y + h) / 2

        for _ in range(count):
            px = cx + random.uniform(-half_thick, half_thick)
            py = cy + random.uniform(-20, 20)
            pz = cz + random.uniform(-half_thick, half_thick)

            speed = random.uniform(3, 8)
            angle = random.uniform(0, 2 * math.pi)
            vx = math.cos(angle) * speed
            vy = random.uniform(2, 8)
            vz = math.sin(angle) * speed

            color_var = random.randint(-20, 20)
            color = (
                max(0, min(255, WALL_COLOR[0] + color_var)),
                max(0, min(255, WALL_COLOR[1] + color_var)),
                max(0, min(255, WALL_COLOR[2] + color_var))
            )

            self.debris_pieces.append(Debris(
                (px, py, pz), color, velocity=(vx, vy, vz)
            ))

    def hit_wall(self, wall_key, damage=0.25, hit_uv=None):
        """
        Apply damage to a wall. Returns True if wall was destroyed.
        
        damage: Amount of health to remove (0.25 = 4 hits to destroy)
        hit_uv: Optional (u, v) position of hit for crack placement
        """
        if wall_key in self.destroyed_walls:
            return False

        # Initialize if first hit
        if wall_key not in self.wall_health:
            self.wall_health[wall_key] = 1.0
            self.wall_states[wall_key] = WallState.INTACT

        old_state = self.wall_states[wall_key]
        old_health = self.wall_health[wall_key]

        # Apply damage
        self.wall_health[wall_key] = max(0.0, old_health - damage)
        new_health = self.wall_health[wall_key]

        # Determine new state based on health
        if new_health <= 0:
            new_state = WallState.DESTROYED
        elif new_health <= 0.25:
            new_state = WallState.FRACTURED
        elif new_health <= 0.6:
            new_state = WallState.CRACKED
        else:
            new_state = WallState.INTACT

        self.wall_states[wall_key] = new_state
        position = self._get_wall_center(wall_key)

        # Add cracks on damage
        if new_state in (WallState.CRACKED, WallState.FRACTURED):
            hit_u = hit_uv[0] if hit_uv else None
            hit_v = hit_uv[1] if hit_uv else None
            self._add_crack(wall_key, hit_u, hit_v)
            if new_state == WallState.FRACTURED:
                # Extra cracks when fractured
                self._add_crack(wall_key)
                self._add_crack(wall_key)

        # Emit events on state transitions
        if new_state != old_state:
            if new_state == WallState.CRACKED:
                self._spawn_impact_debris(wall_key, count=30)
                event_bus.emit(EventType.WALL_CRACKED,
                              wall_key=wall_key, position=position, health=new_health)

            elif new_state == WallState.FRACTURED:
                self._spawn_impact_debris(wall_key, count=60)
                event_bus.emit(EventType.WALL_FRACTURED,
                              wall_key=wall_key, position=position, health=new_health)

            elif new_state == WallState.DESTROYED:
                # Full destruction
                self._destroy_wall_internal(wall_key)
                event_bus.emit(EventType.WALL_DESTROYED,
                              wall_key=wall_key, position=position)
                return True

        elif old_state != WallState.INTACT:
            # Hit but didn't change state
            self._spawn_impact_debris(wall_key, count=20)
            event_bus.emit(EventType.WALL_HIT,
                          wall_key=wall_key, position=position, health=new_health)

        return False

    # === DESTRUCTION ===

    def _destroy_wall_internal(self, wall_key):
        """Internal wall destruction - spawns debris."""
        self.destroyed_walls.add(wall_key)
        self.wall_states[wall_key] = WallState.DESTROYED
        self.wall_health[wall_key] = 0.0

        (x1, z1), (x2, z2) = wall_key
        h = get_scaled_wall_height()
        floor_y = get_scaled_floor_y()
        half_thick = WALL_THICKNESS / 2

        if x1 == x2:
            x = x1
            min_x, max_x = x - half_thick, x + half_thick
            min_z, max_z = min(z1, z2), max(z1, z2)
        else:
            z = z1
            min_z, max_z = z - half_thick, z + half_thick
            min_x, max_x = min(x1, x2), max(x1, x2)

        min_y, max_y = floor_y, h

        base = 1200
        num_particles = max(250, int(base * (1.0 / (1.0 + len(self.destroyed_walls) / 20))))

        for _ in range(num_particles):
            px = random.uniform(min_x, max_x)
            py = random.uniform(min_y, max_y)
            pz = random.uniform(min_z, max_z)

            center_x = (min_x + max_x) / 2
            center_z = (min_z + max_z) / 2

            dx = px - center_x
            dz = pz - center_z
            dist = math.sqrt(dx ** 2 + dz ** 2) + 0.1

            speed = random.uniform(8, 20)
            vx = (dx / dist) * speed + random.uniform(-3, 3)
            vy = random.uniform(-20, -5)
            vz = (dz / dist) * speed + random.uniform(-3, 3)

            color_var = random.randint(-30, 30)
            particle_color = (
                max(0, min(255, WALL_COLOR[0] + color_var)),
                max(0, min(255, WALL_COLOR[1] + color_var)),
                max(0, min(255, WALL_COLOR[2] + color_var))
            )

            self.debris_pieces.append(Debris(
                (px, py, pz),
                particle_color,
                velocity=(vx, vy, vz)
            ))

    def destroy_wall(self, wall_key, destroy_sound=None):
        """Instantly destroy a wall (bypasses progressive damage)."""
        if wall_key in self.destroyed_walls:
            return

        if destroy_sound:
            destroy_sound.play()

        position = self._get_wall_center(wall_key)
        self._destroy_wall_internal(wall_key)
        event_bus.emit(EventType.WALL_DESTROYED,
                      wall_key=wall_key, position=position)

    def destroy_pillar(self, pillar_key, destroy_sound=None):
        """Destroy a pillar and create debris."""
        if pillar_key in self.destroyed_pillars:
            return

        self.destroyed_pillars.add(pillar_key)

        if destroy_sound:
            destroy_sound.play()

        pillar_x, pillar_z = pillar_key
        s = PILLAR_SIZE
        h = get_scaled_wall_height()
        floor_y = get_scaled_floor_y()

        min_x, max_x = pillar_x, pillar_x + s
        min_z, max_z = pillar_z, pillar_z + s
        min_y, max_y = floor_y, h

        # Calculate center for event
        position = (
            (min_x + max_x) / 2,
            (min_y + max_y) / 2,
            (min_z + max_z) / 2
        )

        base = 1200
        num_particles = max(250, int(base * (1.0 / (1.0 + len(self.destroyed_pillars) / 20))))

        for _ in range(num_particles):
            px = random.uniform(min_x, max_x)
            py = random.uniform(min_y, max_y)
            pz = random.uniform(min_z, max_z)

            center_x = (min_x + max_x) / 2
            center_z = (min_z + max_z) / 2

            dx = px - center_x
            dz = pz - center_z
            dist = math.sqrt(dx ** 2 + dz ** 2) + 0.1

            speed = random.uniform(8, 20)
            vx = (dx / dist) * speed + random.uniform(-3, 3)
            vy = random.uniform(-20, -5)
            vz = (dz / dist) * speed + random.uniform(-3, 3)

            color_var = random.randint(-30, 30)
            particle_color = (
                max(0, min(255, PILLAR_COLOR[0] + color_var)),
                max(0, min(255, PILLAR_COLOR[1] + color_var)),
                max(0, min(255, PILLAR_COLOR[2] + color_var))
            )

            self.debris_pieces.append(Debris(
                (px, py, pz),
                particle_color,
                velocity=(vx, vy, vz)
            ))

        # Emit destruction event
        event_bus.emit(EventType.PILLAR_DESTROYED,
                      pillar_key=pillar_key, position=position)

    def spawn_rubble_pile(self, x1, z1, x2, z2):
        """Spawn a persistent rubble pile for pre-destroyed walls."""
        wall_key = tuple(sorted([(x1, z1), (x2, z2)]))

        if wall_key in self._spawned_rubble:
            return

        self._spawned_rubble.add(wall_key)

        floor_y = get_scaled_floor_y()
        half_thick = WALL_THICKNESS / 2

        if x1 == x2:
            min_x, max_x = x1 - half_thick, x1 + half_thick
            min_z, max_z = min(z1, z2), max(z1, z2)
        else:
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_z, max_z = z1 - half_thick, z1 + half_thick

        # Spawn settled debris
        for _ in range(80):
            px = random.uniform(min_x, max_x)
            pz = random.uniform(min_z, max_z)

            color_var = random.randint(-40, 20)
            particle_color = (
                max(0, min(255, 200 + color_var)),
                max(0, min(255, 180 + color_var)),
                max(0, min(255, 160 + color_var))
            )

            self.debris_pieces.append(Debris(
                (px, floor_y, pz),
                particle_color,
                velocity=None  # Settled from the start
            ))

    # === DEBRIS UPDATE ===

    def update_debris(self, dt, player_x, player_z):
        """Update all debris particles."""
        floor_y = get_scaled_floor_y()
        MAX_DEBRIS = 12000
        DEBRIS_CULL_DIST = 900.0

        for d in self.debris_pieces:
            d.update(dt, floor_y)
            if not d.active:
                continue

            # Cull distant debris
            dx = d.cx - player_x
            dz = d.cz - player_z
            if (dx * dx + dz * dz) > (DEBRIS_CULL_DIST * DEBRIS_CULL_DIST):
                d.active = False

        # Remove inactive debris
        self.debris_pieces = [d for d in self.debris_pieces if d.active]

        # Enforce hard cap
        if len(self.debris_pieces) > MAX_DEBRIS:
            self.debris_pieces = self.debris_pieces[-MAX_DEBRIS:]

    # === COLLISION QUERIES ===

    def check_collision(self, x, z):
        """Check if a position collides with walls."""
        if not math.isfinite(x) or not math.isfinite(z):
            return True

        player_radius = 15.0
        half_thick = WALL_THICKNESS / 2
        # MEGA-ARCHITECTURE: Use larger check range for mega-scale zones
        room_size = get_room_size_at_position(x, z, self)
        check_range = room_size * 2  # Check 2 grid cells in each direction
        
        # Calculate grid cell range to check
        min_grid_x = int((x - check_range) // PILLAR_SPACING) * PILLAR_SPACING
        max_grid_x = int((x + check_range) // PILLAR_SPACING) * PILLAR_SPACING
        min_grid_z = int((z - check_range) // PILLAR_SPACING) * PILLAR_SPACING
        max_grid_z = int((z + check_range) // PILLAR_SPACING) * PILLAR_SPACING

        for px_grid in range(min_grid_x, max_grid_x + PILLAR_SPACING, PILLAR_SPACING):
            for pz_grid in range(min_grid_z, max_grid_z + PILLAR_SPACING, PILLAR_SPACING):

                # Check horizontal wall
                if self.has_wall_between(px_grid, pz_grid, px_grid + PILLAR_SPACING, pz_grid):
                    wall_key = tuple(sorted([(px_grid, pz_grid), (px_grid + PILLAR_SPACING, pz_grid)]))
                    if wall_key in self.destroyed_walls:
                        continue

                    opening_type = self.get_doorway_type(px_grid, pz_grid, px_grid + PILLAR_SPACING, pz_grid)
                    wall_z = pz_grid
                    wall_x_start = px_grid
                    wall_x_end = px_grid + PILLAR_SPACING

                    if opening_type == "hallway":
                        opening_width = HALLWAY_WIDTH
                    elif opening_type == "doorway":
                        opening_width = 60
                    else:
                        opening_width = 0

                    if opening_width > 0:
                        opening_start = wall_x_start + (PILLAR_SPACING - opening_width) / 2
                        opening_end = opening_start + opening_width

                        if abs(z - wall_z) < (half_thick + player_radius):
                            if (wall_x_start <= x <= opening_start - player_radius) or \
                                    (opening_end + player_radius <= x <= wall_x_end):
                                return True
                    else:
                        if (wall_x_start - player_radius <= x <= wall_x_end + player_radius and
                                abs(z - wall_z) < (half_thick + player_radius)):
                            return True

                # Check vertical wall
                if self.has_wall_between(px_grid, pz_grid, px_grid, pz_grid + PILLAR_SPACING):
                    wall_key = tuple(sorted([(px_grid, pz_grid), (px_grid, pz_grid + PILLAR_SPACING)]))
                    if wall_key in self.destroyed_walls:
                        continue

                    opening_type = self.get_doorway_type(px_grid, pz_grid, px_grid, pz_grid + PILLAR_SPACING)
                    wall_x = px_grid
                    wall_z_start = pz_grid
                    wall_z_end = pz_grid + PILLAR_SPACING

                    if opening_type == "hallway":
                        opening_width = HALLWAY_WIDTH
                    elif opening_type == "doorway":
                        opening_width = 60
                    else:
                        opening_width = 0

                    if opening_width > 0:
                        opening_start = wall_z_start + (PILLAR_SPACING - opening_width) / 2
                        opening_end = opening_start + opening_width

                        if abs(x - wall_x) < (half_thick + player_radius):
                            if (wall_z_start <= z <= opening_start - player_radius) or \
                                    (opening_end + player_radius <= z <= wall_z_end):
                                return True
                    else:
                        if (wall_z_start - player_radius <= z <= wall_z_end + player_radius and
                                abs(x - wall_x) < (half_thick + player_radius)):
                            return True

                # Check pillar collision
                offset = PILLAR_SPACING // 2
                pillar_x = px_grid + offset
                pillar_z = pz_grid + offset
                pillar_key = (pillar_x, pillar_z)
                
                if self.has_pillar_at(pillar_x, pillar_z) and pillar_key not in self.destroyed_pillars:
                    # Pillar is a box from (pillar_x, pillar_z) to (pillar_x + PILLAR_SIZE, pillar_z + PILLAR_SIZE)
                    pillar_min_x = pillar_x
                    pillar_max_x = pillar_x + PILLAR_SIZE
                    pillar_min_z = pillar_z
                    pillar_max_z = pillar_z + PILLAR_SIZE

                    # Box vs circle collision
                    closest_x = max(pillar_min_x, min(x, pillar_max_x))
                    closest_z = max(pillar_min_z, min(z, pillar_max_z))

                    dist_x = x - closest_x
                    dist_z = z - closest_z
                    dist_sq = dist_x * dist_x + dist_z * dist_z

                    if dist_sq < player_radius * player_radius:
                        return True

        return False

    # === SAVE/LOAD ===

    def get_state_for_save(self):
        """Get world state for saving."""
        return {
            'seed': self.world_seed,
            'destroyed_walls': [list(wall) for wall in self.destroyed_walls]
        }

    def load_state(self, data):
        """Load world state from save data."""
        self.world_seed = data.get('seed', self.world_seed)

        destroyed_walls_list = data.get('destroyed_walls', [])
        self.destroyed_walls = {tuple(tuple(point) for point in wall) for wall in destroyed_walls_list}
        
        # Load wall damage/health states
        if 'wall_damage' in data:
            self.wall_health = {}
            for key_str, damage in data['wall_damage'].items():
                # Convert "x,y,z" string back to tuple
                coords = tuple(map(int, key_str.split(',')))
                # Convert damage (0.0-1.0) to health (1.0-0.0)
                self.wall_health[coords] = 1.0 - damage
        
        # Load destroyed pillars
        if 'destroyed_pillars' in data:
            destroyed_pillars_list = data.get('destroyed_pillars', [])
            self.destroyed_pillars = set(tuple(pillar) for pillar in destroyed_pillars_list)

        # Clear caches
        self.pillar_cache.clear()
        self.wall_cache.clear()
        self.zone_cache.clear()

        print(f"Loaded world with seed: {self.world_seed}")
        print(f"Loaded {len(self.destroyed_walls)} destroyed walls")
        print(f"Loaded {len(self.wall_health)} damaged walls")
        print(f"Loaded {len(self.destroyed_pillars)} destroyed pillars")
