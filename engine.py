"""
Backrooms Engine - Orchestrator.
Owns subsystems and coordinates updates/rendering.
No implementation details - just glue.
"""

import math
import random
import pygame

from config import FOOTSTEP_INTERVAL, BUZZ_INTERVAL
from camera import Camera
from player import Player
from world import World
from renderer import Renderer
from targeting import find_targeted_wall_or_pillar
from events import event_bus, EventType
from collision import CollisionSystem
from smile_entity import SmileManager


class BackroomsEngine:
    """Main engine orchestrator."""

    def __init__(self, width, height, world_seed=None):
        self.width = width
        self.height = height

        # Create subsystems
        self.world = World(world_seed)
        self.player = Player()
        self.camera = Camera(width, height)
        self.renderer = Renderer(width, height)
        
        # Create and connect collision system
        self.collision_system = CollisionSystem(self.world, player_radius=15.0)
        self.player.collision_system = self.collision_system
        
        # Create drawing system
        from drawing_system import WallDrawing
        self.drawing_system = WallDrawing()
        
        # Create smile entity manager
        self.smile_manager = SmileManager()
        # Spawn initial smile near player start
        self.smile_manager.add_spawn_zone(200, 200, count=1)

        # Sound timers
        self.sound_timer = 0
        self.next_footstep = random.uniform(*FOOTSTEP_INTERVAL)
        self.next_buzz = random.uniform(*BUZZ_INTERVAL)
        self.last_footstep_phase = 0

        # Atmosphere effects
        self.screen_shake_intensity = 0.0
        self.screen_shake_decay = 5.0

        # Stats
        self.play_time = 0
        
        # Collision sound
        self.collision_sound = None  # Set by main after sound generation

        # Subscribe to events for atmosphere effects
        self._setup_event_handlers()

    # === PROPERTIES FOR COMPATIBILITY ===

    @property
    def mouse_look(self):
        return self.player.mouse_look

    @mouse_look.setter
    def mouse_look(self, value):
        self.player.mouse_look = value

    @property
    def x(self):
        return self.player.x

    @property
    def y(self):
        return self.player.y

    @property
    def z(self):
        return self.player.z

    @property
    def pitch(self):
        return self.player.pitch

    @property
    def yaw(self):
        return self.player.yaw

    @property
    def world_seed(self):
        return self.world.world_seed

    @property
    def destroyed_walls(self):
        return self.world.destroyed_walls
    
    @property
    def wall_damage(self):
        """
        Access wall damage states from world.
        Converts wall_health (1.0 = healthy, 0.0 = destroyed) 
        to wall_damage (0.0 = healthy, 1.0 = destroyed).
        """
        # Return damage as inverted health
        return {k: 1.0 - v for k, v in self.world.wall_health.items()}
    
    @property
    def destroyed_pillars(self):
        """Access destroyed pillars from world."""
        return self.world.destroyed_pillars

    @property
    def x_s(self):
        return self.camera.x_s

    @property
    def y_s(self):
        return self.camera.y_s

    @property
    def z_s(self):
        return self.camera.z_s

    # === MAIN UPDATE ===

    def update(self, dt, keys, mouse_rel):
        """Main update loop."""
        self.play_time += dt

        # Update player with collision callback
        def collision_callback(intensity):
            """Called when player collides with something."""
            if self.collision_sound:
                # Volume based on intensity
                self.collision_sound.set_volume(0.2 + intensity * 0.3)
                self.collision_sound.play()
        
        self.player.collision_callback = collision_callback
        self.player.update(dt, keys, mouse_rel, self.world.check_collision, self.play_time)

        # Update camera to follow player
        self.camera.update(dt, self.player)

        # Apply screen shake to camera
        if self.screen_shake_intensity > 0.01:
            shake_x = random.uniform(-1, 1) * self.screen_shake_intensity * 3
            shake_y = random.uniform(-1, 1) * self.screen_shake_intensity * 3
            self.camera.x_s += shake_x
            self.camera.y_s += shake_y
            self.screen_shake_intensity *= (1.0 - self.screen_shake_decay * dt)
        else:
            self.screen_shake_intensity = 0

        # Update debris
        self.world.update_debris(dt, self.camera.x_s, self.camera.z_s)
        
        # Update smile entities
        player_pos = (self.player.x, self.player.z)
        self.smile_manager.update(dt, player_pos, self.collision_system, self.camera)

        # Process any queued events
        event_bus.process_queue()

    # === EVENT HANDLERS ===

    def _setup_event_handlers(self):
        """Subscribe to events for atmosphere effects."""
        event_bus.subscribe(EventType.WALL_DESTROYED, self._on_wall_destroyed)
        event_bus.subscribe(EventType.WALL_CRACKED, self._on_wall_cracked)
        event_bus.subscribe(EventType.WALL_FRACTURED, self._on_wall_fractured)
        event_bus.subscribe(EventType.PILLAR_DESTROYED, self._on_pillar_destroyed)

    def _on_wall_destroyed(self, event):
        """Handle wall destruction - big shake + sound."""
        dist = self._distance_to_event(event.position)
        if dist < 500:
            intensity = max(0, 1.0 - dist / 500) * 0.8
            self.screen_shake_intensity = max(self.screen_shake_intensity, intensity)
            # Play destroy sound
            if hasattr(self, 'sound_effects') and 'destroy' in self.sound_effects:
                vol = max(0.2, 1.0 - dist / 500)
                self.sound_effects['destroy'].set_volume(vol)
                self.sound_effects['destroy'].play()

    def _on_wall_cracked(self, event):
        """Handle wall cracking - small shake + crack sound."""
        dist = self._distance_to_event(event.position)
        if dist < 300:
            intensity = max(0, 1.0 - dist / 300) * 0.15
            self.screen_shake_intensity = max(self.screen_shake_intensity, intensity)
            # Play crack sound
            if hasattr(self, 'sound_effects') and 'crack' in self.sound_effects:
                vol = max(0.3, 1.0 - dist / 300)
                self.sound_effects['crack'].set_volume(vol)
                self.sound_effects['crack'].play()

    def _on_wall_fractured(self, event):
        """Handle wall fracturing - medium shake + fracture sound."""
        dist = self._distance_to_event(event.position)
        if dist < 400:
            intensity = max(0, 1.0 - dist / 400) * 0.35
            self.screen_shake_intensity = max(self.screen_shake_intensity, intensity)
            # Play fracture sound
            if hasattr(self, 'sound_effects') and 'fracture' in self.sound_effects:
                vol = max(0.3, 1.0 - dist / 400)
                self.sound_effects['fracture'].set_volume(vol)
                self.sound_effects['fracture'].play()

    def _on_pillar_destroyed(self, event):
        """Handle pillar destruction - big shake + sound."""
        dist = self._distance_to_event(event.position)
        if dist < 500:
            intensity = max(0, 1.0 - dist / 500) * 1.0
            self.screen_shake_intensity = max(self.screen_shake_intensity, intensity)
            if hasattr(self, 'sound_effects') and 'destroy' in self.sound_effects:
                vol = max(0.2, 1.0 - dist / 500)
                self.sound_effects['destroy'].set_volume(vol)
                self.sound_effects['destroy'].play()

    def _distance_to_event(self, position):
        """Calculate distance from camera to event position."""
        dx = position[0] - self.camera.x_s
        dy = position[1] - self.camera.y_s
        dz = position[2] - self.camera.z_s
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    # === RENDERING ===

    def render(self, surface):
        """Main render."""
        self.renderer.render(surface, self.camera, self.world, self.player, self.drawing_system)
        
        # Render smile entities on top (with occlusion checking)
        self.smile_manager.render(surface, self.camera, self.world)

    def update_render_scale(self, dt):
        """Update render scale transition."""
        self.renderer.update_render_scale(dt)

    def toggle_render_scale(self):
        """Toggle performance mode."""
        self.renderer.toggle_render_scale()

    def update_flicker(self, dt):
        """Update light flicker effect."""
        self.renderer.update_flicker(dt)

    # === TARGETING & DESTRUCTION ===

    def find_targeted_wall_or_pillar(self):
        """Find what the player is looking at."""
        return find_targeted_wall_or_pillar(self.camera, self.world)

    def hit_wall(self, wall_key, damage=0.25):
        """
        Apply damage to a wall (progressive destruction).
        Returns True if wall was destroyed.
        """
        return self.world.hit_wall(wall_key, damage)

    def destroy_wall(self, wall_key, destroy_sound):
        """Instantly destroy a wall (bypasses progressive damage)."""
        self.world.destroy_wall(wall_key, destroy_sound)

    def destroy_pillar(self, pillar_key, destroy_sound):
        """Destroy a pillar."""
        self.world.destroy_pillar(pillar_key, destroy_sound)

    # === SOUND SYSTEM ===

    def update_sounds(self, dt, sound_effects):
        """Update ambient sounds."""
        self.sound_timer += dt

        if self.sound_timer >= self.next_footstep:
            angle = random.uniform(0, 2 * math.pi)
            self._play_directional_sound(sound_effects['footstep'], angle)
            self.next_footstep = self.sound_timer + random.uniform(*FOOTSTEP_INTERVAL)

        if self.sound_timer >= self.next_buzz:
            angle = random.uniform(0, 2 * math.pi)
            self._play_directional_sound(sound_effects['buzz'], angle)
            self.next_buzz = self.sound_timer + random.uniform(*BUZZ_INTERVAL)

    def _play_directional_sound(self, sound, world_angle):
        """Play a sound with stereo panning."""
        angle_diff = world_angle - self.camera.yaw_s

        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        pan = 0.5 + (angle_diff / math.pi) * 0.5
        pan = max(0.0, min(1.0, pan))

        channel = sound.play()
        if channel:
            left_volume = 1.0 - pan
            right_volume = pan
            avg_volume = 0.7
            channel.set_volume(avg_volume * left_volume, avg_volume * right_volume)

    def update_player_footsteps(self, dt, footstep_sound, crouch_footstep_sound):
        """Update player footstep sounds synced to animation."""
        if self.player.is_moving:
            current_phase = self.camera.head_bob_time % 1.0
            if ((self.last_footstep_phase > 0.5 and current_phase < 0.5) or
                    (self.last_footstep_phase > current_phase and current_phase < 0.1)):
                if self.player.is_crouching:
                    crouch_footstep_sound.play()
                else:
                    footstep_sound.play()
            self.last_footstep_phase = current_phase
        else:
            self.last_footstep_phase = 0

    # === INPUT ===

    def toggle_mouse(self):
        """Toggle mouse look."""
        self.player.toggle_mouse()

    # === SAVE/LOAD ===

    def load_from_save(self, save_data):
        """Load game state from save data."""
        if save_data:
            player_data = save_data.get('player', {})
            self.player.load_state(player_data)

            # Sync camera to player immediately
            self.camera.x_s = self.player.x
            self.camera.y_s = self.player.y
            self.camera.z_s = self.player.z
            self.camera.pitch_s = self.player.pitch
            self.camera.yaw_s = self.player.yaw

            world_data = save_data.get('world', {})
            self.world.load_state(world_data)
            
            # Load drawings
            drawing_data = save_data.get('drawings', {})
            self.drawing_system.load_state(drawing_data)
            
            # Load smile entities
            smiles_data = save_data.get('smiles', {})
            self.smile_manager.load_state(smiles_data)

            stats = save_data.get('stats', {})
            self.play_time = stats.get('play_time', 0)

    # === COMPATIBILITY METHODS ===
    # These exist for backward compatibility with save_system.py

    def world_to_camera(self, x, y, z):
        """Transform world to camera space (for debris rendering)."""
        return self.camera.world_to_camera(x, y, z)

    def project_camera(self, p):
        """Project camera space to screen (for debris rendering)."""
        return self.camera.project(p)
