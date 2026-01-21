"""
Ceiling Light Audio Sources
============================
Extends your existing AcousticIntegration to add ceiling lights as audio emitters.
Lights emit electrical buzz that raytraces to the player.
"""

import math
import numpy as np
from ceiling_heights import get_ceiling_height_at_position


class LightSource:
    """Single ceiling light that emits audio."""
    
    def __init__(self, grid_x, grid_z, world):
        self.grid_x = grid_x
        self.grid_z = grid_z
        self.world = world
        
        # Position (at ceiling)
        self.x = float(grid_x)
        self.z = float(grid_z)
        self.y = get_ceiling_height_at_position(self.x, self.z, world)
        
        # Audio properties
        self.base_volume = 0.15  # Quiet buzz
        self.frequency = 120  # Hz
        self.max_distance = 800  # Audible range
    
    def get_position(self):
        return (self.x, self.y, self.z)
    
    def get_distance_to(self, px, py, pz):
        dx = self.x - px
        dy = self.y - py
        dz = self.z - pz
        return math.sqrt(dx*dx + dy*dy + dz*dz)


class LightAudioManager:
    """
    Manages ceiling lights as audio sources.
    Integrates with your existing AcousticIntegration system.
    """
    
    def __init__(self, world):
        self.world = world
        self.lights = {}  # (grid_x, grid_z) -> LightSource
        
        # Settings
        self.light_spacing = 400  # One per room
        self.generation_radius = 1200
        
    def update_lights(self, player_x, player_z):
        """Generate/remove lights based on player position."""
        player_grid_x = int(player_x // self.light_spacing) * self.light_spacing
        player_grid_z = int(player_z // self.light_spacing) * self.light_spacing
        
        grid_radius = int(math.ceil(self.generation_radius / self.light_spacing))
        
        # Generate lights in range
        new_lights = set()
        for gx_offset in range(-grid_radius, grid_radius + 1):
            for gz_offset in range(-grid_radius, grid_radius + 1):
                grid_x = player_grid_x + (gx_offset * self.light_spacing)
                grid_z = player_grid_z + (gz_offset * self.light_spacing)
                
                dx = grid_x - player_x
                dz = grid_z - player_z
                if math.sqrt(dx*dx + dz*dz) <= self.generation_radius:
                    key = (grid_x, grid_z)
                    new_lights.add(key)
                    
                    if key not in self.lights:
                        self.lights[key] = LightSource(grid_x, grid_z, self.world)
        
        # Remove far lights
        self.lights = {k: v for k, v in self.lights.items() if k in new_lights}
    
    def get_audible_lights(self, player_x, player_y, player_z):
        """Get lights within audible range, sorted by distance."""
        audible = []
        
        for light in self.lights.values():
            dist = light.get_distance_to(player_x, player_y, player_z)
            if dist < light.max_distance:
                audible.append((light, dist))
        
        audible.sort(key=lambda x: x[1])
        return audible


def trace_light_to_player(raycaster, light, player_pos):
    """
    Trace sound from a ceiling light to player using existing raycaster.
    Returns contribution dict for audio mixing.
    """
    light_pos = light.get_position()
    px, py, pz = player_pos
    
    # Calculate direction: light → player
    dx = px - light_pos[0]
    dy = py - light_pos[1]
    dz = pz - light_pos[2]
    distance = math.sqrt(dx*dx + dy*dy + dz*dz)
    
    if distance < 0.001:
        return None
    
    direction = np.array([dx/distance, dy/distance, dz/distance])
    
    # Use raycaster to check occlusion
    hit_info = raycaster._cast_single_ray(np.array(light_pos), direction)
    
    # Check if ray hit player (approximately)
    direct_path_clear = False
    if hit_info:
        hit_dist, hit_pos = hit_info
        # If hit distance is close to player distance, path is clear
        if abs(hit_dist - distance) < 20:  # Within 20 units
            direct_path_clear = True
    else:
        # No hit means clear path
        direct_path_clear = True
    
    # Calculate volume based on distance and occlusion
    distance_factor = max(0, 1.0 - (distance / light.max_distance))
    volume = light.base_volume * (distance_factor ** 2)
    
    if not direct_path_clear:
        volume *= 0.15  # Heavy attenuation through walls
    
    return {
        'light': light,
        'distance': distance,
        'volume': volume,
        'direct_path': direct_path_clear,
        'position': light_pos,
    }


def integrate_light_audio(acoustic_integration, player_pos, dt):
    """
    Main integration function - call this from your update loop.
    Adds light buzz to the acoustic system.
    
    Args:
        acoustic_integration: Your AcousticIntegration instance
        player_pos: (x, y, z) tuple
        dt: delta time
    
    Returns:
        dict with light audio info for visualization
    """
    px, py, pz = player_pos
    
    # Get the light manager (create if doesn't exist)
    if not hasattr(acoustic_integration, 'light_manager'):
        acoustic_integration.light_manager = LightAudioManager(acoustic_integration.engine.world)
    
    light_mgr = acoustic_integration.light_manager
    
    # Update which lights exist
    light_mgr.update_lights(px, pz)
    
    # Get audible lights
    audible_lights = light_mgr.get_audible_lights(px, py, pz)
    
    # Trace each light to player
    light_contributions = []
    total_buzz_volume = 0.0
    
    for light, dist in audible_lights[:8]:  # Limit to 8 closest lights
        contrib = trace_light_to_player(acoustic_integration.raycaster, light, player_pos)
        if contrib and contrib['volume'] > 0.01:
            light_contributions.append(contrib)
            total_buzz_volume += contrib['volume']
    
    # Clamp total
    total_buzz_volume = min(1.0, total_buzz_volume)
    
    return {
        'contributions': light_contributions,
        'total_volume': total_buzz_volume,
        'num_lights': len(light_contributions),
    }


# ============================================================
# VISUALIZATION
# ============================================================

def render_light_audio_viz(screen, camera, light_audio_result, fade=1.0):
    """
    Render visualization of light audio rays.
    Draw lines from lights → player, colored by volume.
    
    Args:
        screen: pygame surface
        camera: camera object
        light_audio_result: result from integrate_light_audio()
        fade: 0.0-1.0 alpha multiplier
    """
    import pygame
    
    if fade <= 0:
        return
    
    player_pos = (camera.x_s, camera.y_s, camera.z_s)
    
    # Transform player to screen
    player_cam = camera.world_to_camera(*player_pos)
    player_screen = camera.project(player_cam)
    if not player_screen:
        return
    
    # Draw each light contribution
    for contrib in light_audio_result['contributions']:
        light_pos = contrib['position']
        volume = contrib['volume']
        direct = contrib['direct_path']
        
        # Transform light to screen
        light_cam = camera.world_to_camera(*light_pos)
        light_screen = camera.project(light_cam)
        if not light_screen:
            continue
        
        # Color: green for direct, red for occluded
        if direct:
            base_color = (100, 255, 100)  # Green
        else:
            base_color = (255, 100, 100)  # Red
        
        # Alpha based on volume and fade
        alpha = int(volume * fade * 255)
        color = (*base_color, alpha)
        
        # Draw line
        try:
            start_pos = (int(light_screen[0]), int(light_screen[1]))
            end_pos = (int(player_screen[0]), int(player_screen[1]))
            
            temp_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            pygame.draw.line(temp_surface, color, start_pos, end_pos, 2)
            
            # Draw small circle at light position
            pygame.draw.circle(temp_surface, color, start_pos, 4)
            
            screen.blit(temp_surface, (0, 0))
        except:
            pass


# This module is ready to use - no editing required!
# Just import it in your main.py and it integrates automatically.
