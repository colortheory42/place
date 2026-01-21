"""
Acoustic System Integration
============================
Integrates acoustic raycasting into the Backrooms engine.
Adds keyboard controls and visualization.
"""

import pygame
import numpy as np
import math
import threading
import queue
from audio_backends import MicrophoneCapture
from raycasting import ray_intersects_triangle
from config import PILLAR_SPACING, PILLAR_SIZE, WALL_THICKNESS, get_scaled_wall_height, get_scaled_floor_y


class AcousticRaycaster:
    """
    Casts rays from player position to simulate sound propagation.
    Uses collision detection to calculate reverb and occlusion.
    """
    
    def __init__(self, engine):
        self.engine = engine
        self.num_rays = 32  # 32 rays in a sphere around the player
        self.max_ray_distance = float('inf')  # Maximum distance to trace
        
        # Ray results cache
        self.ray_distances = []
        self.ray_hits = []
        
        # Reverb calculation
        self.reverb_amount = 0.0
        self.room_size_estimate = 0.0
        
    def cast_rays(self, player_pos):
        """
        Cast rays in all directions from player position.
        Returns reverb parameters based on environment.
        """
        self.ray_distances = []
        self.ray_hits = []
        
        origin = np.array(player_pos)
        
        # Cast rays in a sphere pattern (Fibonacci sphere)
        for i in range(self.num_rays):
            # Fibonacci sphere distribution for even ray spacing
            phi = math.acos(1 - 2 * (i + 0.5) / self.num_rays)
            theta = math.pi * (1 + 5**0.5) * (i + 0.5)
            
            # Convert to Cartesian
            direction = np.array([
                math.sin(phi) * math.cos(theta),
                math.sin(phi) * math.sin(theta),
                math.cos(phi)
            ])
            
            # Cast ray and find nearest hit
            hit_info = self._cast_single_ray(origin, direction)
            
            if hit_info:
                distance, hit_pos = hit_info
                self.ray_distances.append(distance)
                self.ray_hits.append(hit_pos)
            else:
                self.ray_distances.append(self.max_ray_distance)
                self.ray_hits.append(None)
        
        # Calculate reverb based on ray distances
        self._calculate_reverb()
        
        return {
            'reverb_amount': self.reverb_amount,
            'room_size': self.room_size_estimate,
            'avg_distance': np.mean(self.ray_distances) if self.ray_distances else 0
        }
    
    def _cast_single_ray(self, origin, direction):
        """
        Cast a single ray and find the nearest wall/pillar intersection.
        Returns (distance, hit_position) or None.
        """
        closest_hit = None
        closest_dist = self.max_ray_distance
        
        # Get player position for range calculation
        px_s = self.engine.camera.x_s
        pz_s = self.engine.camera.z_s
        
        # Check area around player
        render_range = 250
        start_x = int((px_s - render_range) // PILLAR_SPACING) * PILLAR_SPACING
        end_x = int((px_s + render_range) // PILLAR_SPACING) * PILLAR_SPACING
        start_z = int((pz_s - render_range) // PILLAR_SPACING) * PILLAR_SPACING
        end_z = int((pz_s + render_range) // PILLAR_SPACING) * PILLAR_SPACING
        
        h = get_scaled_wall_height()
        floor_y = get_scaled_floor_y()
        
        # Check walls
        for px in range(start_x, end_x + PILLAR_SPACING, PILLAR_SPACING):
            for pz in range(start_z, end_z + PILLAR_SPACING, PILLAR_SPACING):
                # Horizontal walls
                if self.engine.world.has_wall_between(px, pz, px + PILLAR_SPACING, pz):
                    wall_key = tuple(sorted([(px, pz), (px + PILLAR_SPACING, pz)]))
                    if not self.engine.world.is_wall_destroyed(wall_key):
                        hit_dist, hit_pos = self._ray_wall_intersection(
                            origin, direction, px, pz, px + PILLAR_SPACING, pz, h, floor_y, 'horizontal'
                        )
                        if hit_dist and hit_dist < closest_dist:
                            closest_dist = hit_dist
                            closest_hit = hit_pos
                
                # Vertical walls
                if self.engine.world.has_wall_between(px, pz, px, pz + PILLAR_SPACING):
                    wall_key = tuple(sorted([(px, pz), (px, pz + PILLAR_SPACING)]))
                    if not self.engine.world.is_wall_destroyed(wall_key):
                        hit_dist, hit_pos = self._ray_wall_intersection(
                            origin, direction, px, pz, px, pz + PILLAR_SPACING, h, floor_y, 'vertical'
                        )
                        if hit_dist and hit_dist < closest_dist:
                            closest_dist = hit_dist
                            closest_hit = hit_pos
        
        # Check pillars
        offset = PILLAR_SPACING // 2
        for px in range(start_x, end_x + PILLAR_SPACING, PILLAR_SPACING):
            for pz in range(start_z, end_z + PILLAR_SPACING, PILLAR_SPACING):
                pillar_x = px + offset
                pillar_z = pz + offset
                
                if self.engine.world.has_pillar_at(pillar_x, pillar_z):
                    pillar_key = (pillar_x, pillar_z)
                    if not self.engine.world.is_pillar_destroyed(pillar_key):
                        hit_dist, hit_pos = self._ray_pillar_intersection(
                            origin, direction, pillar_x, pillar_z, h, floor_y
                        )
                        if hit_dist and hit_dist < closest_dist:
                            closest_dist = hit_dist
                            closest_hit = hit_pos
        
        # Check floor and ceiling
        floor_hit = self._ray_plane_intersection(origin, direction, floor_y, 'floor')
        if floor_hit and floor_hit[0] < closest_dist:
            closest_dist = floor_hit[0]
            closest_hit = floor_hit[1]
        
        ceiling_hit = self._ray_plane_intersection(origin, direction, h, 'ceiling')
        if ceiling_hit and ceiling_hit[0] < closest_dist:
            closest_dist = ceiling_hit[0]
            closest_hit = ceiling_hit[1]
        
        if closest_hit is not None:
            return (closest_dist, closest_hit)
        return None
    
    def _ray_wall_intersection(self, origin, direction, x1, z1, x2, z2, h, floor_y, wall_type):
        """Check ray intersection with a wall."""
        half_thick = WALL_THICKNESS / 2
        
        if wall_type == 'horizontal':
            z = z1
            # Front face
            v0 = (x1, h, z - half_thick)
            v1 = (x2, h, z - half_thick)
            v2 = (x2, floor_y, z - half_thick)
            v3 = (x1, floor_y, z - half_thick)
        else:  # vertical
            x = x1
            # Front face
            v0 = (x - half_thick, h, z1)
            v1 = (x - half_thick, h, z2)
            v2 = (x - half_thick, floor_y, z2)
            v3 = (x - half_thick, floor_y, z1)
        
        # Test both triangles
        for tri in [(v0, v1, v2), (v0, v2, v3)]:
            hit = ray_intersects_triangle(origin, direction, *tri)
            if hit and hit[0] > 0 and hit[0] < self.max_ray_distance:
                t = hit[0]
                hit_pos = origin + direction * t
                return (t, hit_pos)
        
        return (None, None)
    
    def _ray_pillar_intersection(self, origin, direction, pillar_x, pillar_z, h, floor_y):
        """Check ray intersection with a pillar."""
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
        
        closest_dist = self.max_ray_distance
        closest_hit = None
        
        for face in faces:
            v0, v1, v2, v3 = face
            for tri in [(v0, v1, v2), (v0, v2, v3)]:
                hit = ray_intersects_triangle(origin, direction, *tri)
                if hit and hit[0] > 0 and hit[0] < closest_dist:
                    t = hit[0]
                    hit_pos = origin + direction * t
                    closest_dist = t
                    closest_hit = hit_pos
        
        if closest_hit is not None:
            return (closest_dist, closest_hit)
        return (None, None)
    
    def _ray_plane_intersection(self, origin, direction, plane_y, plane_type):
        """Check ray intersection with floor or ceiling."""
        # Plane normal (pointing up for floor, down for ceiling)
        normal = np.array([0, 1 if plane_type == 'floor' else -1, 0])
        
        denom = np.dot(direction, normal)
        if abs(denom) < 0.0001:
            return None
        
        t = (plane_y - origin[1]) / direction[1]
        
        if t > 0 and t < self.max_ray_distance:
            hit_pos = origin + direction * t
            return (t, hit_pos)
        
        return None
    
    def _calculate_reverb(self):
        """Calculate reverb parameters from ray distances."""
        if not self.ray_distances:
            self.reverb_amount = 0.0
            self.room_size_estimate = 0.0
            return
        
        # Average distance to surfaces
        avg_dist = np.mean(self.ray_distances)
        
        # Variance in distances (higher = more complex space)
        variance = np.std(self.ray_distances)
        
        # Room size estimate (normalized)
        self.room_size_estimate = min(1.0, avg_dist / self.max_ray_distance)
        
        # Reverb amount based on average distance and variance
        # Larger rooms = more reverb, more complex spaces = more reverb
        self.reverb_amount = min(0.7, (avg_dist / 100.0) * 0.5 + (variance / 50.0) * 0.2)


class AcousticIntegration:
    """
    Full acoustic raycasting with spatial audio.
    Processes voice through reverb based on room geometry.
    """
    
    def __init__(self, engine):
        self.engine = engine
        
        # Raycaster
        self.raycaster = AcousticRaycaster(engine)
        
        # Audio backend
        self.mic = MicrophoneCapture(sample_rate=44100, chunk_size=1024)
        self.enabled = False
        
        # Reverb parameters (updated by raycasting)
        self.reverb_amount = 0.3
        self.room_size = 0.5
        
        # Reverb buffer (simple comb filter)
        self.reverb_buffer_size = 22050  # 0.5 seconds at 44.1kHz
        self.reverb_buffer = np.zeros(self.reverb_buffer_size, dtype=np.float32)
        self.reverb_buffer_index = 0
        
        # Threading
        self.playback_queue = queue.Queue(maxsize=8)
        self.playback_thread = None
        self.update_thread = None
        self.should_stop = False
        self.current_channel = None
        
        # Raycast update rate (don't raycast every frame, too expensive)
        self.raycast_timer = 0.0
        self.raycast_interval = 0.1  # Update every 100ms
        
        # Visualization
        self.show_echo_viz = False
        self.echo_viz_fade = 0.0
        
    def toggle(self):
        """Toggle acoustic raycasting system."""
        if self.enabled:
            self.stop()
        else:
            self.start()
    
    def start(self):
        """Start acoustic system."""
        if self.mic.start():
            self.enabled = True
            self.should_stop = False
            
            # Start playback thread
            self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
            self.playback_thread.start()
            
            # Start update thread
            self.update_thread = threading.Thread(target=self._update_worker, daemon=True)
            self.update_thread.start()
            
            print("🔊 Acoustic raycasting ON")
            print(f"   Casting {self.raycaster.num_rays} rays for spatial audio")
            print("   Speak into your microphone!")
            return True
        
        print("❌ Failed to start audio system")
        return False
    
    def stop(self):
        """Stop acoustic system."""
        self.mic.stop()
        self.enabled = False
        self.should_stop = True
        
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)
        
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        
        self.current_channel = None
        print("🔇 Acoustic raycasting OFF")
    
    def toggle_visualization(self):
        """Toggle echo visualization."""
        self.show_echo_viz = not self.show_echo_viz
        status = "ON" if self.show_echo_viz else "OFF"
        print(f"👁️  Echo visualization {status}")
    
    def update(self, dt):
        """Update acoustic system (called from main game loop)."""
        if not self.enabled:
            return
        
        # Update raycast periodically
        self.raycast_timer += dt
        if self.raycast_timer >= self.raycast_interval:
            self.raycast_timer = 0.0
            self._update_raycasting()
        
        # Update visualization fade
        if self.echo_viz_fade > 0:
            self.echo_viz_fade -= dt * 2.0
            self.echo_viz_fade = max(0, self.echo_viz_fade)
    
    def _update_raycasting(self):
        """Update raycasting and reverb parameters."""
        player_pos = (
            self.engine.camera.x_s,
            self.engine.camera.y_s,
            self.engine.camera.z_s
        )
        
        params = self.raycaster.cast_rays(player_pos)
        
        # Update reverb parameters
        self.reverb_amount = params['reverb_amount']
        self.room_size = params['room_size']
        
        # Trigger visualization
        if self.show_echo_viz:
            self.echo_viz_fade = 1.0
    
    def _playback_worker(self):
        """Worker thread for smooth audio playback."""
        import time
        
        while not self.should_stop:
            try:
                left, right = self.playback_queue.get(timeout=0.1)
                
                # Wait for previous chunk to finish
                if self.current_channel:
                    while self.current_channel.get_busy() and not self.should_stop:
                        time.sleep(0.002)
                
                # Play new chunk
                stereo = np.column_stack((left, right))
                sound = pygame.sndarray.make_sound(stereo)
                self.current_channel = sound.play()
                
            except queue.Empty:
                continue
    
    def _update_worker(self):
        """Dedicated update thread for audio processing."""
        import time
        
        while not self.should_stop:
            if self.enabled:
                audio_chunk = self.mic.get_audio_chunk()
                if audio_chunk is not None:
                    # Process with reverb
                    left, right = self._process_chunk_with_reverb(audio_chunk)
                    
                    # Queue management
                    if self.playback_queue.full():
                        try:
                            self.playback_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    # Queue processed audio
                    try:
                        self.playback_queue.put_nowait((left, right))
                    except queue.Full:
                        pass
            
            time.sleep(0.001)  # 1000 Hz update rate
    
    def _process_chunk_with_reverb(self, audio_chunk):
        """
        Process audio chunk with spatial reverb AND early reflections.
        Simulates how sound travels from mouth → ears via direct path + reflections.
        """
        # Convert to float
        audio_float = audio_chunk.astype(np.float32) / 32768.0
        
        # Normalize
        max_val = np.max(np.abs(audio_float))
        if max_val > 0.9:
            audio_float = audio_float * (0.9 / max_val)
        
        # Apply gain
        audio_float = audio_float * 2.5
        
        # === SIMULATE SOUND PATH ===
        # In reality, sound from your mouth reaches your ears via:
        # 1. Direct path (instant, very short distance)
        # 2. Early reflections (first bounce off walls - gives you spatial sense)
        # 3. Late reverb (multiple bounces - the "room sound")
        
        # Direct sound (from mouth to ears) - ~0.3 meters
        # At 343 m/s sound speed, that's ~0.87ms delay
        # At 44.1kHz, that's ~38 samples
        direct_delay = 38
        
        # Process with early reflections + late reverb
        output_left = np.zeros_like(audio_float)
        output_right = np.zeros_like(audio_float)
        
        # Get ray information for early reflections
        if hasattr(self.raycaster, 'ray_hits') and len(self.raycaster.ray_hits) > 0:
            early_reflections_left = []
            early_reflections_right = []
            
            # Calculate early reflections from raycasts
            for ray_idx, (hit_pos, distance) in enumerate(zip(self.raycaster.ray_hits, self.raycaster.ray_distances)):
                if hit_pos is None or distance >= self.raycaster.max_ray_distance:
                    continue
                
                # SAFETY: Skip reflections that are too close (causes feedback)
                if distance < 5.0:  # Less than 5 units = too close
                    continue
                
                # Calculate delay based on distance
                # Sound travels at 343 m/s, sample rate is 44100 Hz
                # delay_samples = (distance / 343) * 44100
                delay_samples = int((distance / 343.0) * 44100.0 / 2.0)  # Divide by 2 since sound goes there and back
                delay_samples = min(delay_samples, self.reverb_buffer_size - 1)
                
                if delay_samples < 50:  # Skip very short delays (less than 1ms) - prevents feedback
                    continue
                
                # Determine left/right based on ray direction
                # Calculate angle of reflection relative to player's facing
                player_yaw = self.engine.camera.yaw_s
                
                # Calculate direction to hit point
                dx = hit_pos[0] - self.engine.camera.x_s
                dz = hit_pos[2] - self.engine.camera.z_s
                angle_to_hit = math.atan2(dx, dz)
                
                # Relative angle (-pi to pi)
                relative_angle = angle_to_hit - player_yaw
                while relative_angle > math.pi:
                    relative_angle -= 2 * math.pi
                while relative_angle < -math.pi:
                    relative_angle += 2 * math.pi
                
                # Pan: -1 (left) to +1 (right)
                pan = math.sin(relative_angle)
                
                # Attenuation based on distance (inverse square law with minimum distance)
                # Ensure we never get attenuation > 1.0
                attenuation = 1.0 / (1.0 + distance / 20.0)  # More aggressive falloff
                attenuation = min(0.6, attenuation)  # Cap maximum reflection strength
                
                # Calculate per-ear gain
                left_gain = attenuation * max(0, 1.0 - pan) * 0.2  # Reduced from 0.3
                right_gain = attenuation * max(0, 1.0 + pan) * 0.2
                
                # Only add if gain is meaningful
                if left_gain > 0.01:
                    early_reflections_left.append({
                        'delay': delay_samples,
                        'gain': left_gain,
                    })
                
                if right_gain > 0.01:
                    early_reflections_right.append({
                        'delay': delay_samples,
                        'gain': right_gain,
                    })
            
            # SAFETY: Normalize total reflection energy to prevent clipping
            # Calculate total gain for each ear
            total_left_gain = sum(r['gain'] for r in early_reflections_left)
            total_right_gain = sum(r['gain'] for r in early_reflections_right)
            
            # If total gain exceeds safe threshold, scale everything down
            max_total_gain = 0.8  # Never let reflections sum to more than 80% volume
            
            if total_left_gain > max_total_gain:
                scale = max_total_gain / total_left_gain
                for r in early_reflections_left:
                    r['gain'] *= scale
            
            if total_right_gain > max_total_gain:
                scale = max_total_gain / total_right_gain
                for r in early_reflections_right:
                    r['gain'] *= scale
        else:
            early_reflections_left = []
            early_reflections_right = []
        
        # Process each sample
        for i in range(len(audio_float)):
            dry = audio_float[i]
            
            # === WRITE TO REVERB BUFFER FIRST ===
            # This ensures the buffer is populated even for the first samples
            buffer_write = dry * 0.8  # Write mostly dry signal
            buffer_write = max(-0.95, min(0.95, buffer_write))  # Safety clip
            self.reverb_buffer[self.reverb_buffer_index] = buffer_write
            
            # === 1. DIRECT SOUND (centered, immediate) ===
            direct_left = dry * 0.7  # Increased - ensures you always hear yourself
            direct_right = dry * 0.7
            
            # === 2. EARLY REFLECTIONS (directional, based on raycasts) ===
            early_left = 0.0
            early_right = 0.0
            
            for reflection in early_reflections_left[:8]:  # Use first 8 reflections only
                read_idx = (self.reverb_buffer_index - reflection['delay']) % self.reverb_buffer_size
                early_left += self.reverb_buffer[read_idx] * reflection['gain']
            
            for reflection in early_reflections_right[:8]:
                read_idx = (self.reverb_buffer_index - reflection['delay']) % self.reverb_buffer_size
                early_right += self.reverb_buffer[read_idx] * reflection['gain']
            
            # === 3. LATE REVERB (diffuse, room sound) ===
            # Use average room size for late reverb delay
            late_delay = int(self.room_size * 10000)
            late_delay = min(late_delay, self.reverb_buffer_size - 1)
            late_delay = max(100, late_delay)  # At least 100 samples delay
            
            late_idx = (self.reverb_buffer_index - late_delay) % self.reverb_buffer_size
            late_reverb = self.reverb_buffer[late_idx] * self.reverb_amount * 0.4
            
            # === MIX ALL COMPONENTS ===
            output_left[i] = direct_left + early_left + late_reverb
            output_right[i] = direct_right + early_right + late_reverb
            
            # SAFETY: Hard clip to prevent overflow
            output_left[i] = max(-0.95, min(0.95, output_left[i]))
            output_right[i] = max(-0.95, min(0.95, output_right[i]))
            
            # Advance buffer index
            self.reverb_buffer_index = (self.reverb_buffer_index + 1) % self.reverb_buffer_size
        
        # Convert back to int16
        output_left = np.clip(output_left * 32767, -32768, 32767).astype(np.int16)
        output_right = np.clip(output_right * 32767, -32768, 32767).astype(np.int16)
        
        return output_left, output_right
    
    def render_visualization(self, screen, camera):
        """Render echo visualization showing ray paths."""
        if not self.show_echo_viz or self.echo_viz_fade <= 0:
            return
        
        if not self.raycaster.ray_hits:
            return
        
        # Draw rays from player to hit points
        player_x = self.engine.camera.x_s
        player_y = self.engine.camera.y_s
        player_z = self.engine.camera.z_s
        
        for i, hit_pos in enumerate(self.raycaster.ray_hits):
            if hit_pos is None:
                continue
            
            # Transform start point to screen
            start_cam = camera.world_to_camera(player_x, player_y, player_z)
            start_screen = camera.project(start_cam)
            
            if not start_screen:
                continue
            
            # Transform hit point to screen
            hit_cam = camera.world_to_camera(hit_pos[0], hit_pos[1], hit_pos[2])
            hit_screen = camera.project(hit_cam)
            
            if not hit_screen:
                continue
            
            # Color based on distance
            distance = self.raycaster.ray_distances[i]
            intensity = 1.0 - min(1.0, distance / self.raycaster.max_ray_distance)
            
            # Cyan rays with fade
            alpha = int(self.echo_viz_fade * 128 * intensity)
            color = (0, 255, 255, alpha)
            
            # Draw line
            try:
                start_pos = (int(start_screen[0]), int(start_screen[1]))
                end_pos = (int(hit_screen[0]), int(hit_screen[1]))
                
                # Create temp surface for alpha blending
                temp_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
                pygame.draw.line(temp_surface, color, start_pos, end_pos, 1)
                screen.blit(temp_surface, (0, 0))
            except:
                pass


def add_acoustic_controls_to_help(help_texts):
    """Add acoustic system controls to help overlay."""
    acoustic_help = [
        "=== AUDIO SYSTEM ===",
        "N: Toggle Acoustic Raycasting",
        "M: Toggle Ray Visualization",
        "16 rays cast for spatial reverb!",
    ]
    return help_texts + [""] + acoustic_help
