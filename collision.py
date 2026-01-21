"""
Precise Collision System
=========================
Circle collider vs line segments with sliding response.
Makes the Backrooms feel physically real.
"""

import math
import numpy as np
from config import PILLAR_SPACING, WALL_THICKNESS, HALLWAY_WIDTH, PILLAR_SIZE


class CollisionSystem:
    """
    Handles circle-vs-segment collision with sliding response.
    Returns corrected position instead of just True/False.
    """
    
    def __init__(self, world, player_radius=15.0, skin_width=0.5):
        self.world = world
        self.player_radius = player_radius
        self.skin_width = skin_width  # Prevents jitter at edges
        self.max_iterations = 4  # Sub-stepping for precise collision
    
    def resolve_collision(self, from_pos, to_pos):
        """
        Move from from_pos to to_pos, sliding along any collisions.
        
        Returns:
            tuple: (final_x, final_z, collided)
        """
        x1, z1 = from_pos
        x2, z2 = to_pos
        
        # Calculate movement vector
        move_x = x2 - x1
        move_z = z2 - z1
        move_dist = math.sqrt(move_x*move_x + move_z*move_z)
        
        if move_dist < 0.001:
            return (x1, z1, False)
        
        # Try the movement with multiple resolution passes
        final_x, final_z = x2, z2
        collided = False
        
        # Multiple passes to handle corner cases
        for iteration in range(3):
            # Depenetrate first if stuck
            if self._is_penetrating(final_x, final_z):
                final_x, final_z = self._depenetrate(final_x, final_z)
                collided = True
            
            # Get all nearby segments
            segments = self._get_nearby_segments(final_x, final_z)
            
            # Check for any remaining collisions and resolve
            resolved = False
            for segment in segments:
                result = self._resolve_segment_collision(final_x, final_z, segment)
                if result:
                    final_x, final_z = result
                    collided = True
                    resolved = True
            
            # If no collision found, we're done
            if not resolved:
                break
        
        return (final_x, final_z, collided)
    
    def _is_penetrating(self, x, z):
        """Quick check if currently penetrating any geometry."""
        segments = self._get_nearby_segments(x, z)
        threshold = self.player_radius + self.skin_width
        
        for segment in segments:
            dist = self._distance_to_segment(x, z, segment)
            if dist < threshold:
                return True
        return False
    
    def _distance_to_segment(self, x, z, segment):
        """Calculate distance from point to line segment."""
        seg_x1, seg_z1, seg_x2, seg_z2 = segment
        
        # Segment vector
        seg_dx = seg_x2 - seg_x1
        seg_dz = seg_z2 - seg_z1
        seg_len = math.sqrt(seg_dx*seg_dx + seg_dz*seg_dz)
        
        if seg_len < 0.001:
            return float('inf')
        
        # Normalize
        seg_nx = seg_dx / seg_len
        seg_nz = seg_dz / seg_len
        
        # Project point onto line
        to_x = x - seg_x1
        to_z = z - seg_z1
        proj = to_x * seg_nx + to_z * seg_nz
        proj = max(0, min(seg_len, proj))
        
        # Closest point
        closest_x = seg_x1 + seg_nx * proj
        closest_z = seg_z1 + seg_nz * proj
        
        # Distance
        return math.sqrt((x - closest_x)**2 + (z - closest_z)**2)
    
    def _resolve_segment_collision(self, x, z, segment):
        """
        If point is too close to segment, push it away.
        Returns new position if collision resolved, None otherwise.
        """
        seg_x1, seg_z1, seg_x2, seg_z2 = segment
        
        # Segment vector
        seg_dx = seg_x2 - seg_x1
        seg_dz = seg_z2 - seg_z1
        seg_len = math.sqrt(seg_dx*seg_dx + seg_dz*seg_dz)
        
        if seg_len < 0.001:
            return None
        
        # Normalize
        seg_nx = seg_dx / seg_len
        seg_nz = seg_dz / seg_len
        
        # Normal (perpendicular)
        normal_x = -seg_nz
        normal_z = seg_nx
        
        # Project point onto segment
        to_x = x - seg_x1
        to_z = z - seg_z1
        proj = to_x * seg_nx + to_z * seg_nz
        proj = max(0, min(seg_len, proj))
        
        # Closest point
        closest_x = seg_x1 + seg_nx * proj
        closest_z = seg_z1 + seg_nz * proj
        
        # Vector from segment to point
        diff_x = x - closest_x
        diff_z = z - closest_z
        dist = math.sqrt(diff_x*diff_x + diff_z*diff_z)
        
        # Check if penetrating
        threshold = self.player_radius + self.skin_width
        if dist >= threshold:
            return None
        
        # Penetrating - push out
        if dist > 0.001:
            # Push along vector from closest point to player
            push_x = (diff_x / dist) * (threshold - dist)
            push_z = (diff_z / dist) * (threshold - dist)
        else:
            # Directly on segment - push along normal
            # Determine which side
            side = to_x * normal_x + to_z * normal_z
            if side < 0:
                push_x = -normal_x * threshold
                push_z = -normal_z * threshold
            else:
                push_x = normal_x * threshold
                push_z = normal_z * threshold
        
        return (x + push_x, z + push_z)
    
    def _depenetrate(self, x, z):
        """Push player out if stuck in geometry."""
        segments = self._get_nearby_segments(x, z)
        
        push_x = 0
        push_z = 0
        push_count = 0
        
        for segment in segments:
            seg_x1, seg_z1, seg_x2, seg_z2 = segment
            
            # Segment vector
            seg_dx = seg_x2 - seg_x1
            seg_dz = seg_z2 - seg_z1
            seg_len = math.sqrt(seg_dx*seg_dx + seg_dz*seg_dz)
            
            if seg_len < 0.001:
                continue
            
            seg_nx = seg_dx / seg_len
            seg_nz = seg_dz / seg_len
            
            # Normal
            normal_x = -seg_nz
            normal_z = seg_nx
            
            # Closest point
            to_x = x - seg_x1
            to_z = z - seg_z1
            proj = max(0, min(seg_len, to_x * seg_nx + to_z * seg_nz))
            
            closest_x = seg_x1 + seg_nx * proj
            closest_z = seg_z1 + seg_nz * proj
            
            # Distance
            dist_x = x - closest_x
            dist_z = z - closest_z
            dist = math.sqrt(dist_x*dist_x + dist_z*dist_z)
            
            # If penetrating, push out
            if dist < self.player_radius:
                penetration = self.player_radius - dist
                
                # Determine side
                if dist > 0.001:
                    push_x += (dist_x / dist) * penetration
                    push_z += (dist_z / dist) * penetration
                else:
                    # Directly on segment - push along normal
                    side = to_x * normal_x + to_z * normal_z
                    if side < 0:
                        normal_x = -normal_x
                        normal_z = -normal_z
                    push_x += normal_x * penetration
                    push_z += normal_z * penetration
                
                push_count += 1
        
        if push_count > 0:
            return (x + push_x, z + push_z)
        
        return (x, z)
    
    def _is_stuck(self, x, z):
        """Check if position is stuck inside geometry (legacy method)."""
        return self._is_penetrating(x, z)
    
    def _get_nearby_segments(self, x, z):
        """
        Get all wall and pillar segments near position.
        Returns list of (x1, z1, x2, z2) tuples.
        """
        segments = []
        half_thick = WALL_THICKNESS / 2
        check_range = PILLAR_SPACING * 2
        
        min_grid_x = int((x - check_range) // PILLAR_SPACING) * PILLAR_SPACING
        max_grid_x = int((x + check_range) // PILLAR_SPACING) * PILLAR_SPACING
        min_grid_z = int((z - check_range) // PILLAR_SPACING) * PILLAR_SPACING
        max_grid_z = int((z + check_range) // PILLAR_SPACING) * PILLAR_SPACING
        
        for px in range(min_grid_x, max_grid_x + PILLAR_SPACING, PILLAR_SPACING):
            for pz in range(min_grid_z, max_grid_z + PILLAR_SPACING, PILLAR_SPACING):
                
                # Horizontal walls
                if self.world.has_wall_between(px, pz, px + PILLAR_SPACING, pz):
                    wall_key = tuple(sorted([(px, pz), (px + PILLAR_SPACING, pz)]))
                    if wall_key not in self.world.destroyed_walls:
                        
                        opening_type = self.world.get_doorway_type(px, pz, px + PILLAR_SPACING, pz)
                        wall_z = pz
                        wall_x_start = px
                        wall_x_end = px + PILLAR_SPACING
                        
                        if opening_type == "hallway":
                            opening_width = HALLWAY_WIDTH
                        elif opening_type == "doorway":
                            opening_width = 60
                        else:
                            opening_width = 0
                        
                        if opening_width > 0:
                            # Wall with opening - create segments + doorway edges
                            opening_start = wall_x_start + (PILLAR_SPACING - opening_width) / 2
                            opening_end = opening_start + opening_width
                            
                            # Left wall segment (both sides)
                            segments.append((wall_x_start, wall_z - half_thick,
                                           opening_start, wall_z - half_thick))
                            segments.append((wall_x_start, wall_z + half_thick,
                                           opening_start, wall_z + half_thick))
                            
                            # Right wall segment (both sides)
                            segments.append((opening_end, wall_z - half_thick,
                                           wall_x_end, wall_z - half_thick))
                            segments.append((opening_end, wall_z + half_thick,
                                           wall_x_end, wall_z + half_thick))
                            
                            # Doorway edges (perpendicular segments)
                            segments.append((opening_start, wall_z - half_thick,
                                           opening_start, wall_z + half_thick))
                            segments.append((opening_end, wall_z - half_thick,
                                           opening_end, wall_z + half_thick))
                        else:
                            # Solid wall
                            segments.append((wall_x_start, wall_z - half_thick,
                                           wall_x_end, wall_z - half_thick))
                            segments.append((wall_x_start, wall_z + half_thick,
                                           wall_x_end, wall_z + half_thick))
                
                # Vertical walls (same logic, rotated)
                if self.world.has_wall_between(px, pz, px, pz + PILLAR_SPACING):
                    wall_key = tuple(sorted([(px, pz), (px, pz + PILLAR_SPACING)]))
                    if wall_key not in self.world.destroyed_walls:
                        
                        opening_type = self.world.get_doorway_type(px, pz, px, pz + PILLAR_SPACING)
                        wall_x = px
                        wall_z_start = pz
                        wall_z_end = pz + PILLAR_SPACING
                        
                        if opening_type == "hallway":
                            opening_width = HALLWAY_WIDTH
                        elif opening_type == "doorway":
                            opening_width = 60
                        else:
                            opening_width = 0
                        
                        if opening_width > 0:
                            opening_start = wall_z_start + (PILLAR_SPACING - opening_width) / 2
                            opening_end = opening_start + opening_width
                            
                            # Bottom wall segment (both sides)
                            segments.append((wall_x - half_thick, wall_z_start,
                                           wall_x - half_thick, opening_start))
                            segments.append((wall_x + half_thick, wall_z_start,
                                           wall_x + half_thick, opening_start))
                            
                            # Top wall segment (both sides)
                            segments.append((wall_x - half_thick, opening_end,
                                           wall_x - half_thick, wall_z_end))
                            segments.append((wall_x + half_thick, opening_end,
                                           wall_x + half_thick, wall_z_end))
                            
                            # Doorway edges (perpendicular segments)
                            segments.append((wall_x - half_thick, opening_start,
                                           wall_x + half_thick, opening_start))
                            segments.append((wall_x - half_thick, opening_end,
                                           wall_x + half_thick, opening_end))
                        else:
                            segments.append((wall_x - half_thick, wall_z_start,
                                           wall_x - half_thick, wall_z_end))
                            segments.append((wall_x + half_thick, wall_z_start,
                                           wall_x + half_thick, wall_z_end))
                
                # Pillars (check if they exist)
                offset = PILLAR_SPACING // 2
                pillar_x = px + offset
                pillar_z = pz + offset
                
                if self.world.has_pillar_at(pillar_x, pillar_z):
                    pillar_key = (pillar_x, pillar_z)
                    if pillar_key not in self.world.destroyed_pillars:
                        s = PILLAR_SIZE
                        
                        # Four sides of pillar
                        segments.append((pillar_x, pillar_z, pillar_x + s, pillar_z))  # Front
                        segments.append((pillar_x + s, pillar_z, pillar_x + s, pillar_z + s))  # Right
                        segments.append((pillar_x + s, pillar_z + s, pillar_x, pillar_z + s))  # Back
                        segments.append((pillar_x, pillar_z + s, pillar_x, pillar_z))  # Left
        
        return segments
