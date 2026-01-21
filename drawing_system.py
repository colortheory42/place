"""
Wall Drawing System
===================
Allows player to draw/spray paint on walls.
Drawings are persistent and saved with game state.
"""

import pygame
import numpy as np
from collections import defaultdict
from config import PILLAR_SPACING, WALL_THICKNESS, get_scaled_wall_height, get_scaled_floor_y


class WallDrawing:
    """Manages all drawings on walls."""
    
    def __init__(self):
        # Store drawings per wall: {wall_key: [stroke1, stroke2, ...]}
        # Each stroke is a list of (u, v) coordinates on the wall surface (0-1 range)
        self.wall_drawings = defaultdict(list)
        
        # Store drawings per pillar: {pillar_key: {face: [stroke1, stroke2, ...]}}
        # face can be 'front', 'back', 'left', 'right'
        self.pillar_drawings = defaultdict(lambda: defaultdict(list))
        
        # Current stroke being drawn
        self.current_stroke = None
        self.current_wall = None  # Track which wall we're drawing on
        self.current_pillar = None  # Track which pillar we're drawing on
        self.current_pillar_face = None  # Track which face of the pillar
        self.drawing_active = False
        
        # Drawing settings
        self.brush_size = 0.015  # Size in UV space (0-1)
        self.draw_color = (0, 0, 0)  # Black
        
    def start_stroke(self, wall_key, uv_pos):
        """Start a new drawing stroke on a wall."""
        self.current_stroke = [uv_pos]
        self.current_wall = wall_key
        self.current_pillar = None
        self.current_pillar_face = None
        self.drawing_active = True
    
    def start_pillar_stroke(self, pillar_key, face, uv_pos):
        """Start a new drawing stroke on a pillar face."""
        self.current_stroke = [uv_pos]
        self.current_pillar = pillar_key
        self.current_pillar_face = face
        self.current_wall = None
        self.drawing_active = True
        
    def add_to_stroke(self, uv_pos):
        """Add a point to the current stroke."""
        if self.drawing_active and self.current_stroke is not None:
            # Only add if it's not too close to last point (smooth out)
            if len(self.current_stroke) > 0:
                last_u, last_v = self.current_stroke[-1]
                du = uv_pos[0] - last_u
                dv = uv_pos[1] - last_v
                dist = (du*du + dv*dv) ** 0.5
                
                # Add point if it moved enough
                if dist > 0.005:  # Minimum spacing
                    self.current_stroke.append(uv_pos)
            else:
                self.current_stroke.append(uv_pos)
                
    def end_stroke(self, target_key=None, pillar_face=None):
        """Finish the current stroke and save it."""
        if self.drawing_active and self.current_stroke and len(self.current_stroke) > 0:
            if self.current_wall:
                # Save to wall
                self.wall_drawings[self.current_wall].append(self.current_stroke)
            elif self.current_pillar and self.current_pillar_face:
                # Save to pillar face
                self.pillar_drawings[self.current_pillar][self.current_pillar_face].append(self.current_stroke)
            
        self.current_stroke = None
        self.current_wall = None
        self.current_pillar = None
        self.current_pillar_face = None
        self.drawing_active = False
        
    def get_drawings_for_wall(self, wall_key):
        """Get all drawing strokes for a specific wall."""
        return self.wall_drawings.get(wall_key, [])
    
    def get_drawings_for_pillar(self, pillar_key, face):
        """Get all drawing strokes for a specific pillar face."""
        if pillar_key in self.pillar_drawings:
            return self.pillar_drawings[pillar_key].get(face, [])
        return []
    
    def clear_wall(self, wall_key):
        """Clear all drawings on a specific wall."""
        if wall_key in self.wall_drawings:
            del self.wall_drawings[wall_key]
    
    def clear_pillar(self, pillar_key):
        """Clear all drawings on a specific pillar."""
        if pillar_key in self.pillar_drawings:
            del self.pillar_drawings[pillar_key]
    
    def clear_all(self):
        """Clear all drawings everywhere."""
        self.wall_drawings.clear()
        self.pillar_drawings.clear()
        self.current_stroke = None
        self.current_wall = None
        self.current_pillar = None
        self.current_pillar_face = None
        self.drawing_active = False
    
    def get_state_for_save(self):
        """Get drawing state for saving."""
        # Convert defaultdict to regular dict for JSON serialization
        pillar_data = {}
        for pillar_key, faces in self.pillar_drawings.items():
            pillar_data[str(pillar_key)] = {
                face: strokes for face, strokes in faces.items()
            }
        
        return {
            'wall_drawings': {
                str(key): strokes 
                for key, strokes in self.wall_drawings.items()
            },
            'pillar_drawings': pillar_data
        }
    
    def load_state(self, data):
        """Load drawing state from save data."""
        self.clear_all()
        
        if 'wall_drawings' in data:
            for key_str, strokes in data['wall_drawings'].items():
                # Convert string key back to tuple
                key = eval(key_str)  # Safe here since we control the format
                self.wall_drawings[key] = strokes
        
        if 'pillar_drawings' in data:
            for pillar_key_str, faces_data in data['pillar_drawings'].items():
                pillar_key = eval(pillar_key_str)
                for face, strokes in faces_data.items():
                    self.pillar_drawings[pillar_key][face] = strokes


def world_to_wall_uv(hit_point, wall_key, world):
    """
    Convert a 3D world hit point to UV coordinates on a wall surface.
    Returns (u, v) in 0-1 range, or None if invalid.
    """
    (p1_x, p1_z), (p2_x, p2_z) = wall_key
    hit_x, hit_y, hit_z = hit_point
    
    # Determine if horizontal or vertical wall
    if abs(p1_z - p2_z) < 0.1:  # Horizontal wall (runs along X axis)
        # U coordinate is along wall length (X axis)
        wall_length = abs(p2_x - p1_x)
        u = (hit_x - min(p1_x, p2_x)) / wall_length
        
    else:  # Vertical wall (runs along Z axis)
        # U coordinate is along wall length (Z axis)
        wall_length = abs(p2_z - p1_z)
        u = (hit_z - min(p1_z, p2_z)) / wall_length
    
    # V coordinate is height (Y axis)
    floor_y = get_scaled_floor_y()
    wall_height = get_scaled_wall_height() - floor_y
    v = (hit_y - floor_y) / wall_height
    
    # Clamp to 0-1 range
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))
    
    return (u, v)


def get_wall_hit_point(camera, world, target_info):
    """
    Get the exact 3D point where the camera ray hits the wall.
    Returns (x, y, z) world coordinates or None.
    """
    if not target_info:
        return None
    
    target_type, target_key = target_info
    if target_type != 'wall':
        return None
    
    from raycasting import ray_intersects_triangle
    
    # Get ray from camera
    ray_origin, ray_dir = camera.get_ray_direction()
    max_distance = 100
    
    # Get wall geometry
    (p1_x, p1_z), (p2_x, p2_z) = target_key
    h = get_scaled_wall_height()
    floor_y = get_scaled_floor_y()
    half_thick = WALL_THICKNESS / 2
    
    # Build wall triangles
    if abs(p1_z - p2_z) < 0.1:  # Horizontal wall
        z = p1_z
        x1, x2 = p1_x, p2_x
        
        # Front face
        v0 = (x1, h, z - half_thick)
        v1 = (x2, h, z - half_thick)
        v2 = (x2, floor_y, z - half_thick)
        v3 = (x1, floor_y, z - half_thick)
        
    else:  # Vertical wall
        x = p1_x
        z1, z2 = p1_z, p2_z
        
        # Front face
        v0 = (x - half_thick, h, z1)
        v1 = (x - half_thick, h, z2)
        v2 = (x - half_thick, floor_y, z2)
        v3 = (x - half_thick, floor_y, z1)
    
    # Test ray against both triangles
    for tri in [(v0, v1, v2), (v0, v2, v3)]:
        hit = ray_intersects_triangle(ray_origin, ray_dir, *tri)
        if hit and hit[0] < max_distance:
            # Calculate hit point
            t = hit[0]
            hit_point = (
                ray_origin[0] + ray_dir[0] * t,
                ray_origin[1] + ray_dir[1] * t,
                ray_origin[2] + ray_dir[2] * t
            )
            return hit_point
    
    return None


def get_pillar_hit_point_and_face(camera, world, target_info):
    """
    Get the exact 3D point where the camera ray hits a pillar and which face.
    Returns ((x, y, z), face_name) or (None, None).
    face_name can be 'front', 'back', 'left', 'right'
    """
    if not target_info:
        return None, None
    
    target_type, target_key = target_info
    if target_type != 'pillar':
        return None, None
    
    from raycasting import ray_intersects_triangle
    from config import PILLAR_SIZE
    
    # Get ray from camera
    ray_origin, ray_dir = camera.get_ray_direction()
    max_distance = 100
    
    # Get pillar geometry
    pillar_x, pillar_z = target_key
    h = get_scaled_wall_height()
    floor_y = get_scaled_floor_y()
    s = PILLAR_SIZE
    
    # Define all four faces with their names
    faces = {
        'front': [  # Front face (facing -Z)
            (pillar_x, h, pillar_z),
            (pillar_x + s, h, pillar_z),
            (pillar_x + s, floor_y, pillar_z),
            (pillar_x, floor_y, pillar_z)
        ],
        'back': [  # Back face (facing +Z)
            (pillar_x + s, h, pillar_z + s),
            (pillar_x, h, pillar_z + s),
            (pillar_x, floor_y, pillar_z + s),
            (pillar_x + s, floor_y, pillar_z + s)
        ],
        'left': [  # Left face (facing -X)
            (pillar_x, h, pillar_z),
            (pillar_x, h, pillar_z + s),
            (pillar_x, floor_y, pillar_z + s),
            (pillar_x, floor_y, pillar_z)
        ],
        'right': [  # Right face (facing +X)
            (pillar_x + s, h, pillar_z + s),
            (pillar_x + s, h, pillar_z),
            (pillar_x + s, floor_y, pillar_z),
            (pillar_x + s, floor_y, pillar_z + s)
        ]
    }
    
    # Test ray against all faces
    closest_hit = None
    closest_dist = float('inf')
    hit_face = None
    
    for face_name, verts in faces.items():
        v0, v1, v2, v3 = verts
        for tri in [(v0, v1, v2), (v0, v2, v3)]:
            hit = ray_intersects_triangle(ray_origin, ray_dir, *tri)
            if hit and hit[0] < max_distance and hit[0] < closest_dist:
                closest_dist = hit[0]
                t = hit[0]
                closest_hit = (
                    ray_origin[0] + ray_dir[0] * t,
                    ray_origin[1] + ray_dir[1] * t,
                    ray_origin[2] + ray_dir[2] * t
                )
                hit_face = face_name
    
    return closest_hit, hit_face


def pillar_world_to_uv(hit_point, pillar_key, face):
    """
    Convert a 3D world hit point on a pillar to UV coordinates.
    Returns (u, v) in 0-1 range, or None if invalid.
    """
    from config import PILLAR_SIZE
    
    pillar_x, pillar_z = pillar_key
    hit_x, hit_y, hit_z = hit_point
    s = PILLAR_SIZE
    
    floor_y = get_scaled_floor_y()
    pillar_height = get_scaled_wall_height() - floor_y
    
    # V coordinate is always height (same for all faces)
    v = (hit_y - floor_y) / pillar_height
    
    # U coordinate depends on which face we hit
    if face == 'front':  # Along X axis
        u = (hit_x - pillar_x) / s
    elif face == 'back':  # Along X axis (reversed)
        u = (pillar_x + s - hit_x) / s
    elif face == 'left':  # Along Z axis
        u = (hit_z - pillar_z) / s
    elif face == 'right':  # Along Z axis (reversed)
        u = (pillar_z + s - hit_z) / s
    else:
        return None
    
    # Clamp to 0-1 range
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))
    
    return (u, v)


def render_drawings_on_wall(surface, wall_key, drawing_system, texture_rect):
    """
    Render all drawings for a wall onto its texture surface.
    
    Args:
        surface: pygame.Surface to draw on
        wall_key: the wall identifier
        drawing_system: WallDrawing instance
        texture_rect: pygame.Rect defining the texture area
    """
    strokes = drawing_system.get_drawings_for_wall(wall_key)
    if not strokes:
        return
    
    # Get texture dimensions
    tex_width = texture_rect.width
    tex_height = texture_rect.height
    
    # Draw each stroke
    for stroke in strokes:
        if len(stroke) < 2:
            # Single point - draw a circle
            if len(stroke) == 1:
                u, v = stroke[0]
                x = int(u * tex_width) + texture_rect.x
                y = int(v * tex_height) + texture_rect.y
                radius = max(2, int(drawing_system.brush_size * min(tex_width, tex_height)))
                pygame.draw.circle(surface, drawing_system.draw_color, (x, y), radius)
        else:
            # Multiple points - draw connected line segments
            for i in range(len(stroke) - 1):
                u1, v1 = stroke[i]
                u2, v2 = stroke[i + 1]
                
                x1 = int(u1 * tex_width) + texture_rect.x
                y1 = int(v1 * tex_height) + texture_rect.y
                x2 = int(u2 * tex_width) + texture_rect.x
                y2 = int(v2 * tex_height) + texture_rect.y
                
                width = max(2, int(drawing_system.brush_size * min(tex_width, tex_height)))
                pygame.draw.line(surface, drawing_system.draw_color, (x1, y1), (x2, y2), width)
                
                # Draw circles at each point for smoother line
                pygame.draw.circle(surface, drawing_system.draw_color, (x1, y1), width // 2)
                pygame.draw.circle(surface, drawing_system.draw_color, (x2, y2), width // 2)
