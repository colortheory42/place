"""
Variable Ceiling Height System
===============================
Provides deterministic ceiling heights based on world position and zone properties.
"""

import random
from config import get_scaled_wall_height


def get_ceiling_height_at_position(x, z, world):
    """
    Get ceiling height at a specific world position.
    Returns a height value that can vary based on zone and local position.
    
    Args:
        x: World X coordinate
        z: World Z coordinate
        world: World instance with zone system
        
    Returns:
        float: Ceiling height in world units
    """
    # Get zone properties
    zone_x, zone_z = world.get_zone_at(x, z)
    props = world.get_zone_properties(zone_x, zone_z)
    
    # Get zone-specific ceiling range
    min_ceiling = props.get('min_ceiling', 100)
    max_ceiling = props.get('max_ceiling', 120)
    
    # Base height from config (for compatibility)
    base_height = get_scaled_wall_height()
    
    # Use zone ceiling if it's significantly different from base
    if max_ceiling > base_height * 1.5:
        # In mega-scale zones, use the zone-specific range
        # Deterministic height based on position
        height_seed = int(x * 7919 + z * 6577 + world.world_seed)
        rng = random.Random(height_seed)
        
        # Interpolate between min and max
        t = rng.random()
        return min_ceiling + (max_ceiling - min_ceiling) * t
    else:
        # In normal zones, use base height with variance
        variance = props.get('ceiling_height_var', 8)
        height_seed = int(x * 7919 + z * 6577 + world.world_seed)
        rng = random.Random(height_seed)
        offset = rng.uniform(-variance, variance)
        return base_height + offset


def get_room_size_at_position(x, z, world):
    """
    Get the room/grid size at a specific world position.
    
    Args:
        x: World X coordinate
        z: World Z coordinate
        world: World instance with zone system
        
    Returns:
        int: Room size (grid spacing) in world units
    """
    zone_x, zone_z = world.get_zone_at(x, z)
    props = world.get_zone_properties(zone_x, zone_z)
    return props.get('room_size', 400)


def snap_to_zone_grid(x, z, world):
    """
    Snap a world coordinate to the nearest zone-aware grid point.
    
    Args:
        x, z: World coordinates
        world: World instance
        
    Returns:
        tuple: (grid_x, grid_z) snapped to zone's grid
    """
    room_size = get_room_size_at_position(x, z, world)
    grid_x = int(x // room_size) * room_size
    grid_z = int(z // room_size) * room_size
    return (grid_x, grid_z)


def get_zone_info_string(x, z, world):
    """
    Get human-readable zone info for debugging/UI.
    
    Returns:
        str: Zone type and properties
    """
    zone_x, zone_z = world.get_zone_at(x, z)
    props = world.get_zone_properties(zone_x, zone_z)
    
    zone_type = props.get('zone_type', 'unknown')
    room_size = props.get('room_size', 400)
    scale = props.get('scale_multiplier', 1.0)
    min_c = props.get('min_ceiling', 100)
    max_c = props.get('max_ceiling', 120)
    
    return f"{zone_type.upper()} | {room_size}u rooms | {scale}x scale | ceiling: {min_c}-{max_c}"
