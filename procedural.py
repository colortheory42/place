"""
Procedural zone generation - MEGA-ARCHITECTURE VERSION.
Different zone types with MASSIVELY varying scales for spatial variety.
"""


class ProceduralZone:
    """Represents a procedural zone with specific characteristics."""

    ZONE_TYPES = {
        # === HUMAN-SCALE ZONES ===
        'normal': {
            'pillar_density': 0.35,
            'wall_chance': 0.25,
            'ceiling_height_var': 8,
            'color_tint': (1.0, 1.0, 1.0),
            'scale_multiplier': 1.0,        # Base scale
            'room_size': 400,               # Standard 400x400 rooms
            'min_ceiling': 100,             # Minimum ceiling height
            'max_ceiling': 120,             # Maximum ceiling height
        },
        'dense': {
            'pillar_density': 0.55,
            'wall_chance': 0.4,
            'ceiling_height_var': 5,
            'color_tint': (0.95, 0.95, 0.85),
            'scale_multiplier': 1.0,
            'room_size': 400,
            'min_ceiling': 95,
            'max_ceiling': 110,
        },
        'sparse': {
            'pillar_density': 0.15,
            'wall_chance': 0.1,
            'ceiling_height_var': 18,
            'color_tint': (1.05, 1.05, 1.15),
            'scale_multiplier': 1.0,
            'room_size': 400,
            'min_ceiling': 100,
            'max_ceiling': 140,
        },
        'maze': {
            'pillar_density': 0.7,
            'wall_chance': 0.6,
            'ceiling_height_var': 3,
            'color_tint': (0.9, 0.9, 0.8),
            'scale_multiplier': 1.0,
            'room_size': 400,
            'min_ceiling': 90,
            'max_ceiling': 100,
        },
        'open': {
            'pillar_density': 0.08,
            'wall_chance': 0.05,
            'ceiling_height_var': 30,
            'color_tint': (1.1, 1.1, 1.2),
            'scale_multiplier': 1.5,        # Slightly bigger rooms
            'room_size': 600,
            'min_ceiling': 100,
            'max_ceiling': 160,
        },
        
        # === MEGA-SCALE ZONES ===
        'atrium': {
            'pillar_density': 0.05,         # Very few pillars
            'wall_chance': 0.02,            # Almost no internal walls
            'ceiling_height_var': 100,      # High variance
            'color_tint': (1.05, 1.05, 1.15),
            'scale_multiplier': 6.0,        # 6x scale
            'room_size': 2400,              # 400 * 6 = massive rooms
            'min_ceiling': 300,             # 3x normal ceiling
            'max_ceiling': 500,             # Up to 5x
        },
        'coliseum': {
            'pillar_density': 0.0,          # No pillars
            'wall_chance': 0.0,             # No walls - pure open space
            'ceiling_height_var': 200,      # Massive variance
            'color_tint': (1.1, 1.1, 1.25),
            'scale_multiplier': 10.0,       # 10x scale!
            'room_size': 4000,              # 400 * 10 = stadium-sized
            'min_ceiling': 400,             # 4x normal
            'max_ceiling': 800,             # Up to 8x - cathedral ceiling
        },
        'courtyard': {
            'pillar_density': 0.03,         # Almost no pillars
            'wall_chance': 0.05,            # Sparse walls
            'ceiling_height_var': 150,
            'color_tint': (1.2, 1.2, 1.3),  # Brighter - outdoor feeling
            'scale_multiplier': 8.0,        # 8x scale
            'room_size': 3200,              # 400 * 8
            'min_ceiling': 350,             # Very high
            'max_ceiling': 600,
        },
        'skyscraper_base': {
            'pillar_density': 0.25,         # Some structural pillars
            'wall_chance': 0.15,            # Some walls but open plan
            'ceiling_height_var': 80,
            'color_tint': (0.95, 0.95, 1.05),
            'scale_multiplier': 5.0,        # 5x scale
            'room_size': 2000,              # 400 * 5 = office building scale
            'min_ceiling': 250,             # High office ceilings
            'max_ceiling': 400,
        },
        'grand_hall': {
            'pillar_density': 0.10,         # Few pillars
            'wall_chance': 0.08,            # Minimal walls
            'ceiling_height_var': 120,
            'color_tint': (1.0, 1.0, 1.1),
            'scale_multiplier': 7.0,        # 7x scale
            'room_size': 2800,              # 400 * 7
            'min_ceiling': 300,
            'max_ceiling': 500,
        },
        'cathedral': {
            'pillar_density': 0.15,         # Some pillars (structural)
            'wall_chance': 0.10,
            'ceiling_height_var': 180,
            'color_tint': (1.05, 1.05, 1.15),
            'scale_multiplier': 9.0,        # 9x scale
            'room_size': 3600,              # 400 * 9
            'min_ceiling': 450,             # Very tall
            'max_ceiling': 700,
        },
        'warehouse': {
            'pillar_density': 0.20,         # Support columns
            'wall_chance': 0.05,
            'ceiling_height_var': 60,
            'color_tint': (0.95, 0.95, 0.9),
            'scale_multiplier': 4.0,        # 4x scale
            'room_size': 1600,              # 400 * 4 = warehouse scale
            'min_ceiling': 180,
            'max_ceiling': 280,
        },
    }

    @staticmethod
    def get_zone_type(zone_x, zone_z, seed=12345):
        """Deterministic zone type selection based on coordinates."""
        hash_val = (zone_x * 73856093 + zone_z * 19349663 + seed * 83492791) & 0x7fffffff
        zone_index = hash_val % len(ProceduralZone.ZONE_TYPES)
        return list(ProceduralZone.ZONE_TYPES.keys())[zone_index]

    @staticmethod
    def get_zone_properties(zone_x, zone_z, seed=12345):
        """Get properties for a specific zone."""
        zone_type = ProceduralZone.get_zone_type(zone_x, zone_z, seed)
        props = ProceduralZone.ZONE_TYPES[zone_type].copy()

        # Add decay chance based on zone type
        decay_chances = {
            'normal': 0.20,
            'dense': 0.20,
            'sparse': 0.20,
            'maze': 0.20,
            'open': 0.15,
            # Mega-scale zones have less decay
            'atrium': 0.05,
            'coliseum': 0.02,
            'courtyard': 0.05,
            'skyscraper_base': 0.10,
            'grand_hall': 0.08,
            'cathedral': 0.05,
            'warehouse': 0.12,
        }
        props['decay_chance'] = decay_chances.get(zone_type, 0.10)
        
        # Add zone name for debugging
        props['zone_type'] = zone_type

        return props
