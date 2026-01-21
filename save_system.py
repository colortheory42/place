"""
Save/load system.
Handles game state persistence to JSON files.
"""

import os
import json
from datetime import datetime
from config import SAVE_DIR


class SaveSystem:
    """Manages game saves and loads."""

    @staticmethod
    def ensure_save_dir():
        """Create save directory if it doesn't exist."""
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

    @staticmethod
    def get_save_path(slot=1):
        """Get filepath for a save slot."""
        SaveSystem.ensure_save_dir()
        return os.path.join(SAVE_DIR, f"save_slot_{slot}.json")

    @staticmethod
    def save_game(engine, camcorder=None, slot=1):
        """Save game state to JSON file."""
        save_data = {
            'version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'player': {
                'x': engine.x,
                'y': engine.y,
                'z': engine.z,
                'pitch': engine.pitch,
                'yaw': engine.yaw
            },
            'world': {
                'seed': engine.world_seed,
                'destroyed_walls': [list(wall) for wall in engine.destroyed_walls]
            },
            'drawings': engine.drawing_system.get_state_for_save(),
            'smiles': engine.smile_manager.get_state_for_save(),
            'stats': {
                'play_time': engine.play_time
            }
        }
        
        # Add wall_damage if it exists (for engines that support progressive damage)
        if hasattr(engine, 'wall_damage'):
            save_data['world']['wall_damage'] = {
                f"{k[0]},{k[1]},{k[2]}": v for k, v in engine.wall_damage.items()
            }
        
        # Add destroyed_pillars if it exists
        if hasattr(engine, 'destroyed_pillars'):
            save_data['world']['destroyed_pillars'] = [list(pillar) for pillar in engine.destroyed_pillars]
        
        # Add camcorder state if provided
        if camcorder:
            save_data['camcorder'] = camcorder.get_state_for_save()

        save_path = SaveSystem.get_save_path(slot)
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"Game saved to slot {slot}")
        return True

    @staticmethod
    def load_game(slot=1):
        """Load game state from JSON file.
        
        Note: The engine's load_from_save() method must handle:
        - world.destroyed_walls (list of destroyed wall keys)
        - world.wall_damage (dict of wall keys to damage values)
        - world.destroyed_pillars (list of destroyed pillar keys)
        """
        save_path = SaveSystem.get_save_path(slot)

        if not os.path.exists(save_path):
            print(f"No save found in slot {slot}")
            return None

        try:
            with open(save_path, 'r') as f:
                save_data = json.load(f)
            print(f"Game loaded from slot {slot}")
            return save_data
        except Exception as e:
            print(f"Error loading save: {e}")
            return None

    @staticmethod
    def list_saves():
        """List all available save slots."""
        SaveSystem.ensure_save_dir()
        saves = []
        for i in range(1, 6):  # Check slots 1-5
            save_path = SaveSystem.get_save_path(i)
            if os.path.exists(save_path):
                try:
                    with open(save_path, 'r') as f:
                        data = json.load(f)
                    saves.append({
                        'slot': i,
                        'timestamp': data.get('timestamp', 'Unknown'),
                        'play_time': data.get('stats', {}).get('play_time', 0)
                    })
                except:
                    pass
        return saves
