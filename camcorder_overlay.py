"""
Camcorder Overlay System
========================
Authentic VHS camcorder interface with:
- Battery life indicator (30 minutes of footage)
- Recording indicator (REC dot)
- Timestamp
- Camera settings display
- VHS scan lines and artifacts
- Game ends when battery dies
"""

import pygame
import math
import random
from datetime import datetime, timedelta


class CamcorderOverlay:
    """
    VHS Camcorder overlay that mimics 1990s handheld camcorder UI.
    Includes battery life that depletes over 30 minutes of recording.
    Game ends when battery dies.
    """
    
    def __init__(self, battery_minutes=30.0):
        self.enabled = True  # Always enabled, no toggle
        self.battery_minutes = battery_minutes
        self.battery_seconds = battery_minutes * 60.0
        self.max_battery_seconds = self.battery_seconds
        
        # Recording state - always recording
        self.is_recording = True
        self.recording_time = 0.0
        self.rec_blink_timer = 0.0
        self.rec_visible = True
        
        # Track recording start
        self.has_started_recording = True
        self.tape_start_date = datetime.now()
        
        # VHS effects
        self.scanline_offset = 0.0
        self.tracking_glitch_timer = 0.0
        self.tracking_glitch_active = False
        self.tracking_glitch_offset = 0
        
        # Vignette/corner fade
        self.vignette_intensity = 0.3
        
        # Colors (VHS white/red)
        self.text_color = (255, 255, 255)
        self.rec_color = (255, 50, 50)
        self.battery_low_color = (255, 255, 0)
        self.battery_critical_color = (255, 50, 50)
        
        print("📹 Camcorder recording started (always on)")
        print(f"   Battery life: {battery_minutes:.1f} minutes")
        
    def update(self, dt):
        """Update overlay state. Battery always depletes."""
        # Deplete battery
        if self.battery_seconds > 0:
            self.battery_seconds -= dt
            self.battery_seconds = max(0, self.battery_seconds)
        
        # Update recording time
        self.recording_time += dt
        
        # Blink REC indicator
        self.rec_blink_timer += dt
        if self.rec_blink_timer >= 0.5:  # Blink every 0.5s
            self.rec_visible = not self.rec_visible
            self.rec_blink_timer = 0.0
        
        # Scanline animation
        self.scanline_offset += dt * 100.0
        
        # Random tracking glitches (VHS artifact)
        if not self.tracking_glitch_active:
            if random.random() < 0.002:
                self.tracking_glitch_active = True
                self.tracking_glitch_timer = 0.0
                self.tracking_glitch_offset = random.randint(-5, 5)
        else:
            self.tracking_glitch_timer += dt
            if self.tracking_glitch_timer >= 0.08:
                self.tracking_glitch_active = False
                self.tracking_glitch_offset = 0
    
    def render(self, surface):
        """Render camcorder overlay on screen (always visible)."""
        width, height = surface.get_size()
        
        # === VHS EFFECTS ===
        self._render_vhs_effects(surface, width, height)
        
        # === CORNER VIGNETTE ===
        self._render_vignette(surface, width, height)
        
        # === TOP LEFT: REC + TIMESTAMP ===
        self._render_recording_indicator(surface)
        self._render_timestamp(surface)
        
        # === TOP RIGHT: BATTERY ===
        self._render_battery(surface, width)
        
        # === BOTTOM LEFT: CAMERA INFO ===
        self._render_camera_info(surface, height)
        
        # === BOTTOM RIGHT: TAPE COUNTER ===
        self._render_tape_counter(surface, width, height)
        
        # === CENTER BOTTOM: VIEWFINDER GUIDES ===
        self._render_viewfinder_guides(surface, width, height)
    
    def _render_vhs_effects(self, surface, width, height):
        """Render VHS scanlines and artifacts."""
        # Horizontal scanlines
        scanline_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        for y in range(0, height, 2):
            y_offset = int((y + self.scanline_offset) % 4)
            alpha = 15 if y_offset < 2 else 0
            pygame.draw.line(scanline_surface, (0, 0, 0, alpha), (0, y), (width, y), 1)
        
        surface.blit(scanline_surface, (0, 0))
        
        # Tracking glitch (horizontal displacement)
        if self.tracking_glitch_active:
            glitch_y = int(height * random.uniform(0.3, 0.7))
            glitch_height = random.randint(5, 20)
            
            try:
                glitch_rect = pygame.Rect(0, glitch_y, width, glitch_height)
                glitch_surf = surface.subsurface(glitch_rect).copy()
                pygame.draw.rect(surface, (0, 0, 0), glitch_rect)
                surface.blit(glitch_surf, (self.tracking_glitch_offset, glitch_y))
            except:
                pass
    
    def _render_vignette(self, surface, width, height):
        """Render corner vignette."""
        vignette = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Corner darkening
        for x in range(width // 4):
            for y in range(height // 6):
                dist = math.sqrt((x - 0)**2 + (y - 0)**2)
                alpha = int(self.vignette_intensity * 255 * (1 - dist / (width // 4)))
                if alpha > 0:
                    vignette.set_at((x, y), (0, 0, 0, alpha))
                    vignette.set_at((width - x - 1, y), (0, 0, 0, alpha))
        
        for x in range(width // 4):
            for y in range(height // 6):
                dist = math.sqrt((x - 0)**2 + (y - 0)**2)
                alpha = int(self.vignette_intensity * 255 * (1 - dist / (width // 4)))
                if alpha > 0:
                    vignette.set_at((x, height - y - 1), (0, 0, 0, alpha))
                    vignette.set_at((width - x - 1, height - y - 1), (0, 0, 0, alpha))
        
        surface.blit(vignette, (0, 0))
    
    def _render_recording_indicator(self, surface):
        """Render blinking REC indicator."""
        if not self.rec_visible:
            return
        
        font = pygame.font.SysFont("arial", 16, bold=True)
        
        # Draw red circle
        pygame.draw.circle(surface, self.rec_color, (25, 25), 8)
        
        # Draw REC text
        rec_text = font.render("REC", True, self.rec_color)
        surface.blit(rec_text, (40, 17))
    
    def _render_timestamp(self, surface):
        """Render date/time stamp."""
        font = pygame.font.SysFont("arial", 14, bold=True)
        
        # Calculate current tape time
        current_time = self.tape_start_date + timedelta(seconds=self.recording_time)
        
        # Format: JAN 16 2026  2:32:45 PM
        date_str = current_time.strftime("%b %d %Y").upper()
        time_str = current_time.strftime("%I:%M:%S %p")
        
        # Draw with shadow for readability
        shadow_color = (0, 0, 0)
        
        date_text = font.render(date_str, True, shadow_color)
        surface.blit(date_text, (26, 46))
        date_text = font.render(date_str, True, self.text_color)
        surface.blit(date_text, (25, 45))
        
        time_text = font.render(time_str, True, shadow_color)
        surface.blit(time_text, (26, 66))
        time_text = font.render(time_str, True, self.text_color)
        surface.blit(time_text, (25, 65))
    
    def _render_battery(self, surface, width):
        """Render battery indicator."""
        font = pygame.font.SysFont("arial", 14, bold=True)
        
        # Calculate battery percentage
        battery_percent = (self.battery_seconds / self.max_battery_seconds) * 100.0
        
        # Choose color based on battery level
        if battery_percent > 20:
            color = self.text_color
        elif battery_percent > 10:
            color = self.battery_low_color
        else:
            color = self.battery_critical_color
        
        # Draw battery icon
        battery_x = width - 100
        battery_y = 20
        battery_width = 50
        battery_height = 20
        
        # Battery body
        pygame.draw.rect(surface, color, (battery_x, battery_y, battery_width, battery_height), 2)
        
        # Battery terminal
        pygame.draw.rect(surface, color, (battery_x + battery_width, battery_y + 6, 4, 8))
        
        # Battery fill
        fill_width = int((battery_width - 4) * (battery_percent / 100.0))
        if fill_width > 0:
            pygame.draw.rect(surface, color, (battery_x + 2, battery_y + 2, fill_width, battery_height - 4))
        
        # Battery percentage text
        battery_text = font.render(f"{int(battery_percent)}%", True, color)
        surface.blit(battery_text, (battery_x, battery_y + 25))
        
        # Time remaining
        minutes_remaining = int(self.battery_seconds // 60)
        seconds_remaining = int(self.battery_seconds % 60)
        time_text = font.render(f"{minutes_remaining:02d}:{seconds_remaining:02d}", True, color)
        surface.blit(time_text, (battery_x - 5, battery_y + 45))
    
    def _render_camera_info(self, surface, height):
        """Render camera settings info (bottom left)."""
        font = pygame.font.SysFont("arial", 12, bold=True)
        
        y_base = height - 80
        
        info_lines = [
            "SP MODE",
            "AUTO FOCUS",
            "WHITE BAL: AUTO",
        ]
        
        for i, line in enumerate(info_lines):
            text = font.render(line, True, (0, 0, 0))
            surface.blit(text, (21, y_base + i * 16))
            text = font.render(line, True, self.text_color)
            surface.blit(text, (20, y_base + i * 16 - 1))
    
    def _render_tape_counter(self, surface, width, height):
        """Render tape counter (bottom right)."""
        font = pygame.font.SysFont("courier", 18, bold=True)
        
        # Convert recording time to tape counter format
        hours = int(self.recording_time // 3600)
        minutes = int((self.recording_time % 3600) // 60)
        seconds = int(self.recording_time % 60)
        
        counter_text = f"{hours:01d}:{minutes:02d}:{seconds:02d}"
        
        # Draw with background box
        text_surf = font.render(counter_text, True, self.text_color)
        text_width = text_surf.get_width()
        text_height = text_surf.get_height()
        
        box_x = width - text_width - 30
        box_y = height - text_height - 30
        
        # Black background box
        pygame.draw.rect(surface, (0, 0, 0, 180),
                        (box_x - 5, box_y - 3, text_width + 10, text_height + 6))
        
        # Text
        surface.blit(text_surf, (box_x, box_y))
    
    def _render_viewfinder_guides(self, surface, width, height):
        """Render center viewfinder guides."""
        center_x = width // 2
        center_y = height // 2
        guide_size = 40
        guide_thickness = 1
        
        color = (255, 255, 255, 80)
        
        # Top-left corner
        pygame.draw.line(surface, color,
                        (center_x - guide_size, center_y - guide_size),
                        (center_x - guide_size + 15, center_y - guide_size),
                        guide_thickness)
        pygame.draw.line(surface, color,
                        (center_x - guide_size, center_y - guide_size),
                        (center_x - guide_size, center_y - guide_size + 15),
                        guide_thickness)
        
        # Top-right corner
        pygame.draw.line(surface, color,
                        (center_x + guide_size, center_y - guide_size),
                        (center_x + guide_size - 15, center_y - guide_size),
                        guide_thickness)
        pygame.draw.line(surface, color,
                        (center_x + guide_size, center_y - guide_size),
                        (center_x + guide_size, center_y - guide_size + 15),
                        guide_thickness)
        
        # Bottom-left corner
        pygame.draw.line(surface, color,
                        (center_x - guide_size, center_y + guide_size),
                        (center_x - guide_size + 15, center_y + guide_size),
                        guide_thickness)
        pygame.draw.line(surface, color,
                        (center_x - guide_size, center_y + guide_size),
                        (center_x - guide_size, center_y + guide_size - 15),
                        guide_thickness)
        
        # Bottom-right corner
        pygame.draw.line(surface, color,
                        (center_x + guide_size, center_y + guide_size),
                        (center_x + guide_size - 15, center_y + guide_size),
                        guide_thickness)
        pygame.draw.line(surface, color,
                        (center_x + guide_size, center_y + guide_size),
                        (center_x + guide_size, center_y + guide_size - 15),
                        guide_thickness)
    
    def is_battery_dead(self):
        """Check if battery is depleted."""
        return self.battery_seconds <= 0
    
    def get_battery_percent(self):
        """Get current battery percentage."""
        return (self.battery_seconds / self.max_battery_seconds) * 100.0
    
    def recharge_battery(self, percent=100.0):
        """Recharge battery (for testing or gameplay mechanic)."""
        self.battery_seconds = (percent / 100.0) * self.max_battery_seconds
        print(f"🔋 Battery recharged to {percent:.0f}%")
    
    def reset_tape_counter(self):
        """Reset the tape counter to 0:00:00."""
        self.recording_time = 0.0
        print("⏹️  Tape counter reset")
    
    def get_state_for_save(self):
        """Get overlay state for saving."""
        return {
            'enabled': self.enabled,
            'battery_seconds': self.battery_seconds,
            'recording_time': self.recording_time,
            'has_started_recording': self.has_started_recording,
            'tape_start_date': self.tape_start_date.isoformat() if self.tape_start_date else None
        }
    
    def load_state(self, data):
        """Load overlay state from save."""
        self.enabled = True  # Always enabled
        self.battery_seconds = data.get('battery_seconds', self.max_battery_seconds)
        self.recording_time = data.get('recording_time', 0.0)
        self.has_started_recording = data.get('has_started_recording', True)
        
        # Restore tape start date
        tape_start_str = data.get('tape_start_date')
        if tape_start_str:
            self.tape_start_date = datetime.fromisoformat(tape_start_str)
        else:
            self.tape_start_date = datetime.now()
