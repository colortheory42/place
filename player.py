"""
Player system.
Handles movement state, input processing, jumping, and crouching.
"""

import math
import pygame
from config import (
    ROTATION_SPEED, WALK_SPEED, RUN_SPEED, CROUCH_SPEED,
    CAMERA_HEIGHT_STAND, CAMERA_HEIGHT_CROUCH, CROUCH_TRANSITION_SPEED,
    JUMP_STRENGTH, GRAVITY
)
from collision import CollisionSystem


class Player:
    """Player state and movement."""

    def __init__(self):
        # Position
        self.x = 200
        self.y = CAMERA_HEIGHT_STAND
        self.z = 200
        self.target_y = CAMERA_HEIGHT_STAND

        # Rotation
        self.pitch = 0
        self.yaw = 0

        # Movement states
        self.is_moving = False
        self.is_rotating = False
        self.is_running = False
        self.is_crouching = False
        self.crouch_key_pressed = False

        # Jumping
        self.is_jumping = False
        self.jump_velocity = 0
        self.on_ground = True

        # Input
        self.mouse_look = False
        
        # Collision system (set by engine after world is created)
        self.collision_system = None
        
        # Collision tracking for sound
        self.collision_callback = None  # Function to call when collision occurs
        self.was_colliding = False  # Track if we were colliding last frame

    def toggle_mouse(self):
        """Toggle mouse look mode."""
        self.mouse_look = not self.mouse_look
        pygame.mouse.set_visible(not self.mouse_look)
        pygame.event.set_grab(self.mouse_look)

    def update(self, dt, keys, mouse_rel, collision_checker, current_time=0.0):
        """
        Update player state based on input.
        collision_checker: function(x, z) -> bool that returns True if collision
        current_time: game time for collision sound cooldown
        """
        # Mouse look
        if self.mouse_look and mouse_rel:
            dx, dy = mouse_rel
            self.yaw += dx * 0.002
            self.pitch -= dy * 0.002

        # Keyboard rotation
        self.is_rotating = False
        rot = ROTATION_SPEED * dt
        if keys[pygame.K_j]:
            self.yaw -= rot
            self.is_rotating = True
        if keys[pygame.K_l]:
            self.yaw += rot
            self.is_rotating = True

        # Clamp pitch
        self.pitch = max(-math.pi / 2 + 0.01, min(math.pi / 2 - 0.01, self.pitch))

        # Crouch toggle
        crouch_key_down = keys[pygame.K_c]
        if crouch_key_down and not self.crouch_key_pressed:
            self.is_crouching = not self.is_crouching
            if self.is_crouching:
                self.target_y = CAMERA_HEIGHT_CROUCH
            else:
                self.target_y = CAMERA_HEIGHT_STAND
        self.crouch_key_pressed = crouch_key_down

        # Jump
        if keys[pygame.K_SPACE] and self.on_ground and not self.is_crouching:
            self.is_jumping = True
            self.jump_velocity = JUMP_STRENGTH
            self.on_ground = False

        # Movement speed
        if keys[pygame.K_LSHIFT] and not self.is_crouching:
            self.is_running = True
            speed = RUN_SPEED * dt
        elif self.is_crouching:
            self.is_running = False
            speed = CROUCH_SPEED * dt
        else:
            self.is_running = False
            speed = WALK_SPEED * dt

        # Calculate movement direction
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)

        move_x = 0
        move_z = 0
        self.is_moving = False

        if keys[pygame.K_w] or keys[pygame.K_UP]:
            move_x += sy * speed
            move_z += cy * speed
            self.is_moving = True
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            move_x -= sy * speed
            move_z -= cy * speed
            self.is_moving = True
        if keys[pygame.K_a]:
            move_x -= cy * speed
            move_z += sy * speed
            self.is_moving = True
        if keys[pygame.K_d]:
            move_x += cy * speed
            move_z -= sy * speed
            self.is_moving = True

        # Apply movement with collision
        if move_x != 0 or move_z != 0:
            if self.collision_system:
                # Use precise sliding collision
                from_pos = (self.x, self.z)
                to_pos = (self.x + move_x, self.z + move_z)
                final_x, final_z, collided = self.collision_system.resolve_collision(from_pos, to_pos)
                self.x = final_x
                self.z = final_z
                
                # Only trigger sound on NEW collision (state transition)
                if collided and not self.was_colliding and self.collision_callback:
                    # Calculate collision intensity based on attempted movement
                    attempted_dist = math.sqrt(move_x**2 + move_z**2)
                    intensity = min(1.0, attempted_dist / 5.0)  # Normalize to 0-1
                    self.collision_callback(intensity)
                
                # Update collision state for next frame
                self.was_colliding = collided
            else:
                # Fallback to old system if collision_system not set
                if not collision_checker(self.x + move_x, self.z + move_z):
                    self.x += move_x
                    self.z += move_z
                else:
                    # Try sliding along walls
                    if not collision_checker(self.x + move_x, self.z):
                        self.x += move_x
                    if not collision_checker(self.x, self.z + move_z):
                        self.z += move_z
        else:
            # Not moving - reset collision state
            self.was_colliding = False

        # Smooth camera height (only when grounded)
        if self.on_ground and not self.is_jumping:
            if abs(self.y - self.target_y) > 0.1:
                self.y += (self.target_y - self.y) * CROUCH_TRANSITION_SPEED * dt
            else:
                self.y = self.target_y

        # Jump physics
        if self.is_jumping or not self.on_ground:
            self.jump_velocity -= GRAVITY * dt
            self.y += self.jump_velocity * dt

            # Land when falling below target height
            if self.jump_velocity < 0 and self.y <= self.target_y:
                self.y = self.target_y
                self.jump_velocity = 0
                self.is_jumping = False
                self.on_ground = True

    def get_state_for_save(self):
        """Get player state for saving."""
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'pitch': self.pitch,
            'yaw': self.yaw
        }

    def load_state(self, data):
        """Load player state from save data."""
        self.x = data.get('x', self.x)
        self.y = data.get('y', self.y)
        self.z = data.get('z', self.z)
        self.pitch = data.get('pitch', self.pitch)
        self.yaw = data.get('yaw', self.yaw)
