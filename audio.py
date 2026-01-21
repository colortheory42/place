"""
Procedural audio generation.
Creates all sound effects at runtime using NumPy waveform synthesis.
"""

import numpy as np
import pygame
from config import SAMPLE_RATE

def low_pass(signal, kernel_size):
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(signal, kernel, mode="same")

def generate_backrooms_hum():
    """Generate ambient droning hum."""
    duration = 10
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)

    drone = 0.15 * np.sin(2 * np.pi * 60 * t)
    drone += 0.12 * np.sin(2 * np.pi * 55 * t)
    drone += 0.10 * np.sin(2 * np.pi * 40 * t)
    drone += 0.08 * np.sin(2 * np.pi * 120 * t)
    drone += 0.05 * np.sin(2 * np.pi * 180 * t)

    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)
    drone *= modulation
    noise = np.random.normal(0, 0.02, samples)
    drone += noise

    drone = drone / np.max(np.abs(drone)) * 0.6
    audio = np.array(drone * 32767, dtype=np.int16)
    stereo_audio = np.column_stack((audio, audio))

    return pygame.sndarray.make_sound(stereo_audio)


def generate_footstep_sound():
    """Generate ambient footstep sound (distant)."""
    duration = 0.3
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)

    impact = np.exp(-t * 20) * np.sin(2 * np.pi * 80 * t)
    impact += np.exp(-t * 15) * np.sin(2 * np.pi * 120 * t) * 0.5
    reverb = np.exp(-t * 5) * np.random.normal(0, 0.1, samples)

    sound = impact + reverb * 0.3
    sound = sound / np.max(np.abs(sound)) * 0.7

    audio = np.array(sound * 32767, dtype=np.int16)
    stereo_audio = np.column_stack((audio, audio))

    return pygame.sndarray.make_sound(stereo_audio)


def generate_player_footstep_sound(turn_factor=1.0):
    """Generate deep, soft carpet footstep (pressure into fabric).
    turn_factor: 0.0 (straight) → 1.0 (hard turn)
    """
    duration = 0.14
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)

    # Clamp for safety
    turn_factor = max(0.0, min(turn_factor, 1.0))

    # Fabric noise (very gentle)
    noise = np.random.uniform(-1, 1, samples)

    # Soft envelope: slow rise, smooth release
    attack = int(0.35 * samples)
    decay = samples - attack
    envelope = np.concatenate([
        np.linspace(0, 1, attack),
        np.linspace(1, 0, decay)
    ])

    # Directional carpet absorption (fiber shear when turning)
    kernel_size = int(65 + turn_factor * 25)
    muffled = low_pass(noise, kernel_size=kernel_size)

    # Deep pressure "crush" (felt, not heard)
    bass = (
        0.08 * np.sin(2 * np.pi * 38 * t) +
        0.04 * np.sin(2 * np.pi * 28 * t)
    ) * np.exp(-t * 18)

    sound = muffled * envelope * 0.3 + bass

    # Extremely conservative output level
    sound = sound / np.max(np.abs(sound)) * 0.32

    # Subtle stereo smear (body rotation, not panning)
    left = sound * (1.0 - turn_factor * 0.08)
    right = sound * (1.0 + turn_factor * 0.08)

    audio_l = np.array(left * 32767, dtype=np.int16)
    audio_r = np.array(right * 32767, dtype=np.int16)
    stereo_audio = np.column_stack((audio_l, audio_r))

    return pygame.sndarray.make_sound(stereo_audio)


def generate_crouch_footstep_sound(turn_factor=1.0):
    """Generate ultra-soft, deep carpet crouch footstep (slow pressure).
    turn_factor: 0.0 (straight) → 1.0 (hard turn)
    """
    duration = 0.18
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)

    turn_factor = max(0.0, min(turn_factor, 1.0))

    # Very gentle fabric noise
    noise = np.random.uniform(-1, 1, samples)

    # Extra-slow, smooth envelope (no perceptible onset)
    attack = int(0.45 * samples)
    decay = samples - attack
    envelope = np.concatenate([
        np.linspace(0, 1, attack),
        np.linspace(1, 0, decay)
    ])

    # Strong absorption — more smear when turning
    kernel_size = int(80 + turn_factor * 30)
    muffled = low_pass(noise, kernel_size=kernel_size)

    # Deep, slow pressure (almost sub-audible)
    bass = (
        0.06 * np.sin(2 * np.pi * 32 * t) +
        0.03 * np.sin(2 * np.pi * 24 * t)
    ) * np.exp(-t * 14)

    sound = muffled * envelope * 0.25 + bass

    # Very low output level
    sound = sound / np.max(np.abs(sound)) * 0.24

    # Extremely subtle stereo drift
    left = sound * (1.0 - turn_factor * 0.06)
    right = sound * (1.0 + turn_factor * 0.06)

    audio_l = np.array(left * 32767, dtype=np.int16)
    audio_r = np.array(right * 32767, dtype=np.int16)
    stereo_audio = np.column_stack((audio_l, audio_r))

    return pygame.sndarray.make_sound(stereo_audio)


def generate_electrical_buzz():
    """Generate electrical buzzing sound."""
    duration = 1.5
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)

    buzz = 0.2 * np.sin(2 * np.pi * 120 * t)
    buzz += 0.15 * np.sin(2 * np.pi * 240 * t)
    mod = np.sin(2 * np.pi * 8 * t) * 0.5 + 0.5
    buzz *= mod

    buzz = buzz / np.max(np.abs(buzz)) * 0.3
    audio = np.array(buzz * 32767, dtype=np.int16)
    stereo_audio = np.column_stack((audio, audio))

    return pygame.sndarray.make_sound(stereo_audio)


def generate_destroy_sound():
    """Generate destruction sound for walls breaking."""
    duration = 1.0
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)

    # Big impact
    impact = np.exp(-t * 8) * np.sin(2 * np.pi * 80 * t)
    impact += np.exp(-t * 10) * np.sin(2 * np.pi * 120 * t) * 0.8

    # Crumbling/debris
    crumble = np.random.normal(0, 0.4, samples) * np.exp(-t * 4)

    sound = impact + crumble * 0.7
    sound = sound / np.max(np.abs(sound)) * 0.8

    audio = np.array(sound * 32767, dtype=np.int16)
    stereo_audio = np.column_stack((audio, audio))

    return pygame.sndarray.make_sound(stereo_audio)


def generate_crack_sound():
    """Generate cracking/impact sound for wall damage."""
    duration = 0.25
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)

    # Sharp crack
    crack = np.exp(-t * 40) * np.sin(2 * np.pi * 200 * t)
    crack += np.exp(-t * 30) * np.sin(2 * np.pi * 350 * t) * 0.5

    # Gritty crumble
    noise = np.random.normal(0, 0.3, samples) * np.exp(-t * 15)

    sound = crack * 0.7 + noise * 0.3
    sound = sound / np.max(np.abs(sound)) * 0.5

    audio = np.array(sound * 32767, dtype=np.int16)
    stereo_audio = np.column_stack((audio, audio))

    return pygame.sndarray.make_sound(stereo_audio)


def generate_fracture_sound():
    """Generate heavier fracturing sound for major wall damage."""
    duration = 0.4
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)

    # Multiple cracks layered
    crack1 = np.exp(-t * 25) * np.sin(2 * np.pi * 150 * t)
    crack2 = np.exp(-t * 20) * np.sin(2 * np.pi * 250 * t) * 0.6
    crack3 = np.exp(-t * 35) * np.sin(2 * np.pi * 400 * t) * 0.3

    # Debris/crumble
    crumble = np.random.normal(0, 0.35, samples) * np.exp(-t * 8)

    sound = crack1 + crack2 + crack3 + crumble * 0.5
    sound = sound / np.max(np.abs(sound)) * 0.65

    audio = np.array(sound * 32767, dtype=np.int16)
    stereo_audio = np.column_stack((audio, audio))

    return pygame.sndarray.make_sound(stereo_audio)


def generate_collision_boink(intensity=1.0):
    """Generate 'bonk' sound for wall/pillar collisions - like hitting a hollow surface.
    intensity: 0.0-1.0, scales pitch and volume based on collision strength
    """
    duration = 0.15
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)
    
    # Bass thump (the main impact)
    base_freq = 80 + (intensity * 40)  # 80-120 Hz - deep thud
    thump = np.sin(2 * np.pi * base_freq * t)
    
    # Add a higher "bonk" overtone that decays faster (like hitting wood)
    bonk_freq = 300 + (intensity * 200)  # 300-500 Hz
    bonk = np.sin(2 * np.pi * bonk_freq * t) * 0.4
    
    # Sharp attack, quick decay
    thump_envelope = np.exp(-t * 15)
    bonk_envelope = np.exp(-t * 35)  # Higher freq decays faster
    
    # Combine
    sound = thump * thump_envelope + bonk * bonk_envelope
    
    # Add tiny bit of noise for realism (like hitting a surface)
    noise = np.random.uniform(-0.02, 0.02, samples) * np.exp(-t * 25)
    sound += noise
    
    # Normalize
    sound = sound / np.max(np.abs(sound)) * (0.2 + intensity * 0.25)
    
    audio = np.array(sound * 32767, dtype=np.int16)
    stereo_audio = np.column_stack((audio, audio))
    
    return pygame.sndarray.make_sound(stereo_audio)


def generate_soft_bump():
    """Generate very soft bump sound for gentle collisions (sliding along walls)."""
    duration = 0.08
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)
    
    # Low frequency thud
    bump = np.sin(2 * np.pi * 120 * t) * np.exp(-t * 40)
    
    # Soft noise
    noise = np.random.normal(0, 0.05, samples) * np.exp(-t * 35)
    
    sound = bump * 0.4 + noise
    sound = sound / np.max(np.abs(sound)) * 0.1
    
    audio = np.array(sound * 32767, dtype=np.int16)
    stereo_audio = np.column_stack((audio, audio))
    
    return pygame.sndarray.make_sound(stereo_audio)
