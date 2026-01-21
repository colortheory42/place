"""
Simple Audio Loopback Mode
===========================
Clean mic → speaker passthrough with optional simple reverb.
No raycasting, just smooth audio playback.
"""

import numpy as np
import pygame
import threading
import queue
from audio_backends import MicrophoneCapture


class SimpleAudioLoopback:
    """
    Simple audio loopback - clean passthrough only.
    No reverb, no processing, just mic → speaker.
    """
    
    def __init__(self):
        # Sweet spot: not too big, not too small
        self.mic = MicrophoneCapture(sample_rate=44100, chunk_size=1024)
        self.enabled = False
        
        # Moderate queue for balance
        self.playback_queue = queue.Queue(maxsize=8)
        self.playback_thread = None
        self.update_thread = None
        self.should_stop = False
        
        # Track current playing channel for better synchronization
        self.current_channel = None
    
    def _playback_worker(self):
        """Worker thread for smooth playback with proper timing."""
        import time
        
        while not self.should_stop:
            try:
                left, right = self.playback_queue.get(timeout=0.1)
                
                # Wait for previous chunk to finish playing
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
        """Dedicated update thread - runs at 1000 Hz independent of game FPS."""
        import time
        
        while not self.should_stop:
            if self.enabled:
                # Get audio chunk
                audio_chunk = self.mic.get_audio_chunk()
                if audio_chunk is not None:
                    # Process all audio (no noise gate)
                    left, right = self._process_chunk(audio_chunk)
                    
                    # Queue management
                    if self.playback_queue.full():
                        try:
                            self.playback_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    # Queue fresh audio
                    try:
                        self.playback_queue.put_nowait((left, right))
                    except queue.Full:
                        pass
            
            time.sleep(0.001)  # 1ms = 1000 Hz update rate
    
    def _process_chunk(self, audio_chunk):
        """Process audio chunk - pure passthrough with significant gain."""
        # Convert to float
        audio_float = audio_chunk.astype(np.float32) / 32768.0
        
        # Very gentle normalization to preserve dynamics
        max_val = np.max(np.abs(audio_float))
        if max_val > 0.9:
            audio_float = audio_float * (0.9 / max_val)
        
        # Significant boost for audibility (3x gain)
        audio_float = audio_float * 3.0
        
        # Pure passthrough with gain
        output_left = np.clip(audio_float * 32767, -32768, 32767).astype(np.int16)
        output_right = output_left.copy()
        
        return output_left, output_right
    
    def start(self):
        """Start loopback."""
        if self.mic.start():
            self.enabled = True
            self.should_stop = False
            
            # Start playback thread
            self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
            self.playback_thread.start()
            
            # Start update thread (independent of game FPS)
            self.update_thread = threading.Thread(target=self._update_worker, daemon=True)
            self.update_thread.start()
            
            print(f"🔊 Audio loopback started (dedicated update thread @ 1000 Hz)")
            return True
        return False
    
    def stop(self):
        """Stop loopback."""
        self.mic.stop()
        self.enabled = False
        self.should_stop = True
        
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)
        
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        
        self.current_channel = None
        print("🔇 Audio loopback stopped")
    
    def update(self):
        """Update - handled by dedicated thread, game can call this but it does nothing."""
        pass  # Update thread runs independently at 1000 Hz


# Test standalone
if __name__ == "__main__":
    import time
    
    print("="*60)
    print("SIMPLE AUDIO LOOPBACK TEST")
    print("Pure passthrough - no reverb, no effects")
    print("="*60)
    
    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
    
    loopback = SimpleAudioLoopback()
    
    if loopback.start():
        print("\n🎤 Speak into your microphone!")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                loopback.update()
                time.sleep(0.001)  # Very fast update rate
        except KeyboardInterrupt:
            print("\n\nStopping...")
        
        loopback.stop()
    else:
        print("❌ Failed to start")
    
    print("\nDone!")
