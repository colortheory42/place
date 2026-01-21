"""
Flexible Audio Input Backend
=============================
Tries multiple audio libraries in order of preference:
1. sounddevice (easiest to install, cross-platform)
2. pyaudio (traditional, but harder to install)
3. wave + built-in (file-based fallback for testing)

Uses whichever is available.
"""

import numpy as np
import queue
import threading
from config import SAMPLE_RATE


class AudioBackend:
    """Base class for audio backends."""
    
    def __init__(self, sample_rate=SAMPLE_RATE, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue(maxsize=5)  # Balance queue
        self.recording = False
        
    def start(self):
        """Start capturing audio."""
        raise NotImplementedError
        
    def stop(self):
        """Stop capturing audio."""
        raise NotImplementedError
        
    def get_audio_chunk(self):
        """Get latest audio chunk (non-blocking)."""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None


# ============================================================
# SOUNDDEVICE BACKEND (Easiest)
# ============================================================

class SoundDeviceBackend(AudioBackend):
    """Audio backend using sounddevice (pip install sounddevice)."""
    
    def __init__(self, sample_rate=SAMPLE_RATE, chunk_size=1024):
        super().__init__(sample_rate, chunk_size)
        try:
            import sounddevice as sd
            self.sd = sd
            self.has_sounddevice = True
            print(f"   SoundDevice backend will use {sample_rate} Hz")
        except ImportError:
            self.has_sounddevice = False
            
    def start(self):
        """Start capturing from microphone."""
        if not self.has_sounddevice:
            return False
            
        try:
            def audio_callback(indata, frames, time_info, status):
                """Audio callback for sounddevice."""
                if status:
                    print(f"Audio status: {status}")
                
                # Convert to int16
                audio_data = (indata[:, 0] * 32767).astype(np.int16)
                
                try:
                    self.audio_queue.put_nowait(audio_data)
                except queue.Full:
                    pass  # Drop frame if queue full
            
            self.stream = self.sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.chunk_size,
                callback=audio_callback
            )
            
            self.recording = True
            self.stream.start()
            print("🎤 Microphone capture started (sounddevice)")
            return True
            
        except Exception as e:
            print(f"❌ Sounddevice error: {e}")
            return False
    
    def stop(self):
        """Stop capturing."""
        if not self.has_sounddevice or not self.recording:
            return
            
        self.recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        print("🎤 Microphone capture stopped")


# ============================================================
# PYAUDIO BACKEND (Traditional)
# ============================================================

class PyAudioBackend(AudioBackend):
    """Audio backend using PyAudio (pip install pyaudio)."""
    
    def __init__(self, sample_rate=SAMPLE_RATE, chunk_size=1024):
        super().__init__(sample_rate, chunk_size)
        try:
            import pyaudio
            self.pyaudio = pyaudio
            self.has_pyaudio = True
        except ImportError:
            self.has_pyaudio = False
            
    def start(self):
        """Start capturing from microphone."""
        if not self.has_pyaudio:
            return False
            
        try:
            self.p = self.pyaudio.PyAudio()
            self.stream = self.p.open(
                format=self.pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            self.recording = True
            self.stream.start_stream()
            print("🎤 Microphone capture started (PyAudio)")
            return True
        except Exception as e:
            print(f"❌ PyAudio error: {e}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback for PyAudio."""
        if self.recording:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            try:
                self.audio_queue.put_nowait(audio_data)
            except queue.Full:
                pass
        return (in_data, self.pyaudio.paContinue)
    
    def stop(self):
        """Stop capturing."""
        if not self.has_pyaudio or not self.recording:
            return
            
        self.recording = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()
        print("🎤 Microphone capture stopped")


# ============================================================
# WAVE FILE BACKEND (Testing/Fallback)
# ============================================================

class WaveFileBackend(AudioBackend):
    """
    Fallback backend that loops a test audio file.
    Useful for testing when no mic available.
    """
    
    def __init__(self, sample_rate=SAMPLE_RATE, chunk_size=1024, filepath=None):
        super().__init__(sample_rate, chunk_size)
        self.filepath = filepath
        self.thread = None
        
    def start(self):
        """Start reading from file in a loop."""
        if self.filepath is None:
            # Generate test tone if no file provided
            print("🎵 Using test tone generator (no mic available)")
            self.use_test_tone = True
        else:
            print(f"🎵 Using audio file: {self.filepath}")
            self.use_test_tone = False
        
        self.recording = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        return True
    
    def _read_loop(self):
        """Read audio in a loop."""
        chunk_idx = 0
        
        if self.use_test_tone:
            # Generate a simple repeating tone
            duration = 2.0
            t = np.linspace(0, duration, int(self.sample_rate * duration), False)
            audio = np.sin(2 * np.pi * 440 * t) * 16384  # A4 note
            audio = audio.astype(np.int16)
        else:
            # Load from file
            try:
                import wave
                with wave.open(self.filepath, 'rb') as wf:
                    audio_data = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(audio_data, dtype=np.int16)
            except Exception as e:
                print(f"❌ Failed to load audio file: {e}")
                return
        
        while self.recording:
            # Get chunk
            start = chunk_idx * self.chunk_size
            end = start + self.chunk_size
            
            if end > len(audio):
                chunk_idx = 0
                start = 0
                end = self.chunk_size
            
            chunk = audio[start:end]
            
            try:
                self.audio_queue.put_nowait(chunk)
            except queue.Full:
                pass
            
            chunk_idx += 1
            
            # Sleep to simulate real-time
            import time
            time.sleep(self.chunk_size / self.sample_rate)
    
    def stop(self):
        """Stop reading."""
        self.recording = False
        if self.thread:
            self.thread.join(timeout=1.0)
        print("🎵 Audio file playback stopped")


# ============================================================
# AUTO-DETECT BACKEND
# ============================================================

def create_audio_backend(sample_rate=SAMPLE_RATE, chunk_size=1024):
    """
    Auto-detect and create the best available audio backend.
    
    Returns:
        AudioBackend instance, or None if nothing available
    """
    # Try sounddevice first (easiest to install)
    backend = SoundDeviceBackend(sample_rate, chunk_size)
    if backend.has_sounddevice:
        print("✅ Using sounddevice backend")
        return backend
    
    # Try PyAudio
    backend = PyAudioBackend(sample_rate, chunk_size)
    if backend.has_pyaudio:
        print("✅ Using PyAudio backend")
        return backend
    
    # Fallback to test tone
    print("⚠️  No microphone library found!")
    print("   Install one of these:")
    print("   - sounddevice (recommended): pip install sounddevice")
    print("   - pyaudio: pip install pyaudio")
    print("   Using test tone generator as fallback...")
    
    return WaveFileBackend(sample_rate, chunk_size)


# ============================================================
# SIMPLIFIED MICROPHONE CAPTURE
# ============================================================

class MicrophoneCapture:
    """
    Simplified microphone capture that auto-detects backend.
    Drop-in replacement for the old MicrophoneCapture class.
    """
    
    def __init__(self, sample_rate=SAMPLE_RATE, chunk_size=1024):
        self.backend = create_audio_backend(sample_rate, chunk_size)
        self.has_audio = self.backend is not None
    
    def start(self):
        """Start capturing."""
        if not self.has_audio:
            print("❌ No audio backend available")
            return False
        return self.backend.start()
    
    def stop(self):
        """Stop capturing."""
        if self.has_audio:
            self.backend.stop()
    
    def get_audio_chunk(self):
        """Get latest audio chunk."""
        if not self.has_audio:
            return None
        return self.backend.get_audio_chunk()


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("Testing audio backends...\n")
    
    # Test which backends are available
    print("Checking available backends:")
    
    try:
        import sounddevice
        print("  ✅ sounddevice available")
    except ImportError:
        print("  ❌ sounddevice not available")
        print("     Install: pip install sounddevice")
    
    try:
        import pyaudio
        print("  ✅ pyaudio available")
    except ImportError:
        print("  ❌ pyaudio not available")
        print("     Install: pip install pyaudio")
    
    print("\nCreating microphone capture...")
    mic = MicrophoneCapture()
    
    if mic.start():
        print("\n🎤 Recording for 3 seconds...")
        print("Speak into your microphone!\n")
        
        import time
        start = time.time()
        chunks = 0
        max_amplitude = 0
        
        while time.time() - start < 3.0:
            chunk = mic.get_audio_chunk()
            if chunk is not None:
                chunks += 1
                amplitude = np.max(np.abs(chunk))
                max_amplitude = max(max_amplitude, amplitude)
                
                # Volume meter
                bars = int((amplitude / 32768.0) * 40)
                meter = "█" * bars + "░" * (40 - bars)
                print(f"\r[{meter}] {amplitude:5d}", end="", flush=True)
            
            time.sleep(0.01)
        
        print(f"\n\n📊 Captured {chunks} chunks")
        print(f"   Max amplitude: {max_amplitude}")
        
        mic.stop()
    else:
        print("❌ Failed to start capture")
