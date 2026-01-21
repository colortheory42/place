"""
Event system.
Simple pub/sub for decoupled communication between systems.
"""

from enum import Enum, auto
from collections import defaultdict


class EventType(Enum):
    """All event types in the engine."""
    # Wall events
    WALL_HIT = auto()          # Wall took damage but didn't break
    WALL_CRACKED = auto()      # Wall entered cracked state
    WALL_FRACTURED = auto()    # Wall entered fractured state
    WALL_BREAKING = auto()     # Wall starting to fall
    WALL_DESTROYED = auto()    # Wall fully destroyed
    
    # Pillar events
    PILLAR_HIT = auto()
    PILLAR_DESTROYED = auto()
    
    # Debris events
    DEBRIS_IMPACT = auto()     # Large debris hit ground
    DEBRIS_SETTLED = auto()    # Debris pile formed
    
    # Player events
    PLAYER_STEP = auto()       # Footstep
    PLAYER_LAND = auto()       # Landed from jump
    
    # Atmosphere events
    FLICKER = auto()           # Light flickered


class Event:
    """Event data container."""
    
    def __init__(self, event_type, **data):
        self.type = event_type
        self.data = data
    
    def __getattr__(self, name):
        if name in ('type', 'data'):
            return super().__getattribute__(name)
        return self.data.get(name)
    
    def __repr__(self):
        return f"Event({self.type.name}, {self.data})"


class EventBus:
    """
    Central event dispatcher.
    
    Usage:
        bus = EventBus()
        
        # Subscribe
        bus.subscribe(EventType.WALL_DESTROYED, my_handler)
        
        # Emit
        bus.emit(EventType.WALL_DESTROYED, wall_key=key, position=(x, y, z))
        
        # Handler receives Event object
        def my_handler(event):
            print(event.wall_key, event.position)
    """
    
    def __init__(self):
        self._subscribers = defaultdict(list)
        self._queue = []
        self._processing = False
    
    def subscribe(self, event_type, handler):
        """Subscribe a handler to an event type."""
        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type, handler):
        """Unsubscribe a handler from an event type."""
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
    
    def emit(self, event_type, **data):
        """
        Emit an event immediately.
        Handlers are called synchronously.
        """
        event = Event(event_type, **data)
        for handler in self._subscribers[event_type]:
            try:
                handler(event)
            except Exception as e:
                print(f"Event handler error for {event_type}: {e}")
    
    def queue(self, event_type, **data):
        """
        Queue an event for later processing.
        Use this during update loops to avoid mutation during iteration.
        """
        self._queue.append((event_type, data))
    
    def process_queue(self):
        """Process all queued events."""
        if self._processing:
            return
        
        self._processing = True
        while self._queue:
            event_type, data = self._queue.pop(0)
            self.emit(event_type, **data)
        self._processing = False
    
    def clear(self):
        """Clear all subscribers and queued events."""
        self._subscribers.clear()
        self._queue.clear()


# Global event bus instance
# Systems import this and subscribe/emit directly
event_bus = EventBus()
