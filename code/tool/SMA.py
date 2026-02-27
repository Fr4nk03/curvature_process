# Simple Moving Average for suppressing jitter
from collections import deque

class SMA:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.current_sum = 0.0
    
    def update(self, new_value):
        if len(self.buffer) == self.window_size:
            self.current_sum -= self.buffer[0]
        
        self.buffer.append(new_value)
        self.current_sum += new_value

        return self.current_sum / len(self.buffer)