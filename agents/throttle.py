"""Throttle using a token bucket."""
import threading
import time


class Throttle:
    def __init__(self, rate: int) -> None:
        """Initialize the throttle."""
        self.rate = rate
        self.tokens = 0
        self._consume_lock = threading.Lock()
        self.last = None

    def consume(self, amount: int = 0) -> int:
        """Consume the given amount of tokens."""
        with self._consume_lock:
            now = time.time()

            # initialize on first call to avoid a burst
            if self.last is None:
                self.last = now

            elapsed = now - self.last

            if elapsed * self.rate > 1:
                self.tokens += elapsed * self.rate
                self.last = now

            self.tokens = min(self.tokens, self.rate)

            if self.tokens >= amount:
                self.tokens -= amount
                return amount

            return 0
