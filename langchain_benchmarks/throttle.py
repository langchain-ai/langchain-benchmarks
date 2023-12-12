"""Throttle using a token bucket."""
import threading
import time
from typing import Optional, Any

from langchain.schema.runnable import RunnableLambda


class RateLimiter:
    def __init__(self, *, rate: float) -> None:
        """Initialize the token bucket.

        Args:
            rate: The number of tokens to add per second to the bucket.
                  Must be at least one.
        """
        if rate < 1:
            raise ValueError("Rate must be at least 1 request per second")
        self.rate = rate
        # Number of tokens in the bucket.
        self.tokens = 0.0
        # A lock to ensure that tokens can only be consumed by one thread
        # at a given time.
        self._consume_lock = threading.Lock()
        # The last time we tried to consume tokens.
        self.last: Optional[time.time] = None

    def consume(self, *, amount: int = 0) -> bool:
        """Consume the given amount of tokens if possible.

        Args:
            amount: The number of tokens to consume.

        Returns:
            True, if there were enough tokens to consume, False otherwise.
        """
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
                return True

            return False

    def wait(self, *, amount: int = 1, try_every_n_seconds: float = 0.1) -> None:
        """Blocking wait until the given number of tokens are available.

        Args:
            amount: The number of tokens to wait for.
            try_every_n_seconds: The number of seconds to sleep between checks.
        """
        while not self.consume(amount=amount):
            time.sleep(try_every_n_seconds)


def with_rate_limit(
    throttle: RateLimiter,
    *,
    amount: int = 1,
    try_every_n_seconds: float = 0.1,
) -> RunnableLambda:
    """Create a passthrough that throttles before passing through.

    Args:
        throttle: The throttle to use.
        amount: The number of tokens to wait for.
        try_every_n_seconds: The number of seconds to sleep between checks.

    Returns:
        A runnable lambda that acts as a throttled passthrough.
    """

    def _throttle_step(input: dict, **kwargs: Any) -> dict:
        """Throttle the input."""
        throttle.wait(amount=amount, try_every_n_seconds=try_every_n_seconds)
        return input

    return RunnableLambda(_throttle_step).with_config({"name": "throttle"})
