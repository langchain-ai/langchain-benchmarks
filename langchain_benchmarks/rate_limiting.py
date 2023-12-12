"""Throttle using a token bucket."""
import threading
import time
from typing import Optional, Any

from langchain.schema.runnable import RunnableLambda, Runnable
from langchain.schema.runnable.utils import Input, Output


class RateLimiter:
    def __init__(
        self, *, rate: float = 1, amount: int = 1, check_every_n_seconds: float = 0.1
    ) -> None:
        """Initialize the token bucket.

        Args:
            rate: The number of tokens to add per second to the bucket.
                  Must be at least one.
            amount: default amount of tokens to consume
            check_every_n_seconds: check whether the tokens are available
                every this many seconds. Can be a float to represent
                fractions of a second.
        """
        if rate < 1:
            raise ValueError("Rate must be at least 1 request per second")
        self.rate = rate
        # Number of tokens in the bucket.
        self.tokens = 0.0
        # A lock to ensure that tokens can only be consumed by one thread
        # at a given time.
        self._consume_lock = threading.Lock()
        self.amount = amount
        # The last time we tried to consume tokens.
        self.last: Optional[time.time] = None
        self.check_every_n_seconds = check_every_n_seconds

    def consume(self) -> bool:
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

            if self.tokens >= self.amount:
                self.tokens -= self.amount
                return True

            return False

    def wait(self) -> None:
        """Blocking wait until the given number of tokens are available."""
        while not self.consume():
            time.sleep(self.check_every_n_seconds)


def with_rate_limit(
    runnable: Runnable[Input, Output],
    rate_limiter: RateLimiter,
) -> Runnable[Input, Output]:
    """Add a rate limiter to the runnable.

    Args:
        runnable: The runnable to throttle.
        rate_limiter: The throttle to use.

    Returns:
        A runnable lambda that acts as a throttled passthrough.
    """

    def _rate_limited_passthrough(input: dict, **kwargs: Any) -> dict:
        """Throttle the input."""
        rate_limiter.wait()
        return input

    return (
        RunnableLambda(_rate_limited_passthrough).with_config({"name": "Rate Limit"})
        | runnable
    )
