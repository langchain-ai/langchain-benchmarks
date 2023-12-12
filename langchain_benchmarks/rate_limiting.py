"""Implementation of a rate limiter based on a token bucket."""
import threading
import time
from typing import Any, Optional

from langchain.schema.runnable import Runnable, RunnableLambda
from langchain.schema.runnable.utils import Input, Output


class RateLimiter:
    def __init__(
        self,
        *,
        requests_per_second: float = 1,
        tokens_per_request: int = 1,
        check_every_n_seconds: float = 0.1,
    ) -> None:
        """A rate limiter based on a token bucket.

        This is a rate limiter which works in a threaded environment.

        It works by filling up a bucket with tokens at a given rate. Each
        request consumes a given number of tokens. If there are not enough
        tokens in the bucket, the request is blocked until there are enough
        tokens.

        Args:
            requests_per_second: The number of tokens to add per second to the bucket.
                Must be at least 1.
            tokens_per_request: number of tokens that a request "costs".
                This is 1 by default.
            check_every_n_seconds: check whether the tokens are available
                every this many seconds. Can be a float to represent
                fractions of a second.
        """
        if requests_per_second < 1:
            raise ValueError("Rate must be at least 1 request per second")
        self.requests_per_second = requests_per_second
        # Number of tokens in the bucket.
        self.available_tokens = 0.0
        # A lock to ensure that tokens can only be consumed by one thread
        # at a given time.
        self._consume_lock = threading.Lock()
        # tokens per request sets how many tokens
        self.tokens_per_request = tokens_per_request
        # The last time we tried to consume tokens.
        self.last: Optional[time.time] = None
        self.check_every_n_seconds = check_every_n_seconds

    def consume(self) -> bool:
        """Consume the given amount of tokens if possible.

        Returns:
            True means that the tokens were consumed, and the caller can proceed to
            make the request. A False means that the tokens were not consumed, and
            the caller should try again later.
        """
        with self._consume_lock:
            now = time.time()

            # initialize on first call to avoid a burst
            if self.last is None:
                self.last = now

            elapsed = now - self.last

            if elapsed * self.requests_per_second > 1:
                self.available_tokens += elapsed * self.requests_per_second
                self.last = now

            self.available_tokens = min(self.available_tokens, self.requests_per_second)

            if self.available_tokens >= self.tokens_per_request:
                self.available_tokens -= self.tokens_per_request
                return True

            return False

    def wait(self) -> None:
        """Blocking call to wait until the given number of tokens are available."""
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
