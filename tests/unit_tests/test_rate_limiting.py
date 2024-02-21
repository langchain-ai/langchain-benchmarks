import pytest
from freezegun import freeze_time

from langchain_benchmarks.rate_limiting import RateLimiter


@pytest.mark.parametrize(
    "delta_time, requests_per_second, max_bucket_size, expected_result",
    [
        (
            1,
            1,
            1,
            True,
        ),
        (
            0.5,
            1,
            1,
            False,
        ),
        (
            0.5,
            2,
            1,
            True,
        ),
    ],
)
def test_consume(
    delta_time: float,
    requests_per_second: float,
    max_bucket_size: float,
    expected_result: bool,
) -> None:
    """Test the consumption of tokens over time.

    Args:
        delta_time: The time in seconds to add to the initial time.
        requests_per_second: The rate at which tokens are added per second.
        max_bucket_size: The maximum size of the token bucket.
        expected_result: The expected result of the consume operation.
    """
    rate_limiter = RateLimiter(
        requests_per_second=requests_per_second, max_bucket_size=max_bucket_size
    )

    with freeze_time(auto_tick_seconds=delta_time):
        assert rate_limiter._consume() is False
        assert rate_limiter._consume() is expected_result


def test_consume_count_tokens() -> None:
    """Test to check that the bucket size is used correctly."""
    rate_limiter = RateLimiter(
        requests_per_second=60,
        max_bucket_size=10,
    )

    with freeze_time(auto_tick_seconds=100):
        assert rate_limiter._consume() is False
        assert rate_limiter._consume() is True
        assert (
            rate_limiter.available_tokens == 9
        )  # Max bucket size is 10, so 10 - 1 = 9
