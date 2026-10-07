import pytest
from src.cost import (break_even_requests_per_day, cost_per_million_tokens, instances_needed, monthly_api_cost,
                      monthly_self_host_cost, request_cost)


def test_cost_per_million_tokens():
    assert cost_per_million_tokens(1000, 3.6) == pytest.approx(1.0)
    assert cost_per_million_tokens(1000, 3.6, 0.5) == pytest.approx(2.0)


def test_request_cost():
    assert request_cost(1.0, 3.6) == pytest.approx(0.001)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        cost_per_million_tokens(0, 1)
    with pytest.raises(ValueError):
        instances_needed(1, 1, headroom=1.5)


def test_instances_needed():
    assert instances_needed(100, 50, 0.5) == 4
    assert instances_needed(0.01, 50) == 1


def test_api_and_break_even():
    # 1000 in + 100 out tokens at $1 / $2 per 1M tokens = $0.0012 per request
    assert monthly_api_cost(1000, 1000, 100, 1.0, 2.0, days=30) == pytest.approx(36.0)
    assert monthly_self_host_cost(1, 0.1) == pytest.approx(73.0)
    be = break_even_requests_per_day(73.0, 1000, 100, 1.0, 2.0)
    assert be == pytest.approx(73.0 / (0.0012 * 30))
