"""Pure cost / capacity math (no ML dependencies, fully unit-tested)."""
import math


def _check(rate, utilization):
    if rate <= 0 or not 0 < utilization <= 1:
        raise ValueError("rate must be > 0 and utilization in (0, 1]")


def cost_per_million_tokens(tokens_per_sec: float, hourly_usd: float, utilization: float = 1.0) -> float:
    """USD to process 1M tokens on one instance running at the given utilization."""
    _check(tokens_per_sec, utilization)
    return hourly_usd / (tokens_per_sec * utilization * 3600) * 1_000_000


def request_cost(service_time_s: float, hourly_usd: float, utilization: float = 1.0) -> float:
    """USD for one request that occupies an instance for service_time_s seconds."""
    _check(service_time_s, utilization)
    return hourly_usd * service_time_s / 3600 / utilization


def instances_needed(peak_rps: float, instance_rps: float, headroom: float = 0.8) -> int:
    """Instances to serve peak traffic while keeping each one below `headroom` of its measured capacity."""
    _check(instance_rps, headroom)
    return max(1, math.ceil(peak_rps / (instance_rps * headroom)))


def monthly_self_host_cost(n_instances: int, hourly_usd: float, hours_per_month: float = 730) -> float:
    return n_instances * hourly_usd * hours_per_month


def monthly_api_cost(requests_per_day: float, input_tokens: float, output_tokens: float,
                     input_usd_per_m: float, output_usd_per_m: float, days: float = 30) -> float:
    per_request = (input_tokens * input_usd_per_m + output_tokens * output_usd_per_m) / 1_000_000
    return requests_per_day * days * per_request


def break_even_requests_per_day(self_host_monthly_usd: float, input_tokens: float, output_tokens: float,
                                input_usd_per_m: float, output_usd_per_m: float, days: float = 30) -> float:
    per_request = (input_tokens * input_usd_per_m + output_tokens * output_usd_per_m) / 1_000_000
    if per_request <= 0:
        raise ValueError("API price per request must be > 0")
    return self_host_monthly_usd / (per_request * days)
