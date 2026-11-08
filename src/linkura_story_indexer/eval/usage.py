"""Per-request model usage capture and cost estimation for benchmark runs."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from linkura_story_indexer.eval.answer_models import ModelCallUsage, UsageSummary


@dataclass(frozen=True)
class ModelPrice:
    """Standard-tier list prices in USD per 1M tokens.

    ``cached_input`` and ``cache_write`` of ``None`` bill those tokens at the input rate.
    Requests whose input exceeds ``long_context_threshold`` bill every token of that request
    at the long-context multipliers.
    """

    input: float
    output: float
    cached_input: float | None = None
    cache_write: float | None = None
    long_context_threshold: int | None = None
    long_input_multiplier: float = 1.0
    long_output_multiplier: float = 1.0


# Rates are list prices the maintainer supplied (2026-11-06 and 2026-11-07); costs computed
# from them are estimates, not provider invoices.
MODEL_PRICING: dict[str, ModelPrice] = {
    "gemini-3.8-flash": ModelPrice(0.75, 3.75),
    "gpt-6.1-terra": ModelPrice(1.00, 5.00, cached_input=0.05),
    "gpt-6.1-sol": ModelPrice(2.00, 10.00),
    "gemini-4-argon": ModelPrice(2.00, 10.00, cached_input=0.10),
    "gpt-6-luna": ModelPrice(
        0.10,
        0.50,
        cached_input=0.01,
        cache_write=0.125,
        long_context_threshold=272_000,
        long_input_multiplier=2.0,
        long_output_multiplier=1.5,
    ),
}

_recorder: ContextVar[list[ModelCallUsage] | None] = ContextVar("usage_recorder", default=None)


class UsageRecordingModel(WrapperModel):
    """Records provider-reported usage for each request into the active recorder."""

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        response = await self.wrapped.request(messages, model_settings, model_request_parameters)
        sink = _recorder.get()
        if sink is not None:
            usage = response.usage
            sink.append(
                ModelCallUsage(
                    model_name=response.model_name or self.wrapped.model_name,
                    provider=response.provider_name,
                    input_tokens=usage.input_tokens,
                    cache_read_tokens=usage.cache_read_tokens,
                    cache_write_tokens=usage.cache_write_tokens,
                    output_tokens=usage.output_tokens,
                )
            )
        return response


def maybe_record_usage(model: Any) -> Any:
    """Wraps ``model`` for usage capture when a recorder is active; otherwise returns it."""
    if _recorder.get() is None or not isinstance(model, Model):
        return model
    if isinstance(model, UsageRecordingModel):
        return model
    return UsageRecordingModel(model)


@contextmanager
def record_usage() -> Iterator[list[ModelCallUsage]]:
    calls: list[ModelCallUsage] = []
    token = _recorder.set(calls)
    try:
        yield calls
    finally:
        _recorder.reset(token)


def _price_for(model_name: str) -> ModelPrice | None:
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]
    # Providers sometimes report dated snapshots such as ``gpt-6.1-sol-2026-10-01``.
    for known, price in MODEL_PRICING.items():
        if model_name.startswith(f"{known}-"):
            return price
    return None


def request_cost(call: ModelCallUsage, price: ModelPrice) -> float:
    """Estimated USD cost of one request; cached and written tokens are part of input_tokens."""
    long_context = (
        price.long_context_threshold is not None
        and call.input_tokens > price.long_context_threshold
    )
    input_multiplier = price.long_input_multiplier if long_context else 1.0
    output_multiplier = price.long_output_multiplier if long_context else 1.0
    cached = call.cache_read_tokens if price.cached_input is not None else 0
    written = call.cache_write_tokens if price.cache_write is not None else 0
    uncached = max(call.input_tokens - cached - written, 0)
    cost = uncached * price.input
    cost += cached * (price.cached_input or 0.0)
    cost += written * (price.cache_write or 0.0)
    cost *= input_multiplier
    cost += call.output_tokens * price.output * output_multiplier
    return cost / 1_000_000


def summarize_usage(calls: Sequence[ModelCallUsage]) -> UsageSummary:
    summary = UsageSummary(requests=len(calls))
    cost = 0.0
    unpriced: set[str] = set()
    for call in calls:
        summary.input_tokens += call.input_tokens
        summary.cache_read_tokens += call.cache_read_tokens
        summary.output_tokens += call.output_tokens
        bucket = summary.by_model.setdefault(
            call.model_name,
            {
                "requests": 0,
                "input_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "output_tokens": 0,
            },
        )
        bucket["requests"] += 1
        bucket["input_tokens"] += call.input_tokens
        bucket["cache_read_tokens"] += call.cache_read_tokens
        bucket["cache_write_tokens"] += call.cache_write_tokens
        bucket["output_tokens"] += call.output_tokens
        price = _price_for(call.model_name)
        if price is None:
            unpriced.add(call.model_name)
            continue
        cost += request_cost(call, price)
    if unpriced:
        summary.cost_unavailable_reason = "no price configured for " + ", ".join(sorted(unpriced))
    elif calls:
        summary.estimated_cost_usd = round(cost, 6)
    return summary


def merge_usage(summaries: Sequence[UsageSummary]) -> UsageSummary:
    merged = UsageSummary()
    reasons: set[str] = set()
    cost = 0.0
    cost_known = True
    for summary in summaries:
        merged.requests += summary.requests
        merged.input_tokens += summary.input_tokens
        merged.cache_read_tokens += summary.cache_read_tokens
        merged.output_tokens += summary.output_tokens
        for model_name, counts in summary.by_model.items():
            bucket = merged.by_model.setdefault(model_name, dict.fromkeys(counts, 0))
            for key, value in counts.items():
                bucket[key] = bucket.get(key, 0) + value
        if summary.cost_unavailable_reason:
            reasons.add(summary.cost_unavailable_reason)
            cost_known = False
        elif summary.estimated_cost_usd is not None:
            cost += summary.estimated_cost_usd
    if reasons:
        merged.cost_unavailable_reason = "; ".join(sorted(reasons))
    if cost_known and merged.requests:
        merged.estimated_cost_usd = round(cost, 6)
    return merged
