from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter
from typing import Any

@dataclass(slots=True)
class CounterMetric:
    """
    Monotonic counter metric.
    """

    name: str
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def inc(
        self,
        amount: float = 1.0,
    ) -> None:
        """
        Increment counter.
        """

        with self._lock:
            self.value += amount

@dataclass(slots=True)
class GaugeMetric:
    """
    Mutable gauge metric.
    """

    name: str
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def set(
        self,
        value: float,
    ) -> None:
        """
        Sets gauge value.
        """

        with self._lock:
            self.value = value

    def inc(
        self,
        amount: float = 1.0,
    ) -> None:
        """
        Increment gauge.
        """

        with self._lock:
            self.value += amount

    def dec(
        self,
        amount: float = 1.0,
    ) -> None:
        """
        Decrement gauge.
        """

        with self._lock:
            self.value -= amount

@dataclass(slots=True)
class HistogramMetric:
    """
    Lightweight latency histogram.
    """

    name: str
    observations: list[float] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def observe(
        self,
        value: float,
    ) -> None:
        """
        Records observation.
        """

        with self._lock:
            self.observations.append(value)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.observations)

    @property
    def avg(self) -> float:
        with self._lock:
            if not self.observations:
                return 0.0

            return sum(
                self.observations
            ) / len(
                self.observations
            )

    @property
    def max(self) -> float:
        with self._lock:
            if not self.observations:
                return 0.0

            return max(self.observations)


def _metric_key(
    name: str,
    labels: dict[str, str] | None,
) -> str:
    if not labels:
        return name

    serialized = ",".join(
        f"{k}={labels[k]}"
        for k in sorted(labels)
    )

    return f"{name}|{serialized}"


def _prom_labels(
    labels: dict[str, str],
) -> str:
    if not labels:
        return ""

    parts = []
    for key in sorted(labels):
        value = labels[key].replace(
            "\\",
            "\\\\",
        ).replace('"', '\\"')
        parts.append(
            f'{key}="{value}"'
        )

    return "{" + ",".join(parts) + "}"

class MetricsRegistry:
    """
    Prometheus-compatible metrics registry.

    Responsibilities:
    - counters
    - gauges
    - latency histograms
    - runtime instrumentation

    Scope:
    - in-memory only
    - scrape-friendly export
    """

    def __init__(self):
        self._lock = Lock()

        self._counters: dict[
            str,
            CounterMetric,
        ] = {}

        self._gauges: dict[
            str,
            GaugeMetric,
        ] = {}

        self._histograms: dict[
            str,
            HistogramMetric,
        ] = {}

    def counter(
        self,
        name: str,
        *,
        labels: dict[str, str] | None = None,
    ) -> CounterMetric:
        """
        Returns counter metric.
        """

        with self._lock:
            key = _metric_key(
                name,
                labels,
            )

            if key not in self._counters:
                self._counters[key] = (
                    CounterMetric(
                        name=name,
                        labels=(
                            labels
                            or {}
                        ),
                    )
                )

            return self._counters[key]

    def gauge(
        self,
        name: str,
        *,
        labels: dict[str, str] | None = None,
    ) -> GaugeMetric:
        """
        Returns gauge metric.
        """

        with self._lock:
            key = _metric_key(
                name,
                labels,
            )

            if key not in self._gauges:
                self._gauges[key] = (
                    GaugeMetric(
                        name=name,
                        labels=(
                            labels
                            or {}
                        ),
                    )
                )

            return self._gauges[key]

    def histogram(
        self,
        name: str,
        *,
        labels: dict[str, str] | None = None,
    ) -> HistogramMetric:
        """
        Returns histogram metric.
        """

        with self._lock:
            key = _metric_key(
                name,
                labels,
            )

            if (
                key
                not in self._histograms
            ):

                self._histograms[
                    key
                ] = HistogramMetric(
                    name=name,
                    labels=(
                        labels
                        or {}
                    ),
                )

            return self._histograms[key]

    def timer(
        self,
        histogram_name: str,
    ):
        """
        Context timer helper.
        """

        registry = self

        class TimerContext:

            def __enter__(
                self,
            ):
                self.started = perf_counter()

                return self

            def __exit__(
                self,
                exc_type,
                exc,
                tb,
            ):
                elapsed = (
                    (
                        perf_counter()
                        - self.started
                    )
                    * 1000
                )

                registry.histogram(
                    histogram_name
                ).observe(
                    elapsed
                )

        return TimerContext()

    def export(
        self,
    ) -> dict[str, Any]:
        """
        Structured metrics snapshot.
        """

        with self._lock:
            counters = list(
                self._counters.items()
            )
            gauges = list(
                self._gauges.items()
            )
            histograms = list(
                self._histograms.items()
            )

        return {
            "counters": {
                key: {
                    "name": metric.name,
                    "value": metric.value,
                    "labels": metric.labels,
                }
                for key, metric in counters
            },
            "gauges": {
                key: {
                    "name": metric.name,
                    "value": metric.value,
                    "labels": metric.labels,
                }
                for key, metric in gauges
            },
            "histograms": {
                key: {
                    "name": metric.name,
                    "count": metric.count,
                    "avg": metric.avg,
                    "max": metric.max,
                    "labels": metric.labels,
                }
                for key, metric in histograms
            },
        }

    def prometheus(
        self,
    ) -> str:
        """
        Prometheus exposition format.
        """

        lines: list[str] = []

        with self._lock:
            counters = list(
                self._counters.values()
            )
            gauges = list(
                self._gauges.values()
            )
            histograms = list(
                self._histograms.values()
            )

        for metric in counters:
            labels = _prom_labels(
                metric.labels
            )

            lines.append(
                (
                    f"# TYPE {metric.name} "
                    f"counter"
                )
            )

            lines.append(
                f"{metric.name}{labels} {metric.value}"
            )

        for metric in gauges:
            labels = _prom_labels(
                metric.labels
            )

            lines.append(
                (
                    f"# TYPE {metric.name} "
                    f"gauge"
                )
            )

            lines.append(
                f"{metric.name}{labels} {metric.value}"
            )

        for metric in histograms:
            labels = _prom_labels(
                metric.labels
            )

            lines.append(
                (
                    f"# TYPE {metric.name} "
                    f"histogram"
                )
            )

            lines.append(
                (
                    f"{metric.name}_count{labels} "
                    f"{metric.count}"
                )
            )

            lines.append(
                (
                    f"{metric.name}_avg{labels} "
                    f"{metric.avg}"
                )
            )

            lines.append(
                (
                    f"{metric.name}_max{labels} "
                    f"{metric.max}"
                )
            )

        return "\n".join(lines)

