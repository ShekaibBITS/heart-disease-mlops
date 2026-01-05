from prometheus_client import Counter, Histogram, Gauge

CI_PIPELINE_RUNS_TOTAL = Counter(
    "ci_pipeline_runs_total",
    "Total CI pipeline runs triggered by admin API",
    ["mode"],
)

CI_PIPELINE_FAILURES_TOTAL = Counter(
    "ci_pipeline_failures_total",
    "Total CI pipeline failures triggered by admin API",
)

CI_PIPELINE_DURATION_SECONDS = Histogram(
    "ci_pipeline_duration_seconds",
    "Total CI pipeline duration in seconds",
)

CI_PIPELINE_IN_PROGRESS = Gauge(
    "ci_pipeline_in_progress",
    "CI pipeline running (1/0)",
)
