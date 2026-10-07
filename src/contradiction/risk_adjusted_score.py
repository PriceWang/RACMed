"""Reliability- and deployment-adjusted aggregate score."""

def risk_adjusted_score(average_performance,worst_task_drop,ece=0.,false_negative_rate=0.,normalized_latency=0.,alpha=1.,beta=1.,gamma=1.,delta=.1):
    return float(average_performance-alpha*worst_task_drop-beta*ece-gamma*false_negative_rate-delta*normalized_latency)
