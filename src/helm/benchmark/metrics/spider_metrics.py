from helm.benchmark.metrics.bird_sql_metrics import BirdSQLMetric


class SpiderMetric(BirdSQLMetric):
    """Score metrics for Spider. Based on Bird-SQL."""

    ANNOTATOR_NAME = "spider"
