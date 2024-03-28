# Request and Response Caching

In HELM, most requests to models and tokenizers are cached. The cache serves the following purposes:

- **Failure recovery**: Many model inference APIs can occasionally return errors, which can cause evaluation runs to fail midway. The cache allows the user to redo the evaluation run up to the point of failure from the cache, allows users to avoid spending time and money on sending the same requests again.
- **Reproducibility**: Many model inference APIs have non-deterministic behavior, _even_ with temperature zero or fixed random seeds. By caching the exact requests and responses from an evaluation run, we can use this cache later to recreate the exact output from that evaluation run.


## Cache Stats

At the end of each evaluation run, `helm-run` will print a `CacheStats` object that indicates the number of cache hits. If a `helm-run` invocation resulted in multiple evaluation runs (i.e. multiple `RunSpec`s), the `CacheStats` printed will reflect the cumulative totals for all evaluation runs so far, rather than only the last evaluation run.

For instance, the `CacheStats` printed below indicates that there were a total of 20 model and tokenizer requests made. 5 of them were cache hits, and 15 were cache misses.

```
CacheStats.print_status {
  prod_env/cache/simple.sqlite: 20 queries, 15 computes
}
```

## Configuring cache storage

HELM currently supports two different backing databases for the cache: SQLite and MongoDB.

HELM stores its cache in SQLite by default. The `*.sqlite` files will be placed in the `cache/` subfolder underneath your local configuration folder (usually `./prod_env/` under your current working directory). You can specify a different location by setting the local configuration folder using the `--local-path` flag of `helm-run`.

You can configure HELM to use MongoDB instead of SQLite. To do this, specify the `--mongo-uri` flag to `helm-run` e.g. `--mongo-uri 'mongodb://username:password@host/database'`. Refer to the [MongoDB connection string documentation](https://www.mongodb.com/docs/manual/reference/connection-string/) for more information.

Note: HELM currently does not support loading the MongoDB password from `credentials.conf` or from an environment variable. This feature request is tracked in GitHub Issue.

## Disabling the cache

You can disable all cache usage completely by passing `--disable-cache` to `helm-run`. This can be useful for development, such as if you are activately making changes to your model inference API and want to see its new responses immediately.
