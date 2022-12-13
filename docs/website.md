# Website Deployment

The [HELM website](https://crfm.stanford.edu/helm/) consists entirely of static content. To build the website, make a copy of the static web assets folder provided by this package, place the `benchmark_output` directory immediately under this server, then serve the folder with a web server. The command `helm-server --print-static-assets-path` will print the path to the static web assets directory. Take care to resolve symbolic links when copying files from this directory.

Note that opening the local files directly in the browser will not work, because local HTML files viewed over the file URI scheme are not allowed to make AJAX calls, which are required for the website to work. Therefore, a web server must be used to serve the website. For development purposes, the `helm-server` command or the built in Python `http.server` can be used as a server, but these tools are not suitable for production. Instead, use a production web server such as such as a Apache or NGINX.

Example:

```bash
# Replace $PATH_TO_BENCHMARK_OUTPUT to the path to benchmark_output
cp -rL $(helm-server --print-static-assets-path) helm-website
ln -s $PATH_TO_BENCHMARK_OUTPUT helm-website/benchmark_output
python -m http.server --directory helm-website
```
