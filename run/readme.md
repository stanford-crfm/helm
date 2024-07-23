# Install the thing in install-dev.sh

HF_TOKEN=<> OPENAI_API_KEY=<> helm-run --conf-paths run_entries.conf --suite v1 --max-eval-instances 16 --models-to-run fireworks/llama3-405fp16 
helm-summarize --suite v1
helm-server
