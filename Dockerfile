FROM python:3.10
RUN pip install setuptools==76.1.0 uv==0.6.3
RUN pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
RUN uv pip install --system -r https://raw.githubusercontent.com/stanford-crfm/helm/refs/heads/main/requirements.txt
