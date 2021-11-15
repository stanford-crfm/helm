# Introduction

We provide a single unified interface into accessing large language models through API (e.g., GPT-3, Jurassic).

To start the server:

    python src/server.py

To update the server:

    rsync -arvz requirements.txt src credentials.conf crfm-models.stanford.edu:benchmarking
