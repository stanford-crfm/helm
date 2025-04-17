#!/bin/bash

# Extra dependencies for SHELM when evaluating the following:
# Models: qwen/qwen-audio-chat

# This script fails when any of its commands fail.

set -e
# Directory to store ffmpeg binaries
FFMPEG_DIR=~/bin/ffmpeg

# Check if ffmpeg is already installed
if [ -x "$(command -v ffmpeg)" ]; then
  echo "ffmpeg is already installed and accessible."
  exit 0
fi

# Download ffmpeg if not present
mkdir -p $FFMPEG_DIR
curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | tar xJ -C $FFMPEG_DIR --strip-components=1

# Add to PATH
echo "export PATH=$FFMPEG_DIR" >> ~/.bashrc
source ~/.bashrc

echo "ffmpeg installation complete."
