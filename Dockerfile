# Use an official Python image as the base
FROM python:3.8

# Set working directory
WORKDIR /app

# Install crfm-helm package
# RUN pip install crfm-helm

# Copy run_specs.conf into the container
COPY . /app

# Copy entrypoint script into the container
# COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set the default command to run the entrypoint script
CMD ["/app/entrypoint.sh"]
