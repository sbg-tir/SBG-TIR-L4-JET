# Using minimal Ubuntu base image
FROM ubuntu:20.04 as base
ENV HOME /root

# Update Ubuntu and install required dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    wget \
    python3 \
    python3-pip

FROM base as installation

# Creating directory inside container to store PGE code
RUN mkdir /app
RUN mkdir /pge
# Copying current snapshot of PGE code repository into container
COPY . /app/
COPY ./PGE/*.sh /pge/
RUN chmod +x /pge/*.sh

# Running pip install using pyproject.toml
COPY pyproject.toml /app/pyproject.toml
WORKDIR /app
RUN pip3 install .

FROM installation as build
RUN rm -rvf build; rm -rvf dist; rm -rvf *.egg-info; rm -rvf CMakeFiles
