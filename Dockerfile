FROM nvidia/cuda:8.0-cudnn6-runtime-ubuntu16.04

RUN apt-get update ; \
    apt-get install -y software-properties-common

RUN add-apt-repository -y ppa:jonathonf/python-3.6 ; \
    apt-get update

RUN apt-get install --fix-missing -y \
    python3.6 python3.6-dev python3.6-venv \
    libgeos-dev  pkg-config libfreetype6-dev \
    libsm6 libxext6 libxrender-dev libcurl4-openssl-dev\
    libssl-dev gcc

ADD https://bootstrap.pypa.io/get-pip.py /tmp/get-pip.py
RUN ln -s /usr/bin/python3.6 /usr/local/bin/python3 ; \
    python3 /tmp/get-pip.py ; \
    pip install --upgrade pip; \
    python3.6 -m pip install --ignore-installed pycurl

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3.6 install -r requirements.txt

COPY . .

ENTRYPOINT [ "python3", "run_demo_server.py" ]

EXPOSE 8769

RUN apt-get install -y curl
HEALTHCHECK --interval=5s --timeout=10s --retries=20 \
    CMD curl --fail http://localhost:8769 || exit 1