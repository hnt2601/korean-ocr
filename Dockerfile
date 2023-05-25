FROM nvidia/cuda:10.1-base-ubuntu18.04 as base

LABEL maintainer hoangnt@meditech

ARG PADDLE_IPADDR=paddle-detector
ARG PADDLE_PORT=9292
ARG PORT=8009
ARG HOST="0.0.0.0"
ARG LOGIN_API="https://go.iview.vn/api/v1/login"

# Install dependence
RUN apt-get update && apt-get upgrade -y && apt-get install -y python3 \
    python3-pip python3-dev \
    libcurl4-openssl-dev libb64-dev \
    iputils-ping libsm6 libxext6 libxrender-dev \
    libgl1-mesa-glx \
    build-essential cmake

RUN apt-get autoremove


# config time zone
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN echo "Asia/Ho_Chi_Minh" > /etc/timezone
RUN rm -f /etc/localtime
RUN dpkg-reconfigure -f noninteractive tzdata

# Setup ENV
#RUN python3 -m pip install --upgrade pip
#COPY ./requirements.txt /tmp/
#RUN python3 -m pip install -r /tmp/requirements.txt
#RUN rm -rf /tmp
#RUN rm -rf /var/lib/apt/lists/*

ENV PADDLE_IPADDR=${PADDLE_IPADDR}
ENV PADDLE_PORT=${PADDLE_PORT}
ENV PORT=${PORT}
ENV HOST=${HOST}

EXPOSE ${PORT}
RUN mkdir /workspace
COPY ./ /workspace/
WORKDIR /workspace

RUN python3 -m pip install -r requirements.txt

CMD ["python3", "main.py"]
