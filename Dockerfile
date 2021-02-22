FROM python:3.6.13-alpine3.13

EXPOSE 8265

VOLUME /workspace/bin
VOLUME /workspace/configs
VOLUME /workspace/riddbot

# General dependencies
RUN apk add --no-cache git curl jq make

# PyTorch dependencies
RUN apk add --no-cache libexecinfo libgomp py3-cffi linux-headers openblas-dev

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

COPY . /workspace
WORKDIR /workspace

RUN make clean
RUN make python_deps

ENTRYPOINT make
