# we use a multi-stage build, see:
# - https://vsupalov.com/build-docker-image-clone-private-repo-ssh-key/
# - https://blog.realkinetic.com/building-minimal-docker-containers-for-python-applications-37d0272c52f3

FROM python:3.7 as builder
ENV PYTHONUNBUFFERED 1

RUN python3 -m pip install -IU pip poetry shiv

RUN mkdir /build
WORKDIR /build

ADD ./src ./src
COPY ./pyproject.* ./

RUN DIST=$(poetry build | grep whl | cut -d " " -f 4) \
    && shiv \
        -c pypare \
        -o pypare \
        dist/${DIST}


FROM python:3.7-slim as base
ENV PYTHONUNBUFFERED 1
ENV PYPARE_LOG_LEVEL INFO
ENV PYPARE_HOST 0.0.0.0
ENV PYPARE_PORT 3141
ENV PYPARE_CACHE_DIR /data
ENV PYPARE_CACHE_TIMEOUT 86400

RUN mkdir /microservice ${PYPARE_CACHE_DIR}
WORKDIR /microservice

COPY --from=builder /build/pypare .
# extract zip app
RUN ./pypare --version

EXPOSE ${PYPARE_PORT}
ENTRYPOINT ["/microservice/pypare"]
CMD ["pypi"]
