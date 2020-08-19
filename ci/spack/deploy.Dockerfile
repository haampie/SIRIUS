# The build image should have dependencies installed
ARG BUILD_BASE=ubuntu:18.04

# This is stripped down deployment image
ARG DEPLOY_BASE=ubuntu:18.04

FROM $BUILD_BASE as builder

# This is the version of sirius we want to install
ARG SPEC

# This is the sources of sirius we copy into build container
ARG SOURCES=.

COPY "${SOURCES}" /sources

SHELL ["/bin/bash", "-c"]

# Install sirius
RUN spack --color=always -e ci dev-build --source-path /sources $SPEC

# Bundle everything
RUN . /opt/spack/share/spack/setup-env.sh && \
    apt-get update -qq && \
    apt-get install -qq wget tar && \
    cd /root && \
    wget -q https://github.com/haampie/libtree/releases/download/v1.2.0/libtree_x86_64.tar.gz && \
    tar -xzf libtree_x86_64.tar.gz && \
    rm libtree_x86_64.tar.gz && \
    ln -s /root/libtree/libtree /usr/local/bin/libtree && \
    spack load sirius && \
    libtree -d /root/sirius.bundle --chrpath `which sirius.scf`

# Create a stripped down image
FROM $DEPLOY_BASE

COPY --from=builder /root/sirius.bundle /root/sirius.bundle
COPY --from=builder /sources /sources

# Make nvidia happy
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=7.5"

# Make sirius.scf available in the path
ENV PATH="/root/sirius.bundle/usr/bin:$PATH"

# Make sarus happy
RUN echo "/root/sirius.bundle/usr/lib/" > /etc/ld.so.conf.d/sirius.conf && ldconfig