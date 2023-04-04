# -----------------------------------------------------------------------------
#  Stage 0: install pyenv
# -----------------------------------------------------------------------------
FROM ubuntu:22.04 AS pyenv

ENV DEBIAN_FRONTEND=noninteractive

# install pyenv with pyenv-installer
COPY pyenv_dependencies.txt pyenv_dependencies.txt

ENV PYENV_GIT_TAG=v2.3.14

RUN apt-get update && \
    apt-get install -y $(cat pyenv_dependencies.txt)
RUN curl https://pyenv.run | bash
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# set python enviroment
COPY requirements.txt requirements.txt

ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN pyenv install 3.6.15 && \
    pyenv global 3.6.15
RUN pip install -r requirements.txt


# -----------------------------------------------------------------------------
#  Stage 1: user setup
# -----------------------------------------------------------------------------
FROM ubuntu:22.04

COPY --from=pyenv /root/.pyenv /root/.pyenv
ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
ENV PYTHONIOENCODING utf-8

RUN apt-get update && \
    apt-get install -y git

# RUN git clone https://github.com/jimenaRL/linate.git

COPY python/linate /linate

WORKDIR /linate


VOLUME /dbmigration/


# TO DO
# https://github.com/goodwithtech/dockle/blob/master/CHECKPOINT.md
