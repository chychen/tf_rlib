FROM tensorflow/tensorflow:latest-gpu-py3-jupyter 

RUN apt update
RUN apt install -y vim git
# jupyter
RUN pip install jupyterlab
RUN pip install ipywidgets
RUN jupyter nbextension enable --py widgetsnbextension
RUN curl -sL https://deb.nodesource.com/setup_11.x | bash -
RUN apt install -y nodejs
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

RUN pip install tqdm sklearn yapf

EXPOSE 6006
EXPOSE 8888
EXPOSE 5000


WORKDIR /

