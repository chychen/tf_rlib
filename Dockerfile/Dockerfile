FROM tensorflow/tensorflow:latest-gpu-py3-jupyter 

RUN apt update --fix-missing
RUN apt install -y vim git screen wget
# conda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda --version
# rapids
RUN conda install -y -c rapidsai -c nvidia -c conda-forge \
    -c defaults rapids=0.11 python=3.6
# jupyter
RUN pip install jupyterlab
RUN pip install ipywidgets
RUN jupyter nbextension enable --py widgetsnbextension
RUN curl -sL https://deb.nodesource.com/setup_11.x | bash -
RUN apt install -y nodejs
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

RUN pip install tqdm sklearn yapf pytz tensorflow_datasets tensorflow-addons pandas seaborn pillow

EXPOSE 6006
EXPOSE 8888
EXPOSE 8000

WORKDIR /

CMD nohup tensorboard --logdir /results --bind_all --port 6006 & jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token='' --allow-root --NotebookApp.allow_origin='*' --notebook-dir='/'


