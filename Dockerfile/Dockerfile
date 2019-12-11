FROM tensorflow/tensorflow:latest-gpu-py3-jupyter 

RUN apt update
RUN apt install -y vim git screen wget
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


