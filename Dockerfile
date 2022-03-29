FROM 10jqkaaicubes/cuda:10.0-py3.7.9

COPY ./ /home/jovyan/opinion_classification

RUN cd /home/jovyan/opinion_classification  && \
    python -m pip install -r requirements.txt 