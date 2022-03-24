FROM python:3.8-slim

WORKDIR back

RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget \
      python3-tk &&\
      rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN (apt-get autoremove -y; \
     apt-get autoclean -y)

CMD ["python", "app.py"]
