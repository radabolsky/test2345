FROM python:3.7
EXPOSE 8501
WORKDIR /medical
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY . .
CMD streamlit run main.py