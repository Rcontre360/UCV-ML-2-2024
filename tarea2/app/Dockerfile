FROM python:3.11-buster
EXPOSE 8502

RUN apt-get update
RUN apt-get install -y libsndfile1-dev
RUN apt-get install -y ffmpeg

WORKDIR /usr/app
COPY . .

RUN pip install -r requirements-hw.txt

CMD ["streamlit", "run", "src/app.py", "--server.port", "8502"]