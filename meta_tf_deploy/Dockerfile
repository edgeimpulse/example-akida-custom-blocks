FROM python:3.8-slim-bullseye
RUN python3 -m pip install --upgrade pip
COPY . ./

RUN python3 -m pip install akida
RUN python3 -m pip install -r requirements.txt

COPY . ./
# This is the script our application should run (-u to disable buffering)
ENTRYPOINT ["python3", "-u", "build.py"] 
