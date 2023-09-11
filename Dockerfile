FROM python:3.10


# Install cryocloud dependencies
RUN pip install --upgrade pip

RUN apt update; apt install -y libgl1
RUN pip3 install mediapipe imagehash fuzzywuzzy Levenshtein requests

ADD libmediapipe /livedar/libmediapipe/
ADD dar.py /livedar/
ADD entrypoint.sh /livedar/

WORKDIR /livedar

ENV HOME=/tmp

# ENTRYPOINT ["/livedar/entrypoint.sh"]
