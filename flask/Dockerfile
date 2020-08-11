FROM python:3.7

ADD requirements.txt /deploy/
WORKDIR /deploy/
RUN pip install --no-cache-dir -r requirements.txt

ADD html_skeleton /deploy/html_skeleton
ADD model /deploy/model
ADD static /deploy/static
ADD templates /deploy/templates
ADD DFdetective.py /deploy/
ADD DFdetective_utils.py /deploy/
ADD EfficientNet.py /deploy/
ADD EfficientNet_utils.py /deploy/
ADD FaceDetectionModule.py /deploy/
ADD app.ini /deploy/
ADD run.py /deploy/

EXPOSE 5000
CMD ["uwsgi", "app.ini"]
