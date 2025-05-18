FROM dustynv/l4t-pytorch:r36.4.0

RUN pip install --upgrade pip

WORKDIR /app
COPY requirements-jetson.txt /app/requirements-jetson.txt
RUN pip install -r requirements-jetson.txt
COPY . /app
CMD ["python3", "stereo_cli.py",  "sample/left_images/2018-07-11-14-48-52_2018-07-11-14-50-22-775.jpg", "sample/right_images/2018-07-11-14-48-52_2018-07-11-14-50-22-775.jpg", "--output", "output.png", "--benchmark"]
