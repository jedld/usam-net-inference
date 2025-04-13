FROM dustynv/l4t-pytorch:r36.4.0

RUN pip install --upgrade pip

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

CMD ["python3", "stereo_cli.py", "--left_img", "left.png", "--right_img", "right.png", "--output", "output.png"]