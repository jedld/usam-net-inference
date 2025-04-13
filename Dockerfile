FROM dustynv/l4t-pytorch:r36.4.0

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . /app

WORKDIR /app

CMD ["python", "stereo_cli.py", "--left_img", "left.png", "--right_img", "right.png", "--output", "output.png"]