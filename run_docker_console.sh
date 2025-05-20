#!/bin/sh

docker build . -t jetson-stereo
docker run -it --net=host  -v $(pwd):/usr/src/app  -p 5000:5000 jetson-stereo /bin/bash
