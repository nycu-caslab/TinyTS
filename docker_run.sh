#! /usr/bin/bash
sudo docker run --name="tinyts_container" \
    -ti --privileged \
    --device="/dev/ttyACM0" \
    -v /dev/disk/by-id:/dev/disk/by-id \
    -v /dev/serial/by-id:/dev/serial/by-id \
    -v /run/udev:/run/udev:ro \
    -v $(pwd)/TFLM_CMSISNN_4_1_0:/TinyTS/TFLM_CMSISNN_4_1_0 \
    -v $(pwd)/models:/TinyTS/models \
    -v $(pwd)/scripts:/TinyTS/scripts \
    -v $(pwd)/TinyTS:/TinyTS/TinyTS \
    tinyts