#!/bin/bash

docker run --rm -v /home:/home -v /var/www:/var/www -u $UID livedar ./dar.py $@
