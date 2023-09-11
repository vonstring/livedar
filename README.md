# LiveDAR

Dynamic aspect ratio for live MPEG Dash

Only works for VERY SELECTED FEW dash streams for now!


## Server side:

On the server side, we connect to an existing dash stream (tested with TV2 live news and generic ffmpeg).

Example ffmpeg command:
```
ffmpeg -f v4l2 -i /dev/video0 -map 0 -b:v 1M -c:v libx264 -profile:v main -pix_fmt yuv420p -r 30 -g 60 -keyint_min 30 -sc_threshold 0 -f dash -min_seg_duration 5000  -use_template 1 -use_timeline 1 -movflags +frag_keyframe -segment_format mp4 <destination.mpd>
```

Each segment is downloaded and analyzed locally, the end results are timestamped according to the internal timestamps in the video, and written periodically to the destination file provieded to dar.py. This file should be periodically read by the clients to get updated data.

The data is timed positional data, with "start" and "end" timestamps as well as "posX" and "posY" in percent of the determined center point.

```
[
    {
        "animate": true,
        "box":
        {
            "bottom": 66,
            "left": 40,
            "right": 62,
            "top": 35
        },
        "end": 1416.1,
        "heading": "askew",
        "movement": 0,
        "name": "face",
        "orientation":
        [
            0.2163309375247488,
            0.5839974925478305
        ],
        "pos":
        [
            48,
            53
        ],
        "posX": 48,
        "posY": 53,
        "size": 6.891763151085684,
        "start": 1416.0333333333333,
        "value": 0.9042618274688721
    },
    {
        "animate": true,
        "box":
        {
            "bottom": 66,
            "left": 40,
            "right": 63,
            "top": 35
        },
        "end": 1416.1666666666667,
        "heading": "askew",
        "movement": 0.0,
        "name": "face",
        "orientation":
        [
            0.20898296420753504,
            0.5784380643052168
        ],
        "pos":
        [
            48,
            53
        ],
        "posX": 48,
        "posY": 53,
        "size": 6.99161016479195,
        "start": 1416.1,
        "value": 0.8909662961959839
    },
    ....
]
```



### Building

```
docker build -t livedar .
```


### Running
```

export URL="...mpd"
export REFERRER="..."

./run.sh -s $URL -r $REFERRER -o /var/www/html/live/mydestination.mpd $@

```


Alternative:
```

docker run -it --rm -v /var/www:/var/www -u $UID livedar /bin/bash

./dar.py -s <URL> -r <REFERRER> -o <DESTINATION>
```


## Player

In the html directory, a player is provided. It wraps the Shaka player, and takes in two URLs - the first is the manifest (mpd) file, the second is the DAR json file, as produced by dar.py.

