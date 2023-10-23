import requests
import json
from dar import Options
from libmediapipe.analyze import Analyzer, FFmpegVideoSource
import os
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
import time

analyze_executor = None
analyze_jobs = set()

app = Flask(__name__)

def get_stream(asset_id):
    url = "https://api.play.tv2.no/play/%s?stream=DASH" % asset_id # Assuming DASH support as default
    headers = {'Content-Type': 'application/json'}
    data = {
        "device": {
            "id": "1-1-1",
            "name": "Nettleser (HTML)"
        }
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return None

from xml.etree import ElementTree as ET

import xml.etree.ElementTree as ET

def analyze_video(stream_url, segment_length=5):
    source = FFmpegVideoSource(stream_url)
    pos = 0
    options = Options(False, None)
    analyzer = Analyzer(options)
    while not source.done:
        if pos % 20 == 0:
            print("Position: %d" % pos)
        res = analyzer.analyze_video(source, options, duration=5)
        for item in res:
            if not "end" in item:
                # TODO: find out if this is clever
                item["end"] = item["start"] + 1 / source.fps
            item["start"] += pos
            item["end"] += pos
        yield {
            "pos": pos,
            "data": res
        }
        pos += segment_length

def analyze_to_disk(stream_url, segment_length=5, dest="output/"):
    analyze_jobs.add(stream_url)
    try:
        for item in analyze_video(stream_url):
            fn = os.path.join(dest, "dar-%d.json" % item["pos"])
            with open(fn, "w") as f:
                json.dump(item["data"], f)
            with open(os.path.join(dest, "complete.txt"), "w") as f:
                f.write("ok")
    except Exception as e:
        print(e)
    finally:
        analyze_jobs.remove(stream_url)


@app.route('/hasDar', methods=['POST'])
def analyze():
    asset_id = request.json.get('assetId')
    if asset_id is None:
        return jsonify({'status': 'error', 'message': 'Missing asset_id'})
    
    # check if the completion file exists
    if os.path.isfile(os.path.join(args.output or "output/", f'{asset_id}', 'complete.txt')):
        return jsonify({'status': 'ok', 'message': 'Already analyzed'})
    elif asset_id in analyze_jobs:
        return jsonify({'status': 'ok', 'message': 'Already analyzing'})
    else:
        stream = get_stream(asset_id)
        if stream is not None:
            if stream['playback']['live']:
                return jsonify({'status': 'error', 'message': 'Live stream not supported'})
            stream_url = stream['playback']['streams'][0]['url']
            # start a new thread to analyze the video
            dest = os.path.join(args.output or "output/", f'{asset_id}')
            os.makedirs(dest, exist_ok=True)
            analyze_executor.submit(analyze_to_disk, stream_url, dest=dest)
            return jsonify({'status': 'ok'})
        else:
            return jsonify({'status': 'error', 'message': 'Unable to fetch stream URL'})
 
# if main
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", help="Output dir")
    parser.add_argument("asset_id", type=int, help="Asset ID", nargs='?')
    parser.add_argument("--serve", type=int, help="Serve on specified port")
    parser.add_argument("--threads", type=int, help="Number of threads", default=2)
    args = parser.parse_args()
    #exit(0)
    if args.serve is not None:
        analyze_executor = ThreadPoolExecutor(max_workers=args.threads)
        print("Serving on port %d" % args.serve)
        app.run(port=args.serve)
    else:
        stream = get_stream(args.asset_id)
        stream_url = stream['playback']['streams'][0]['url']
        print(stream_url)
        for item in analyze_video(stream_url):
            fn = os.path.join(args.output, "dar-%s-%d.json" % (args.asset_id, item["pos"]))
            with open(fn, "w") as f:
                json.dump(item["data"], f)
            print(item["data"])
    