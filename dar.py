#!/usr/bin/env python3
import requests
from xml.etree import ElementTree
import time
import os
import re
import copy
import queue
import tempfile
import threading
import json
import shutil
from datetime import datetime, timedelta

from libmediapipe.analyze import Analyzer


class DashDownloader:

    def __init__(self, mpd_url, referrer, user_agent=None):
        """
        Files are put in the target_mpd directory
        """
        if user_agent:
            self.user_agent = user_agent
        else:
            self.user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"

        self.mpd_url = mpd_url
        self.referrer = referrer
        self.target_dir = tempfile.mkdtemp()

        self.target_mpd = os.path.join(self.target_dir, "tmp.mpd")
        # Set up the requests session
        self.session = requests.Session()
        self.session.headers.update({'Referer': self.referrer, "User-Agent": self.user_agent})

        self.video_queues = []

    def __del__(self):
        try:
            shutil.rmtree(self.target_dir)
        except Exception as e:
            print("**** Failed to remove temporary directory")

    def add_video_processing_queue(self, queue):
        """
        Add a queue which will get notifications on each new added low res video segment, for processing
        """

        self.video_queues.append(queue)

    def append_segment(self, url, segnum, fd):
        print("Download", segnum)

        url = url.replace("$Number$", str(segnum))
        response = self.session.get(url, stream=True)
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    fd.write(chunk)
        else:
            print(f"Failed to download segment #{segnum}. HTTP Status code: {response.status_code}")

    def download_segment(self, url, segnum, filename):

        number_format_match = re.search(r'\$Number([^$]+)\$', url)
        if number_format_match:
            number_format = number_format_match.group(1)
            # Format the number using the extracted format string
            formatted_number = number_format % segnum
            url = re.sub(r'\$Number[^$]+\$', formatted_number, url)
            filename = re.sub(r'\$Number[^$]+\$', formatted_number, filename)

        else: 
            url = url.replace("$Number$", str(segnum))
            filename = filename.replace("$Number$", str(segnum))

        # If filenames have a "?" in them, we ignore the shit of of it
        filename = re.sub(r'\?.*$', '', filename)

        # print("Download segment", url, "to", filename)
        if segnum and os.path.exists(filename):
            return

        response = self.session.get(url, stream=True)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return filename
        else:
            print(f"Failed to download segment. HTTP Status code: {response.status_code}")
        return None

    def get_segments_from_mpd(self, mpd_url, mpd_dst):
        urls = []
        init = None
        response = self.session.get(mpd_url, stream=True)
        if response.status_code == 200:
            mpd_content = response.content
            root = ElementTree.fromstring(mpd_content)

            # Extract the availabilityStartTime from the MPD
            availability_start_time = root.get('availabilityStartTime')
            if availability_start_time:
                availability_start_time_dt = datetime.strptime(availability_start_time, '%Y-%m-%dT%H:%M:%SZ')
            else:
                print("Warning: availabilityStartTime not found in MPD")
                availability_start_time_dt = datetime.utcnow()

            ns = {'ns': 'urn:mpeg:dash:schema:mpd:2011'}

            for representation in root.findall(".//ns:Representation", namespaces=ns):
                segment_template = representation.find('ns:SegmentTemplate', namespaces=ns)
                media = segment_template.get('media')
                initialization = segment_template.get('initialization')
                timescale = int(segment_template.get('timescale'))
                start_number = int(segment_template.get('startNumber'))
                rep_id = representation.get("id", 0)
                media = media.replace("$RepresentationID$", rep_id)
                initialization = initialization.replace("$RepresentationID$", rep_id)
                segment_timeline = segment_template.find('ns:SegmentTimeline', namespaces=ns)
                for s in segment_timeline.findall('ns:S', namespaces=ns):
                    t = int(s.get('t'))
                    d = int(s.get('d'))
                    r = int(s.get('r'))
                    segnum = start_number + r

                    startts = (float(t) + (r * d)) / timescale
                    endts = startts + (float(d) / timescale)

                    # Compute the real-time timestamp using availabilityStartTime
                    real_start_time = (availability_start_time_dt + timedelta(seconds=startts)).timestamp()
                    real_end_time = (availability_start_time_dt + timedelta(seconds=endts)).timestamp()

                    s = {
                        "segnum": segnum,
                        "startts": real_start_time,
                        "endts": real_end_time,
                        "url": os.path.split(mpd_url)[0] + "/" + media,
                        "dst": media,
                        "init_dst": initialization,
                        "init_url": os.path.split(mpd_url)[0] + "/" + initialization
                        }
                    if representation.get("width"):
                        s["quality"] = representation.get("width") + "x" + representation.get("height")

                    urls.append(s)
        else:
            print("Error getting MPD file:", response.status_code, response.reason)
            raise SystemExit(5)
        return urls

    def process_segments(self, stop_event):
        """
        Process until stopped
        """

        last_segment = None  # We want to download *1* version of each segment

        while not stop_event.is_set():
            segments = self.get_segments_from_mpd(self.mpd_url, self.target_mpd)

            for segment in segments:

                if not os.path.exists(os.path.join(self.target_dir, segment["init_dst"])):
                    self.download_segment(segment["init_url"], 0, os.path.join(self.target_dir, segment["init_dst"]))

                segnum = segment["segnum"]
                if last_segment and segnum - last_segment["segnum"] > 1:
                    offset = last_segment["segnum"]
                    print(f"Lost some segments, {offset} -> {segment['segnum']}")

                    for s in range(1, segnum - offset):
                        fn = self.download_segment(segment["url"], offset + s, os.path.join(self.target_dir, segment["dst"]))
                        # Guess timestamps and stuff
                        seg = copy.copy(segment)
                        seg["segnum"] = offset + s
                        duration = last_segment["endts"] - last_segment["startts"]
                        seg["startts"] = last_segment["startts"] + (duration * s)
                        seg["endts"] = seg["startts"] + duration
                        self.notify_video(seg, fn)

                elif last_segment and last_segment["segnum"] == segnum:
                    # Already did this one
                    continue

                fn = self.download_segment(segment["url"], segment["segnum"], os.path.join(self.target_dir, segment["dst"]))
                last_segment = segment
                self.notify_video(segment, fn)


            # Sleep for a period of time before refreshing the MPD
            # Adjust the sleep time based on the MPD's refresh rate
            time.sleep(1.5)

    def notify_video(self, segment, filename):
        if not filename:
            return

        # If we hit any queues, append and notify
        for q in self.video_queues:
            s = copy.copy(segment)
            s["path"] = re.sub(r'\?.*$', '', filename)
            s["initfile"] = os.path.join(os.path.split(filename)[0], re.sub(r'\?.*$', '', segment["init_dst"]))
            q.put(s)
            print("  - queueing")


class Options:
    def __init__(self, align_iframes, cast):
        self.selfie = False
        self.tile = None
        self.iframes = align_iframes
        self.startts = 0
        self.endts = None
        self.cast = cast
        self.show = False


class LiveAnalyzer(threading.Thread):

    def __init__(self, queue, stop_event, target_file):
        threading.Thread.__init__(self)
        self.queue = queue
        self.stop_event = stop_event

        self.options = Options(False, None)
        self.analyzer = Analyzer(self.options)
        self.target_file = target_file

    def run(self):

        aux_state = []
        last_seg = 0
        print(" **** Live analyzer running *** ")
        while not self.stop_event.is_set():

            try:
                segment = self.queue.get(block=True, timeout=5.0)
            except Exception as e:
                print("No segments to process", e)
                continue

            if last_seg and segment["segnum"] - last_seg > 1:
                print("   **** Skipped segment")
            if last_seg and segment["segnum"] == last_seg:
                continue
            last_seg = segment["segnum"]

            print("ANALYZING segment", segment)

            # Create a temporary named file

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                with open(segment["initfile"], 'rb') as init_file, open(segment["path"], 'rb') as data_file:
                    temp_file.write(init_file.read())
                    temp_file.write(data_file.read())
                temp_file_path = temp_file.name

                # Cleanup the segment file
                os.remove(segment["path"])
                res = self.analyzer.analyze_video(temp_file_path, self.options)

                # Copy the video segment for debugging
                shutil.copy(temp_file_path, "/tmp/segment.mp4")

                # As the analyzer starts on timestamp 0, we must update the times
                for info in res:
                    if info["start"] > segment["endts"] - segment["startts"]:
                        # Timestamp is NOT relative, just keep it
                        continue
                    info["start"] += segment["startts"]

                    if "end" in info:
                        info["end"] += segment["startts"]
                    else:
                        info["end"] = info["start"] + 1

            if len(res) == 0:
                print(" ** No DAR points detected")

            # We now got results, append to a running window of state and update the target file
            aux_state.extend(res)
            # We run with a 120 second window
            cutoff = segment["startts"] - 120
            aux_state = [d for d in aux_state if d["start"] >= cutoff]

            # We now save the state too
            with open(self.target_file+".tmp", "w") as f:
                json.dump(aux_state, f)
            os.rename(self.target_file+".tmp", self.target_file)

            if len(aux_state) == 0:
                print("No DAR state")
            else:
                print("DAR info from", aux_state[0]["start"], "to", aux_state[-1]["start"])


if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-s", "--source", dest="src", help="Input URL (MPD)", required=True)

    parser.add_argument("-o", "--dst", dest="dst", help="Output file (aux_file)", required=True)
    parser.add_argument("-r", "--referrer", dest="referrer", help="Referrer", default="")
    options = parser.parse_args()

    stop_event = threading.Event()

    # Downloader for dash segments
    downloader = DashDownloader(options.src, options.referrer)

    # Queue to get new segments to analyze
    segment_queue = queue.Queue()
    downloader.add_video_processing_queue(segment_queue)

    # DAR analyzer
    aux_dst = options.dst
    analyzer = LiveAnalyzer(segment_queue, stop_event, aux_dst)
    analyzer.start()

    # This blocks
    try:
        downloader.process_segments(stop_event)
    finally:
        stop_event.set()
