#!/usr/bin/env python3
import cv2
import mediapipe as mp
import json
import numpy as np
from argparse import ArgumentParser
import os
from operator import itemgetter, attrgetter
import functools
import copy
import math
import random
import subprocess

from PIL import Image
from io import BytesIO
import imagehash
from fuzzywuzzy import StringMatcher

from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector
import time


try:
    from .Analyzers import FaceAnalyzer, PoseAnalyzer
except:
    from Analyzers import FaceAnalyzer, PoseAnalyzer


def dist_a_b(a, b):
    dist = math.sqrt(math.pow(a["x"] - b["x"], 2) + math.pow(a["y"] - b["y"], 2))
    return dist


def extract_text(img_data, lang=None):

    cmd = ["tesseract"]
    if lang:
        cmd.extend(["-l", lang])
    cmd.extend(["--dpi", "70"])
    cmd.extend(["-", "-"])
    text = subprocess.check_output(cmd, stderr=subprocess.PIPE, input=img_data)
    return text.strip().decode("utf-8")


def clean_up(string, names_only=True):
    """
    Clean up a string - remove any tiny words, any punctuation, pipes etc
    We also only allow one line of text for now (the longest one)
    """
    lines = []

    for line in string.split("\n"):
        words = []
        is_crap = True

        for w in line.split(" "):
            if len(w) <= 2:
                continue

            if not w.isalnum():
                continue

            if len(w) > 3:
                is_crap = False

            if names_only and w[0] != w[0].upper():
                continue

            if names_only:
                bad = False
                for i in range(1, len(w)):
                    if w[i] == w[1].upper():
                        bad = True
                if bad:
                    continue
            words.append(w)
        if not is_crap:
            lines.append(words)

    if len(lines) == 0:
        return ""

    # Which line is longer?
    def get_length(a):
        return len(a)
    lines.sort(key=get_length, reverse=True)

    words = lines[0]
    res = " ".join(words).strip()
    if len(res) < 7:
        return ""

    return res


class Face:
    def __init__(self, max_detections=None):
        self.id = random.randint(0, 10000)
        self.position = (0, 0)
        self.last_seen_ts = 0
        self.max_detections = max_detections
        self.detections = []
        self.movement = 0
        self.startts = None
        self.endts = None

    @staticmethod
    def is_same(face1, face2):

        if not face1 or not face2:
            return False

        # Calculate the distance
        dist = dist_a_b({"x": face1["posX"], "y": face1["posY"]},
                        {"x": face2["posX"], "y": face2["posY"]});

        if dist > 8:  # Percent of the screen in difference
            return False

        return True

    def get_position(self):
        if len(self.detections) > 0:
            return (self.detections[0][0]["posX"], self.detections[0][0]["posY"])
        else:
            return (50, 50)



    def get_positions(self, startts, endts):
        """
        Return all positions between the given times
        """
        detections = []
        for detection, ts in self.detections:

            if ts >= startts and ts <= endts:
                detections.append(detection)
            detection["pos"] = (detection["posX"], detection["posY"])
            detection["animate"] = True  # These are frame updates

        return detections


    def hasExpired(self, ts, grace_period=0.3):
        if ts - self.last_seen_ts > grace_period:
            # print(self.id, "EXPIRED", ts, "last seen", self.last_seen_ts, ts - self.last_seen_ts, "s ago")
            return True

        self._recalculate_movement(ts)

        return False

    def is_same_face(self, detection):
        if len(self.detections) == 0:
            print(" INTERNAL - no faces")
            return False

        return self.is_same(self.detections[-1][0], detection)

    def add_detection(self, detection, ts):
        """
        Add detection(a face)
        """
        self.last_seen_ts = ts

        if len(self.detections) > 0:
            self.detections[-1][0]["end"] = ts

        detection["start"] = ts

        self.detections.append((detection, ts))

        if self.max_detections and len(self.detections) > self.max_detections:
            self.detections.pop(0)

        # Update movement factor
        self._recalculate_movement(ts)

    def _recalculate_movement(self, ts):
        """
        Calculate the movement of this face for the last samples
        """
        if len(self.detections) < 2:
            self.movement = 0
            return

        # If we haven't seen this face for a tiny bit, assume them as standing still
        if ts - self.detections[-1][1] > 0.100:
            self.movement = 0
            return

        last_face = self.detections[0][0]
        total_distance = max_distance = 0
        for face, _ in self.detections[1:]:
            dist = dist_a_b({"x": last_face["posX"], "y": last_face["posY"]},
                            {"x": face["posX"], "y": face["posY"]})
            total_distance += dist
            max_distance = max(max_distance, dist)

        avg_movement = total_distance / (len(self.detections) - 1)


        # avg_movement_ts = total_distance / (ts - self.detections[0][1])

        # If we haven't seen this face for a tiny bit, assume them as standing still


        # print(self.id, "dist", total_distance, "ts:%.2f, from:%.2f, delta:%.2f" % (ts, self.detections[0][1], ts -self.detections[0][1]),
        #      "avg", avg_movement, "avgts")

        self.movement = avg_movement


    def __str__(self):

        pos = (self.detections[-1][0]["posX"], self.detections[-1][0]["posY"])
        return "[%d]: %.02f-%.02f - movement %.1f"% (self.id, self.startts, self.endts, self.movement) + str(pos)


class Person:

    def __init__(self):
        self._last_change_ts = None
        self.poses = []
        self.current_state = {}
        self.last_ts_verified = None

    # Calculate distance between two points
    def dist_a_b(self, a, b, point=None):
        if point:
            a = a[point]
            b = b[point]
        dist = math.sqrt(math.pow(a["x"] - b["x"], 2) + math.pow(a["y"] - b["y"], 2))
        return dist

    def hasExpired(self, ts):
        if self.last_ts_verified and ts - self.last_ts_verified > 0.5:
                return True
        return False

    def isPerson(self, ts, pose, max_distance=100):
        """
        """

        # Calculate a distance between the last pose and this one
        if len(self.poses) == 0:
            raise SystemExit("Can't compare pose to person without pose")

        diff = self._calculate_pose_difference(self.poses[-1], pose)

        physical_difference = self.dist_a_b(self.poses[-1], pose, "nose")

        if ts - self.last_ts_verified > 0.5:
            print("Too long since I was seen, not me (I should be removed)")
            return False

        if physical_difference < max_distance:
            # Nose closer than 50px, this is the same person
            self.last_ts_verified = ts
            return True

        print("--------")
        print("DIFF", diff, "PH", physical_difference)
        print("--------")

        return False

    def __str__(self):
        return "Person (%s): %d poses, current state %s" % (self.last_ts_verified, len(self.poses), str(self.current_state))

    def _calculate_pose_difference(self, pose1, pose2):
        """
        Come up with a measure of change and what kind it might be
        """
        areas = {
            "hands": ["left_wrist", "right_wrist"],
            "feet": ["left_ankle", "right_ankle"]
        }

        delta = {"pose1": {}, "pose2": {}, "delta": {}}
        for area in areas:
            # Get the lengths of each pose
            ptname = areas[area]
            print("ptname", ptname)
            delta["pose1"][area] = self.dist_a_b(pose1[ptname[0]], pose1[ptname[1]])
            if (pose2):
                delta["pose2"][area] = self.dist_a_b(pose2[ptname[0]], pose2[ptname[1]])

                delta["delta"][area] = abs(delta["pose1"][area] - delta["pose2"][area])

        print("Distances", delta)
        return delta

    @staticmethod
    def is_garbage(pose):
        """
        Is this pose just garbage?
        """

        # If the head is under the elbow, ankle over the
        # shoulder etc, ignore ( higher numbers are higher on screen)

        points = ["elbow", "ankle", "shoulder", "hip"]
        positions = {}
        for point in points:
            positions[point] = min(pose["left_%s" % point]["y"], pose["right_%s" % point]["y"])

        positions["nose"] = pose["nose"]["y"]

        # positions = { point: min(pose["left_%s" % point]["y"], pose["right_%s" % point]["y"]) for point in points}

        print("Positions", positions)

        if positions["ankle"] > positions["shoulder"]:
            return True
        if positions["ankle"] > positions["elbow"]:
            return True
        if positions["ankle"] > positions["hip"]:
            return True
        if positions["hip"] > positions["shoulder"]:
            return True
        if positions["head"] < positions["shoulder"]:
            return True
        if positions["elbow"] < positions["hip"]:
            return True
        if positions["shoulder"] > positions["nose"]:
            return True
        if positions["shoulder"] > positions["nose"]:
            return True

        return False


    def analyze(self, ts, pose):

        # If this pose is the same timestamp as the last one, we use the better one
        # based on number of visible bits
        if len(self.poses) > 0:
            if ts == self.poses[-1]["ts"]:
                print("DUPE - which is better?")
                return  # Just return for now

        self.current_state = {"tags":[]}
        # print("  *** ANALYZING ***", pose)
        # Do we see the whole person?
        limit = 1.5
        visible = {"lower": "ankle", "mid": "elbow", "top": "shoulder"}
        visibility = {what: pose["right_%s" % visible[what]]["visibility"] + \
                            pose["left_%s" % visible[what]]["visibility"] for what in visible}

        if visibility["top"] > limit and visibility["mid"] > limit and visibility["lower"] > limit:
            self.current_state["posetype"] = "full"
        elif visibility["top"] > limit and visibility["mid"] > limit:
            self.current_state["posetype"] = "bust"
        elif visibility["top"] > limit:
            self.current_state["posetype"] = "head"
        else:
            self.current_state["posetype"] = "uncertain"


        # Check if they are waving hands
        wy = min(pose["right_wrist"]["y"], pose["left_wrist"]["y"])
        ey = min(pose["right_elbow"]["y"], pose["left_elbow"]["y"])
        sy = min(pose["right_shoulder"]["y"], pose["left_shoulder"]["y"])
        print("Wrist", wy, "Elbow", ey, "Shoulder", sy, "cutoff", (ey + (0.5*(sy - ey))))
        if ey < sy:
            self.current_state["tags"].append("HandsUp")
            print("Hands up")
        if wy < (ey + (0.5*(sy - ey))):
            print("Waving arms", wy, ey)
            self.current_state["tags"].append("WavingArms")
        else:
            print("Not waving")


        # If we have other poses, check distance to the last one
        if len(self.poses) > 0:
            distances = self._calculate_pose_difference(self.poses[-1], pose)
            print("Updated pose, distances:", distances)
        # Are the poses very similar ("stationary")

        self.current_state["pos"] = (pose["nose"]["x"], pose["nose"]["y"])

        # Remember this pose
        self.poses.append(pose)
        self.last_ts_verified = pose["ts"]




class Analyzer:

    def __init__(self, options):
        self.options = options
        self._face_analyzer = FaceAnalyzer(options)
        self._pose_analyzer = PoseAnalyzer(options)
        self._detector = AdaptiveDetector(adaptive_threshold=3.0)
        self.people = []
        self.faces = {}
        self._global_frame_num = 0

    def make_segments(self, split_x, split_y, shape, padding=0.1):
        height, width, channels = shape
        tile_size = (width/split_x, height / split_y)
        segments = []
        pad_x = tile_size[0] * padding
        pad_y = tile_size[1] * padding
        for y in range(split_y):
            for x in range(split_x):
                segments.append({
                    "xmin": max(0, round((x * tile_size[0]) - pad_x)),
                    "xmax": min(width, round(((x + 1) * tile_size[0]) + pad_x)),
                    "ymin": max(0, round((y * tile_size[1]) - pad_y)),
                    "ymax": min(height, round(((y + 1) * tile_size[1]) + pad_y))
                })
        return segments

    def uncrop(self, what, asset, segment, shape):
        """
        Returns a copy that is uncropped
        """
        # print("UNCROP", asset)
        asset = copy.copy(asset)

        if what == "faces":
            for faces in asset:
                for box in faces["box"]:
                    pass

                    # These are relative, percentage. :/

        elif what == "poses":
            for poses in asset:
                for name in poses:
                    pose = poses[name]
                    pose["x"] += segment["xmin"]
                    pose["y"] += segment["ymin"]

                    if name == "nose":
                        print("NOSE", pose);

        return asset


    def get_iframes(self, path, transcode=True):  # We can't assume iframes are on scene changes, transcode to get that

        # Do we already have the index file?
        idxfile = os.path.splitext(path)[0] + ".idx"

        if os.path.exists(idxfile):
            with open(idxfile, "r") as f:
                return json.load(f)

        # Transcode?
        tf = path
        if transcode:
            print("*** Transcoding for iframes")
            tf = os.path.join(os.path.split(tf)[0], "tmp_%s.mp4" % random.randint(0, 4000000000))
            cmd = "ffmpeg -i '%s' -s 128x76 -an %s" % (path, tf)
            print(cmd)
            res = subprocess.getoutput(cmd)
            print(res)

        print("Detecting iframes for", path)
        FFPROBE="ffprobe -v error -skip_frame nokey -show_entries frame=pkt_pts_time -select_streams v -of csv=p=0 "

        index = {}
        cmd = FFPROBE + tf
        res = subprocess.getoutput(cmd)

        print(res)
        index = [float(x) for x in res.split("\n")]

        with open(idxfile, "w") as f:
            json.dump(index, f)

        if transcode:
            os.remove(tf)
        return index


    def analyze(self, image, ts, options):

        retval = {}
        # Start by looking for faces

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        try:
            if self.options.selfie:
              image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            else:
              image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print("Bad image, stopping")
            return None

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False


        if (not options.tile):
            faces = self._face_analyzer.fine_detection(image)
            if 0 and faces:
                print("FULL FACES")
                for face in faces:
                    print(face)


            # Should analyze poses of each face
            # poses = self._pose_analyzer.detection(image)
            # print("POSES", poses)
            poses = []

            retval["faces"] = faces
            return retval

        else:

            segments = []
            height, width, channels = image.shape

            # Try to rather just split the image into 10x4 with a bit of overlap
            segments = self.make_segments(7, 2, image.shape, padding=0.2)

            # for segment in segments:
            #     print(segment)

            faces = []
            poses = []

            # Go through the segments and do a finer face detection
            for segment in segments: 
                subimg = image[segment["ymin"]:segment["ymax"],
                               segment["xmin"]:segment["xmax"]]  # .copy(order='C')
                subimg = np.asarray(subimg, order='C')

                # subimg = image[0:height, 300:1500].copy(order='C')
                subimg.flags.writeable = False

                # subimg = image
                if 1:
                    subfaces = self._face_analyzer.fine_detection(subimg)

                    if len(subfaces) == 0:
                        # print("  No fine faces found!")
                        continue

                # print("  ", len(subfaces), "faces found")
                # for f in subfaces:
                #    print(f)

                # We now also do pose analysis on this segment - there are very likely people there
                subposes = self._pose_analyzer.detection(subimg)

                # print("  ", len(subposes), "poses detected")

                # uncrop!
                faces.append(self.uncrop("faces", subfaces, segment, image.shape))
                poses.append(self.uncrop("poses", subposes, segment, image.shape))


        detections = {"faces": faces, "poses": poses}

        print("Detect people")
        people = self.detect_people(ts, detections)

        return detections


    def detect_people(self, ts, detections, min_distance=60):
        """
        Try to detect actual people
        """

        print(ts, "Detecting people in ", len(detections["poses"]), "poses")
        # We take poses first

        for poses in detections["poses"]:
            for pose in poses:
                print("POSE IS", pose)
                if Person.is_garbage(pose):
                    continue

                pose["ts"] = ts
                found = False
                for person in self.people:
                    if person.hasExpired(ts):
                        self.people.remove(person)
                        continue

                    if person.isPerson(ts, pose):
                        print("Found person")
                        found = True
                        person.analyze(ts, pose)
                        break

                if not found:
                    person = Person()
                    person.analyze(ts, pose)
                    self.people.append(person)

        print(ts, "Detected people", len(self.people))
        for i, person in enumerate(self.people):
            print(i, person)


    def analyze_images(self, images, options):
        ret = []
        for idx, path in enumerate(images):
            image = cv2.imread(path)
            ret.append(self.analyze(image, idx, options))

        ret2 = []
        for img in ret:
            ret2.append(self.choose_center([{"analysis": img}]))
        return ret2

    def check_people(self, img, possible_name, detection, textbox, ts, target_dir):


        target_dir = os.path.join(target_dir, "faces")
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        for name in list(self.faces):
            if (StringMatcher.distance(possible_name, name) < 7):
                print(" Found person '%s' based on '%s'" % (name, possible_name))
                return False
            if (StringMatcher.ratio(possible_name, name) > 0.7):
                print(" Found person '%s' based on '%s'" % (name, possible_name))
                return False
            if possible_name.find(name) > -1:
                return False
            if name.find(possible_name) > -1:
                return False

        # Add the person
        self.faces[possible_name] = {"detection": detection, "img": None, "ts": ts}

        i = Image.fromarray(img)
        fn = os.path.join(target_dir, "%s.png" % possible_name)
        i.save(fn)
        self.faces[possible_name]["img"] = fn


        # CROP
        facebox = detection["faces"][0]["box"]
        padding = 75
        box = [max(0, (i.width * facebox["left"] / 100.) - padding),
               min(0, (i.height * facebox["top"] / 100.) - padding),
               min(i.width, (i.width * facebox["right"] / 100.) + padding),
               max(0, (i.height * facebox["bottom"] / 100.) + padding)]

        print("Crop box", facebox, box, (i.width, i.height))
        i2 = i.crop(box)
        fn = os.path.join(target_dir, "%s_CLOSE.png" % possible_name)
        i2.save(fn)
        self.faces[possible_name]["closeup"] = fn

        padding = 25
        b = [textbox[0] - padding, textbox[1] - padding,
             textbox[2] + padding, textbox[3] + padding]
        print("Crop-box", b)
        i2 = i.crop(b)
        fn = os.path.join(target_dir, "%s_text.png" % possible_name)
        i2.save(fn)
        self.faces[possible_name]["text"] = fn

        print("Know of these:", list(self.faces))
        return possible_name

    def get_cast(self):

        def autocolor():
            return "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) + "A6"

        import copy
        faces = copy.copy(self.faces)

        faces = self.dedupe_people(faces)
        faces = self.dedupe_people(faces, "text")

        cast = {}
        for key in faces:
            cast[key] = {
                "name": key,
                "closeup": "faces/" + os.path.basename(faces[key]["closeup"]),
                "text": "faces/" + os.path.basename(faces[key]["text"]),
                "ts": faces[key]["ts"],
                "color": autocolor()
                }

        return cast

    def dedupe_people(self, people, tag="closeup"):
        """
        Try to find duplicates (very similar face images), and remove all but one
        """
        hashes = []
        to_remove = []
        for name in people:

            hash = imagehash.average_hash(Image.open(people[name][tag]))
            hashes.append((hash, name))

        # Find similar
        to_remove = []
        for hash, name in hashes:
            for hash2, name2 in hashes:

                if name == name2:
                    continue

                if name in to_remove:
                    continue

                if StringMatcher.distance(str(hash), str(hash2)) <= 5:
                    to_remove.append(name2)
                    print("Similar:", name, name2, StringMatcher.distance(str(hash), str(hash2)))

        for name in to_remove:
            if name in people:
                del people[name]
        return people
    
    def detect_scene(self, img):
        scene = self._detector.process_frame(self._global_frame_num, img)
        self._global_frame_num += 1
        return scene

    def analyze_video(self, video, options):
        if options.iframes:
            iframes = self.get_iframes(video)
        else:
            iframes = None
        
        start_global_frame = self._global_frame_num

        # We need the directory of the video
        target_dir = os.path.split(video)[0]


        if 0 and os.path.exists("/tmp/analysis.json"):
            print("Using cached results, iframes:", len(iframes))
            ret = json.load(open("/tmp/analysis.json", "r"))
            if options.startts:
                for idx, item in enumerate(ret):
                    if item["start"] >= options.startts:
                        break
                ret = ret[idx:]

            if options.endts:
                for idx, item in enumerate(ret):
                    if item["start"] >= options.endts:
                        break
                ret = ret[:idx]

            faces = self.track_faces(ret)

            return self.select_by_face(faces, iframes=iframes)

        cap = cv2.VideoCapture(video)
        ret = []
        i = 0
        frame_nr = -1
        last_text = None
        last_person_found = -10
        snap_at = {"ts": 0}
        people = {}
        text_debug = open("/tmp/text_extracts.txt", "w")
        if not cap.isOpened():
            print("Could not open video capture")

        while cap.isOpened():
            success, image = cap.read()
            frame_nr += 1

            if not success:
                if os.path.exists(video):
                    break
                # if video is string and starts with http

                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            try:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                print("No more images - guessing we're done?")
                break

            fps = cap.get(cv2.CAP_PROP_FPS)

            new_scene = self.detect_scene(img)
            if new_scene:
                iframes = (iframes or []) + [(f - start_global_frame) / fps for f in new_scene]
                print("New scene at", cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.)
                print(new_scene, frame_nr, iframes)

            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.

            # If we're using iframes, only process iframes (for now)
            if options.startts > 0 and options.startts > ts:
                continue

            if options.endts is not None and options.endts < ts:
                break

            analysis = self.analyze(image, ts, options)
            if not analysis:
                continue
            frameinfo = {"start": ts, "analysis": analysis}

            if options.cast and ts > last_person_found + 5 and \
              len(analysis["faces"]) == 1 and frame_nr % 5 == 0:

                # img = Image.fromarray(image, 'RGB')
                # Crop to something sensible?
                # Look for lower thirds
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                gray, img_bin = cv2.threshold(gray,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                gray = cv2.bitwise_not(img_bin)
                kernel = np.ones((2, 1), np.uint8)
                img2 = cv2.erode(gray, kernel, iterations=1)
                img2 = cv2.dilate(img2, kernel, iterations=1)

                img2 = Image.fromarray(img2)
                box = [160, 530, 850, 650]
                # box = [int(img.width * 0.1), int((img.height * 0.75)), int(img.width * 0.6), img.height * 0.95]
                img2 = img2.crop(box)
                bio = BytesIO()
                img2.save(bio, "PNG")

                import pytesseract
                t = pytesseract.image_to_string(img2, lang="nor").strip()

                t_clean = clean_up(t)

                if not t_clean:
                    continue

                if snap_at["ts"] == 0:  # Not waiting for a snap, check for text
                    # If we cleaned up this text, we assume we'll get a very
                    # similar one for a few seconds, wait for that image
                    snap_at["ts"] = ts + 1.0
                elif snap_at["ts"] > ts:
                    print(ts, "Will snap at", snap_at["ts"], "in", snap_at["ts"] - ts)
                    continue
                elif snap_at["ts"] < ts - 1:
                    print(ts, "Woops, failed to snap", snap_at["ts"])
                    snap_at["ts"] = 0
                    continue
                else:
                    # Should snap now!
                    # t = extract_text(bio.getvalue(), "nor")
                    if t and t_clean:
                        added = self.check_people(img, t_clean, analysis, box,
                                                  snap_at["ts"], target_dir=target_dir)
                        if added:
                            last_person_found = ts
                            print(snap_at["ts"], "Found", t_clean)

                        if last_text and (last_text.find(t) > -1 or t.find(last_text) > -1):
                            print("%05d Found text" % frame_nr, t[:t.find("\n")])
                            print(analysis)
                        else:
                            if len(t) > 5:
                                last_text = t
                            else:
                                last_text = None

                        text_debug.write("%05d: %s\n" % (frame_nr, t.strip().replace("\n", "|")))
                        text_debug.flush()
                        Image.fromarray(img).save("/tmp/test/fesk_%d.png" % frame_nr)
                        img2.save("/tmp/test/txt_%d.png" % frame_nr)
                        snap_at["ts"] = 0

            ret.append(frameinfo)

        # open("/tmp/analysis.json", "w").write(json.dumps(ret, indent=" "))

        if options.startts:
            for idx, item in enumerate(ret):
                if item["start"] >= options.startts:
                    break
            ret = ret[idx:]

        if options.endts:
            for idx, item in enumerate(ret):
                if item["start"] >= options.endts:
                    break
            ret = ret[:idx]

        faces = self.track_faces(ret)

        return self.select_by_face(faces, iframes=iframes)


    def sort_faces_by_heading_and_size(self, faces, startts, endts):
        if len(faces) == 0:
            return []

        # We first sort by heading
        headings = {"straight": 0, "frontal": 1, "askew": 2, "sideview": 3, "unknown": 4}

        # For each face, we need to calculate the amount of each heading (can be lots of detections)

        heading_index = []

        ret = copy.copy(faces)

        for face in ret:
            face.headings = {h: 0 for h in headings.keys()}
            for p in face.get_positions(startts, endts):
                face.headings[p["heading"]] += 1
            # print("HEADNGS", face.headings)
        # Now we can sort the list of faces by headings...
        def by_heading(a, b):
            for key in headings:
                if a.headings[key] != b.headings[key]:
                    return a.headings[key] - b.headings[key]

            return 1

        ret.sort(key=functools.cmp_to_key(by_heading))

        return ret


    def sort_by_heading_and_size(self, bboxes):

        if len(bboxes) == 0:
            return []

        # We first sort by heading
        headings = {"straight": 0, "frontal": 1, "askew": 2, "sideview": 3, "unknown": 4}

        l = [(headings[box["heading"]], box["heading"], box["orientation"], box["size"], idx) for idx, box in enumerate(bboxes)]

        l.sort(key=itemgetter(0))

        print("1", l)

        # We now have a list sorted by headings. If more than one has the same value, we
        # sort by size
        hl = {h: [] for h in headings}
        for item in l:
            hl[item[1]].append(item)

        print("h1", hl)

        candidates = []
        for h in headings:
            if len(hl[h]) > 0:
                candidates.extend(hl[h])

        print("candiates", candidates)
        if not candidates:
            raise Exception("INTERNAL: Had bboxes but no candidates?")

        # Biggest first
        candidates.sort(key=itemgetter(3), reverse=True)

        return [bboxes[c[4]] for c in candidates]

    def sort_by_movement(self, bboxes):

        if len(bboxes) == 0:
            return []

        bboxes.sort(key=itemgetter("movement"), reverse=True)

        # We failed, fallback
        if len(bboxes) > 1 and bboxes[0]["movement"] == bboxes[1]["movement"]:
            return self.sort_by_heading_and_size(bboxes)

        return bboxes


    def select_best_center(self, bboxes, ts, sortfunc=None):

        switch_timeout = 3
        max_focus_time = 10

        # Don't switch too often
        if self.last_face and ts - self.last_face["ts"] < switch_timeout:
          return None

        if len(bboxes) == 0:
          return None

        if len(bboxes) == 1:
          return bboxes[0]

        # Sort by size (should perhaps rather be score?)
        if sortfunc:
            bboxes = sortfunc(bboxes)
        else:
            bboxes.sort(key=itemgetter("size"))

        center = bboxes[0]  # Start out with the biggest face

        if self.last_face:
          limit = 10; # Total of 5% off or less
          # Trivial one - if there is a face very close to the last one, keep using it.
          for box in bboxes:
            if abs(box["posX"] - self.last_face["posX"]) + abs(box["posY"] - self.last_face["posY"]) < limit:
              center = box
              print(ts, "Keeping face")
              break

          # If we've focused on this face for a while, choose another (if available)
          if ts - self.last_face["ts"] > max_focus_time:
            print(ts, "Switching face due to time")
            for box in bboxes:
              if box != center:
                center = box
                self.last_face = None
                break

        if not self.last_face:
          self.last_face = {"ts": ts, "posX": center["posX"], "posY": center["posY"]}


        return center


    def track_faces(self, analysis):
        """
        Try to track people and their movement
        """

        faces = []
        active_faces = []

        grace_period = 0.8  # Now long before a face is not active any more, i.e. not present for this long (will be back-dated)

        for frame in analysis:

            # Mark timed out faces
            for face in active_faces:
                if face.hasExpired(frame["start"], grace_period):
                    active_faces.remove(face)
                    faces.append(face)
                    face.endts = face.last_seen_ts

            for detected in frame["analysis"]["faces"]:
                detected["movement"] = 0
                found = False
                for face in active_faces:
                    if face.is_same_face(detected):
                        found = True
                        face.add_detection(detected, frame["start"])
                        detected["movement"] = face.movement
                        break
                if found:
                    continue

                # Not a know face, make a new one
                new_face = Face()
                new_face.startts = frame["start"]
                new_face.add_detection(detected, frame["start"])
                active_faces.append(new_face)

            if 0:
                print("FACES")
                for face in faces:
                    print("  ", face)

        # Archive all faces
        for face in active_faces:
            face.endts = face.last_seen_ts
        faces.extend(active_faces)

        # Remove any faces that were visible too short
        for face in faces:
            if face.endts - face.startts < 0.5:
                faces.remove(face)

        f = [str(face) for face in faces]
        open("/tmp/faces.json", "w").write(json.dumps(f, indent=" "))

        return faces

    def select_by_face(self, faces, min_show_time=1.5, max_show_time=5, iframes=None):
        """
        Select position based on face tracking.
        If iframes are given, use them to pick a new face on scene change
        """
        new_selection = []
        last_position = (50, 50)
        last_ts = 0
        current_ts = 0

        faces.sort(key=attrgetter("startts"))

        # Find end time
        end_time = 0
        for face in faces:
            end_time = max(end_time, face.endts)

        scene_shift = False

        while True:
            if current_ts >= end_time:
                break

            # print()
            candidates = []
            all_possible = []
            # We start out by selecting a face for the current time (if we have any)
            for face in faces:
                if face.endts < current_ts:
                    continue

                # print("T: %0.2f" % current_ts, "-", current_ts + min_show_time, face)
                if face.startts <= current_ts and face.endts >= current_ts + min_show_time:
                    # print("Found a candidate face" ,face)
                    candidates.append(face)
                elif face.startts <= current_ts and face.endts >= current_ts + 0.5:
                    all_possible.append(face)

                if face.startts > current_ts + min_show_time:
                    break  # No need to look further


            if len(candidates) == 0:
                candidates = all_possible
            # print("%0.2f: candidates" % current_ts, candidates)

            if len(candidates) == 0:
                # Find next time
                next_ts = 100000
                for face in faces:
                    if face.startts > current_ts:
                        next_ts = min(face.startts, next_ts)

                print("No candidates, skipping to", next_ts)

                current_ts = next_ts
                continue

            next_ts =  current_ts + max(min_show_time, min(max_show_time, face.endts - face.startts))
            print("looking at", current_ts, next_ts, iframes)

            # MOVE THIS TO A COMPRESS FUNCTION
            if iframes:
                # Align with iframe if one is very close                
                for ts in iframes:
                    if ts < current_ts:
                        continue
                    if ts == current_ts:
                        print("setting sceneshift")
                        scene_shift = True
                        continue
                    if ts >= next_ts:
                        break
                    next_ts = ts
                if scene_shift:
                    print("Scene shift at", ts)

            # Pick one (sort by orientation or something?)
            selected = self.sort_faces_by_heading_and_size(candidates, current_ts, next_ts)

            # face = random.sample(candidates, 1)[0]
            face = candidates[0]
            positions = face.get_positions(current_ts, next_ts)
            if len(positions) > 0:
                if iframes and scene_shift:
                    print("adding sceneshift to selection")
                    print(scene_shift)
                    positions[0]["animate"] = False
                    scene_shift = False

                new_selection.extend(positions)
            else:
                if iframes:
                    for ts in iframes:
                        if ts <= current_ts:
                            continue
                        if ts >= next_ts:
                            break
                        print("adding dummy scene shift at", ts)
                        new_selection.append({"start": ts, "end": ts + 1/50, "pos": [50,50], "animate": True, "dummy": True})

            # new_selection.append({"start": current_ts, "end": next_ts, "pos": face.get_position()})

            current_ts = min(face.endts, next_ts)

        return new_selection



    def choose_center(self, analysis):
        """
        Analyse the detections and try to choose sensibly 
        """

        if len(analysis) == 0:
            raise Exception("Cant choose center with no frame info")

        new_selection = []

        last_position = (0, 0)
        last_face = None
        last_face_switch_ts = 0
        tentative = {"face": None, "frames": []}

        for frame in analysis:

            done = False
            print()
            print("Analysing frame", frame)
            selected_face = None
            if 1:
                selected = self.sort_by_heading_and_size(frame["analysis"]["faces"]) 
            else:
                selected = self.sort_by_movement(frame["analysis"]["faces"]) 

            if len(selected) == 0:
                continue  # No info for this frame

            # We now have the available faces sorted after what we think is importance
            # See if the currently active position has a face first
            if 0:
                for face in selected:
                    if last_face and Face.is_same(last_face, face):
                        print("Last face is still present, use it for now", frame["start"] - last_face_switch_ts)

                        tentative = {"face": None, "frames": []}

                        if frame["start"] - last_face_switch_ts > 5.0:
                            print("   - stuck too long, try another")
                            break

                        # TODO: Do something sensible here, like switch angle after a while or 
                        # monitor the movement of a face for example
                        # last_face.active_at(frame["start"])
                        # We can add this frame, it's good

                        if len(new_selection) > 0:
                            new_selection[-1]["end"] = frame["start"]

                        frame["pos"] = [face["posX"], face["posY"]]

                        last_face = face
                        new_selection.append(frame)
                        done = True
                        break
                if done:
                    continue

            # The last face didn't work out, do we have a tentative face to switch to?
            if tentative["face"]:
                for face in selected:
                    if Face.is_same(face, tentative["face"]):

                        # This is the same, add the frame
                        if len(tentative["frames"]) > 0:
                            tentative["frames"][-1]["end"] = frame["start"]

                        frame["pos"] = [face["posX"], face["posY"]]

                        tentative["frames"].append(frame)
                        print("Added to tentative", len(tentative["frames"]), [face["posX"], face["posY"]])

                        if len(tentative["frames"]) > 30:
                            # we've been stable at this face for n frames, go for it (retrospectively)
                            last_face_switch_ts = tentative["frames"][0]["start"]

                            if len(new_selection) > 0:
                                new_selection[-1]["end"] = last_face_switch_ts

                            print(" **|* ", tentative["frames"][0])
                            new_selection.extend(tentative["frames"])
                            last_face = tentative["face"]
                            tentative = {"face": None, "frames": []}
                            print("SWITCHED FACE", [face["posX"], face["posY"]], face)

                        done = True

            if done:
                continue

            # We didn't find a tentative one either, forget it and use this
            # one as tentative
            print(" * NEW TENTATIVE FACE", selected[0], [selected[0]["posX"], selected[0]["posY"]])
            tentative = {"face": selected[0], "frames": [frame]}
            continue

        return analysis


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('images', metavar='N', type=str, nargs='*',
                        help='List of images to analyze')

    parser.add_argument("-v", "--video", dest="video", help="Input video file (or stream?)")

    parser.add_argument("-o", "--output", dest="output", help="Output file (aux json)", required=True)

    parser.add_argument("-c", "--cast", dest="cast", help="Output CAST file (json)", required=False)

    parser.add_argument("--selfie", dest="selfie", help="Mirror video (for selfies)",
                        action="store_true", default=False)

    parser.add_argument("--show", dest="show", help="Show work", action="store_true", default=False)

    parser.add_argument("--iframes", dest="iframes", help="Align to iframes", action="store_true", default=False)


    parser.add_argument("--startts", dest="startts", help="Start at", default=0.0)
    parser.add_argument("--endts", dest="endts", help="Stop at", default=None)

    parser.add_argument("--tile", dest="tile", help="Tile the video for analysis (for 'busy' content)",
                        action="store_true", default=False)


    options = parser.parse_args()

    analyzer = Analyzer(options)
    options.startts = float(options.startts)
    if (options.endts):
        options.endts = float(options.endts)

    if options.video:
      res = analyzer.analyze_video(options.video, options)
    else:
      res = analyzer.analyze_images(options.images, options)


    # cv2.waitKey()

    print("Dumping result")

    open(options.output, "w").write(json.dumps(res, indent=" "))

    if options.cast:

        cast = analyzer.get_cast()
        open(options.cast, "w").write(json.dumps(cast, indent=" "))
