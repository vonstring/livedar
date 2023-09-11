#!/bin/env python3
import cv2
import mediapipe as mp
import json
from argparse import ArgumentParser
import os
from operator import itemgetter


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

class FaceAnalyzer():

  def __init__(self):

    self.last_face = None

  def get_bbox(self, detection):
    """
    Return the bounding box for the given detection
    """

    print(dir(mp_face_detection.FaceKeyPoint))
    if 1:
      print("Eyes", 
            mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EYE),
            mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE),
            mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP),
            mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION),
            mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION)            
      )

    bbox = detection.location_data.relative_bounding_box  
    pos = mp_face_detection.get_key_point(detection,
                                          mp_face_detection.FaceKeyPoint.NOSE_TIP)
    return {
      "name": "face",
      "box": {
        "left": round((bbox.xmin) * 100),
        "bottom": round((bbox.ymin) * 100),
        "right": round((bbox.xmin + bbox.width) * 100),
        "top": round((bbox.ymin + bbox.height) * 100),
      },
      "size": (bbox.width * bbox.height) * 100,
      "posX": round(pos.x * 100),
      "posY": round(pos.y * 100),
      "value": detection.score[0]
    }


  # For static images:
  def analyze_images(self, options):
    ret = []
    IMAGE_FILES = options.images
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
      for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw face detections of each face.
        if not results.detections:
          ret.append({"path": file, "detections": []})
          continue
        annotated_image = image.copy()
        d = []
        for detection in results.detections:
          print(dir(detection.score))
          print(detection.score)
          box = self.get_bbox(detection)
          d.append(box)
          # print(mp_face_detection.get_key_point(
          #    detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
          # mp_drawing.draw_detection(annotated_image, detection)

        ret.append({"path": file, "detections": d})
        # cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

    return ret

  def analyze_video(self, options):
    # For webcam input: 
    #cap = cv2.VideoCapture(0)

    print("Analyzing video from", options.video)
    cap = cv2.VideoCapture(options.video)
    ret = []
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.4) as face_detection:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          if os.path.exists(options.video):
            break

          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        if options.selfie:
          image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        else:
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:

          ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.

          bboxes = []
          for detection in results.detections:
            bboxes.append(self.get_bbox(detection))
            # print("DETECTION", detection)
            if options.show:
              mp_drawing.draw_detection(image, detection)

          if len(bboxes) > 0:

            # Try to select a good center point
            center = self.select_center(bboxes, ts)
            if center:
              if len(ret) > 0:
                ret[-1]["end"] = ts
              ret.append({"start": ts,
                          "pos": [center["posX"], center["posY"]],
                          "alt": bboxes})

          if options.show:
            cv2.imshow('MediaPipe Face Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
          break
    cap.release()
    return ret


  def select_center(self, bboxes, ts):

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



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('images', metavar='N', type=str, nargs='*',
                        help='List of images to analyze')

    parser.add_argument("-v", "--video", dest="video", help="Input video file (or stream?)",
                        required=False)

    parser.add_argument("-o", "--output", dest="output", help="Output file", required=True)

    parser.add_argument("--selfie", dest="selfie", help="Mirror video (for selfies)", action="store_true", default=False,
                        required=False)

    parser.add_argument("--show", dest="show", help="Show work", action="store_true", default=False,
                        required=False)

    options = parser.parse_args()

    analyzer = FaceAnalyzer()
    if options.video:
      res = analyzer.analyze_video(options)
    else:
      res = analyzer.analyze_images(options)


    print("Dumping result")

    open(options.output, "w").write(json.dumps(res, indent=" "))
