#!/bin/env python3
import cv2
import mediapipe as mp

from argparse import ArgumentParser
import os
from operator import itemgetter


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def get_bbox(detection):
  """
  Return the bounding box for the given detection
  """

  if 0:
    print("Eyes", 
          mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EYE),
          mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)
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
def analyze_images(options):

    IMAGE_FILES = options.images
    BG_COLOR = (192, 192, 192) # gray
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
      for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
          continue
        print(
            f'Nose coordinates: ('
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
        )

        annotated_image = image.copy()
        # Draw segmentation on the image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)
        # Draw pose landmarks on the image.
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
        # Plot pose world landmarks.
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    print(dir(mp_pose.PoseLandmark))

def analyze_video(options):
    # For webcam input:
    cap = cv2.VideoCapture(options.video)
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        if options.selfie:
          image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        else:
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, channels = image.shape
        image2 = image[0:height, 0:round(width * 0.5)]
        print("Resized to", image.shape, image2.shape)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image2)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            print(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB])

        # print(dir(results.pose_landmarks.landmark[0]))
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
          break
    cap.release()




if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('images', metavar='N', type=str, nargs='*',
                        help='List of images to analyze')

    parser.add_argument("-v", "--video", dest="video", help="Input video file (or stream?)",
                        required=False)

    parser.add_argument("-o", "--output", dest="output", help="Output file", required=True)

    parser.add_argument("--selfie", dest="selfie", help="Mirror video (for selfies)", action="store_true", default=False,
                        required=False)

    options = parser.parse_args()

    if options.video:
      res = analyze_video(options)
    else:
      res = analyze_images(options)


    print("Dumping result")

    open(options.output, "w").write(json.dumps(res, indent=" "))

