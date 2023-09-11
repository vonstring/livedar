import math
import cv2
import mediapipe as mp


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def dist(a, b):
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))

class FaceAnalyzer:

    def __init__(self, options):

        self.options = options
        self._coarse_detector = \
            mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.2)

        self._fine_detector = \
            mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.35)


    def to_boxes(self, detections):
        bboxes = []
        if detections is None:
            return bboxes

        for detection in detections:
            bboxes.append(self.get_bbox(detection))

        return bboxes

    def show_img(self, image, detections):
        if detections is None:
            return
        # image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for detection in detections:
            mp_drawing.draw_detection(image, detection)
        cv2.imshow('MediaPipe Face Detection', image)
        cv2.waitKey()


    def coarse_detection(self, image):
        results = self._coarse_detector.process(image)

        if self.options.show:
            self.show_img(image, results.detections)

        return self.to_boxes(results.detections)


    def fine_detection(self, image):
        results = self._fine_detector.process(image)

        if self.options.show:
            self.show_img(image, results.detections)        

        return self.to_boxes(results.detections)


    def get_bbox(self, detection):
        """
        Return the bounding box for the given detection
        """

        # print(dir(mp_face_detection.FaceKeyPoint))
        points = [
            ["left_eye", mp_face_detection.FaceKeyPoint.LEFT_EYE],
            ["right_eye", mp_face_detection.FaceKeyPoint.RIGHT_EYE],
            ["nose", mp_face_detection.FaceKeyPoint.NOSE_TIP],
            ["left_ear", mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION],
            ["right_ear", mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION]
        ]

        pos = {}
        for point in points:
            p = mp_face_detection.get_key_point(detection, point[1])
            pos[point[0]] = (p.x, p.y)
        dst_ears = abs(dist(pos["left_ear"], pos["nose"]) - dist(pos["right_ear"], pos["nose"])) / dist(pos["left_ear"], pos["right_ear"])
        dst_eyes = abs(dist(pos["left_eye"], pos["nose"]) - dist(pos["right_eye"], pos["nose"])) / dist(pos["left_eye"], pos["right_eye"])

        heading = "unknown"

        if (pos["left_eye"] < pos["nose"] and pos["right_eye"] < pos["nose"]):
            heading = "sideview"
        elif dst_ears < 0.1:
            heading="straight"
        elif dst_eyes < 0.1:
            heading="frontal"
        elif dst_eyes < 0.5:
            heading="askew"

        bbox = detection.location_data.relative_bounding_box
        pos = mp_face_detection.get_key_point(detection,
                                              mp_face_detection.FaceKeyPoint.NOSE_TIP)

        # Try to find out if this is a side face or a front shot or even direct view.
        # We check the placement of the eyes and ears compared to the nose

        # print("Exp:", (dst_ears + dst_eyes) * detection.score[0])

        return {
          "name": "face",
          "heading": heading,
          "orientation": [dst_eyes, dst_ears],
          "box": {
            "left": round((bbox.xmin) * 100),
            "top": round((bbox.ymin) * 100),
            "right": round((bbox.xmin + bbox.width) * 100),
            "bottom": round((bbox.ymin + bbox.height) * 100),
          },
          "size": (bbox.width * bbox.height) * 100,
          "posX": round(pos.x * 100),
          "posY": round(pos.y * 100),
          "value": detection.score[0]
        }



class PoseAnalyzer:

    def __init__(self, options):

        self.options = options
        self.detector = \
            mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


    def to_boxes(self, detections):
        bboxes = []
        for detection in detections:
            bboxes.append(self.get_bbox(detection))
            if self.options.show:
              mp_drawing.draw_detection(image, detection)
        
        if self.options.show:
            cv2.imshow('MediaPipe Pose Detection', image)

    def detection(self, image):
        results = []

        detections = self.detector.process(image)
        image_height, image_width, _ = image.shape

        detection = detections.pose_landmarks

        if not detection:
            return results

        pose = {}  # {"value": detection.score[0]}

        for landmark in mp_pose.PoseLandmark:
                pose[landmark.name.lower()] = {
                    "x": detection.landmark[landmark].x * image_width,
                    "y": detection.landmark[landmark].y * image_height,
                    "presence": detection.landmark[landmark].presence,
                    "visibility": detection.landmark[landmark].visibility
                    }


        results.append(pose)

        if self.options.show:
            self.show_img(image, detection)

        return results
        # return self.to_structure(results)

    def to_structure(self, detections):

        res = []
        if detections is None:
            return res

        # for detection in detections.pose_landmarks:
        detection = detections.pose_landmarks

        if 1:
            # print("SCORE", detections.score)

            pose = {}  # {"value": detection.score[0]}

            for landmark in mp_pose.PoseLandmark:
                pose[landmark.name.lower()] = detection.landmark[landmark]
            res.append(pose)

        return res

    def show_img(self, image, detections):
        if detections is None:
            return

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # print(dir(results.pose_landmarks.landmark[0]))
        mp_drawing.draw_landmarks(
            image,
            detections,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imshow('MediaPipe Pose', image)
        cv2.waitKey()
