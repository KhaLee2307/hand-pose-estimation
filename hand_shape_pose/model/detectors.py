import os
import cv2
import numpy as np
from hand_shape_pose.util import detector_utils as detector_utils
import tensorflow as tf
from shapely.geometry import Polygon

############################################################################
class Detector:
    detector_params = {}
    detector = None

    def __init__(self):
        pass

    def set_detector_params(self, params):
        self.detector_params = params

    def detect(self):
        pass


############################################################################

class HandDetector(Detector):
    def __init__(self):
        self.detection_graph, self.sess = detector_utils.load_inference_graph()

    def detect(self, rgb_image):
        # returns (top [0], left [1], bottom [2], right [3])
        boxes, confidences = detector_utils.detect_objects(rgb_image, self.detection_graph, self.sess)

        im_height, im_width = rgb_image.shape[:2]

        detection_th = self.detector_params.get('detection_th', 0.5)

        #top, left, bottom, right
        objects= [(box[0] * im_height, box[1] * im_width, box[2] * im_height, box[3] * im_width) for box, score  in zip(boxes, confidences) if score >= detection_th]
        #objects = [(box[0] * im_height, box[3] * im_width, box[2] * im_height, box[1] * im_width) for box, score in zip(boxes, confidences) if score >= detection_th]

        bounding_box = detector_utils.non_max_suppression_fast(np.array(objects), overlapThresh=0.4)
        # change to an array of (x, y, w, h)
        bounding_box = [(int(left), int(top), int(right - left), int(bottom - top)) for (top, left, bottom, right) in bounding_box]
        return bounding_box

############################################################################
def add_objects_to_image(img_, objects, color=(255, 0, 0)):
    img = np.copy(img_)
    for (x, y, w, h) in objects:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    return img

def obj_to_poly(obj):
    x, y, w, h = obj
    return Polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h)])


def crop_frame(frame, box, ratio= 0.5):
    (x, y, width, height) = box
    (h, w, d) = frame.shape
    x1, y1 = int(x - width*ratio), int(y - height*ratio)
    x2, y2 = int(x + width + (x - x1)), int(y + height + (y - y1))
    coord = (x1, y1, x2, y2)
    
    if (x1 < 0 or y1 < 0  or x2 > w or y2 > h):
        return None, frame
    return coord, frame[y1:y2, x1:x2, :]
