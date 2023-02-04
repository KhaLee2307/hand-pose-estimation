from hand_shape_pose.util.detector_utils import load_inference_graph, detect_objects

class Detector:
    detector_params = {}
    detector = None

    def __init__(self):
        pass

    def set_detector_params(self, params):
        self.detector_params = params

    def detect(self):
        pass

class HandDetector(Detector):
    def __init__(self):
        self.detection_graph, self.sess = load_inference_graph()

    def detect(self, rgb_image):
        boxes, confidences = detect_objects(rgb_image, self.detection_graph, self.sess)

        im_height, im_width = rgb_image.shape[:2]

        # top, left, bottom, right
        objects= [(box[0] * im_height, box[1] * im_width, box[2] * im_height, box[3] * im_width) for box, score  in zip(boxes, confidences)]

        if len(objects) == 0:
            return None

        # change to an array of (x, y, w, h)
        (top, left, bottom, right) = objects[0]
        bounding_box = [int(left), int(top), int(right - left), int(bottom - top)]
        return bounding_box

def crop_frame(frame, box, ratio= 0.5):
    (x, y, width, height) = box
    (h, w, d) = frame.shape
    max_len= max(width, height)
    x1, y1 = int(x - max_len*ratio), int(y - max_len*ratio)
    x2, y2 = int(x + max_len + (x - x1)), int(y + max_len + (y - y1))
    coord = (x1, y1, x2, y2)
    
    if (x1 < 0 or y1 < 0  or x2 > w or y2 > h):
        return None, frame
    return coord, frame[y1:y2, x1:x2, :]