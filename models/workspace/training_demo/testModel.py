import numpy as np
from PIL import Image
import cv2
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

PATH_TO_MODEL_DIR = "./exported-models/my_efficient"
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
PATH_TO_CFG = PATH_TO_MODEL_DIR + "/pipeline.config"
PATH_TO_LABELS = "./annotations/label_map.pbtxt"
PATH_TO_CKPT = PATH_TO_MODEL_DIR + "/checkpoint"
PATH_TEST_IMAGE = "./images/myTest/"

detect_fn2 = tf.saved_model.load(PATH_TO_SAVED_MODEL)

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

image_path = "./images/test/0001.png"
testImagesFiles = [f for f in listdir(PATH_TEST_IMAGE) if isfile(join(PATH_TEST_IMAGE, f))]

for i, img in enumerate(testImagesFiles):
    image_path = "./images/myTest/" + img
    image_np = load_image_into_numpy_array(image_path)

    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn2(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)

    #print(detections)

    cv2.imshow('object detection', image_np_with_detections)

    plt.figure()
    plt.imshow(image_np_with_detections)    # matplotlib is configured for command line only so we save the outputs instead
    plt.savefig("detection_output{}.png".format(i+1))

plt.show()

cv2.waitKey()