import time
import numpy as np
import os
import tensorflow as tf
import cv2

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from visualization_utils import visualize_boxes_and_labels_on_image_array
from tracker import TrackerHandler
from counter import Counter

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'graph/mobilenet_v1/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('', 'object-detection.pbtxt')
NUM_CLASSES = 1


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Get tensors from graph
with detection_graph.as_default():
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
      'num_detections', 'detection_boxes', 'detection_scores',
      'detection_classes'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

def squeeze_output (output_dict):
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict

# Set up video reader and writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('car_detection.avi',fourcc, 20.0, (1280,720))
cap = cv2.VideoCapture('test_video/video1080p.mp4')
tracker_handler = TrackerHandler()
counter = Counter()
cnt = 1
pre_time = time.time()

with detection_graph.as_default():
    with tf.Session() as sess:
        while (cap.isOpened()):
          ret, image_np = cap.read()
          if ret == False:
              break

          # Run inference
          output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image_np, 0)})
          output_dict = squeeze_output(output_dict)

          # Visualization of the results of a detection.
          visualize_boxes_and_labels_on_image_array(
              image_np,
              output_dict['detection_boxes'],
              output_dict['detection_classes'],
              output_dict['detection_scores'],
              category_index,
              tracker_handler = tracker_handler,
              counter = counter,
              instance_masks=output_dict.get('detection_masks'),
              use_normalized_coordinates=True,
              line_thickness=4,
              skip_scores = True,
              hide_tracker_boxes=False)

          out.write(image_np)

          cv2.imshow('Car Detection', image_np)
          if cv2.waitKey(25) & 0xFF == ord('q'):
              break
          #if cnt > 155:
          #  time.sleep(1.5)
          print ("frame", cnt, "second per frame:" , (time.time()-pre_time))
          cnt += 1
          pre_time = time.time()
out.release()
cap.release()
cv2.destroyAllWindows()