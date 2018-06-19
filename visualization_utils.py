import numpy as np
import collections

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from object_detection.utils.visualization_utils import STANDARD_COLORS
from object_detection.utils.visualization_utils import draw_bounding_box_on_image_array

def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    tracker_handler,
    counter = None,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color='black',
    skip_scores=False,
    skip_labels=False):
    """Overlay labeled boxes on an image with formatted scores and label names.

    This function groups boxes that correspond to the same location
    and creates a display string for each detection and overlays these
    on the image. Note that this function modifies the image in place, and returns
    that same image.

    Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a numpy array of shape [N, image_height, image_width] with
      values ranging between 0 and 1, can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
    keypoints: a numpy array of shape [N, num_keypoints, 2], can
      be None
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
    """
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_instance_boundaries_map = {}
    box_to_keypoints_map = collections.defaultdict(list)
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if instance_boundaries is not None:
                box_to_instance_boundaries_map[box] = instance_boundaries[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                display_str = ''
            if not skip_labels:
                if not agnostic_mode:
                    if classes[i] in category_index.keys():
                        class_name = category_index[classes[i]]['name']
                    else:
                        class_name = 'N/A'
                    display_str = str(class_name)
            if not skip_scores:
                if not display_str:
                    display_str = '{}%'.format(int(100*scores[i]))
                else:
                    display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
            box_to_display_str_map[box].append(display_str)
            if agnostic_mode:
                box_to_color_map[box] = 'DarkOrange'
            else:
                box_to_color_map[box] = STANDARD_COLORS[
                    classes[i] % len(STANDARD_COLORS)]

    boxes_list = [box for box in box_to_display_str_map]
    box_to_id_map, tracker_box_to_id_map = tracker_handler.track(boxes_list, image)

    for box in box_to_display_str_map:
        box_to_display_str_map[box][0] += ': ' + box_to_id_map[box]

    if False: # Hide trackers if boxes exist
        id_list = [box_to_id_map[box] for box in box_to_id_map]
        delete_tracker_boxes = [box for box in tracker_box_to_id_map if tracker_box_to_id_map[box] in id_list]
        for box in delete_tracker_boxes:
            del tracker_box_to_id_map[box]
        draw_tracker_boxes(image, tracker_box_to_id_map, line_thickness, use_normalized_coordinates, color='Chartreuse')
    else:
        draw_tracker_boxes(image, tracker_box_to_id_map, line_thickness, use_normalized_coordinates, color='Pink')

    draw_boxes(image, box_to_color_map, box_to_display_str_map, line_thickness, use_normalized_coordinates)

    if counter != None:
        id_list = [box_to_id_map[box] for box in box_to_id_map if box[2] < 0.95 ]
        id_list += [tracker_box_to_id_map[box] for box in tracker_box_to_id_map if box[2] < 0.95]
        #id_list += [tracker_box_to_id_map[box] for box in tracker_box_to_id_map]
        text = "Count: " + str(counter.update(id_list))
        draw_text(image, text=text, xy=(50,30), font_size=40)

    return image


def draw_boxes (image, box_to_color_map, box_to_display_str_map, line_thickness, use_normalized_coordinates):
    # Draw all boxes onto image.
    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box

        draw_bounding_box_on_image_array(
            image,
            ymin,
            xmin,
            ymax,
            xmax,
            color=color,
            thickness=line_thickness,
            display_str_list=box_to_display_str_map[box],
            use_normalized_coordinates=use_normalized_coordinates)
    return image

def draw_tracker_boxes (image, box_to_id_map, line_thickness, use_normalized_coordinates, color):
    # Draw all boxes onto image.
    for box, id in box_to_id_map.items():
        ymin, xmin, ymax, xmax = box

        draw_bounding_box_on_image_array(
            image,
            ymin,
            xmin,
            ymax,
            xmax,
            color= color,#'Pink''Chartreuse',
            thickness=line_thickness,
            display_str_list= ['car: '+id],
            use_normalized_coordinates=use_normalized_coordinates)
    return image

def draw_text (image, text, xy=(0,0), color=(0,0,255), font_size=16):
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw = ImageDraw.Draw(image_pil)

    font = ImageFont.truetype("abel/abel-regular.ttf", font_size)
    draw.text(xy, text, color, font=font)

    np.copyto(image, np.array(image_pil))

    return image