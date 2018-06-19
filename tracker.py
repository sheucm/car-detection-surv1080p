import cv2
import random

from object_detection.utils.visualization_utils import draw_bounding_box_on_image_array

class TrackerHandler (object):

    class _Car(object):
        def __init__(self, box, id=None):
            self.box = box
            self.id = id

    def __init__ (self, iou_threshold=0.4,
                        iou_threshold_of_tracker=0.6,
                        tracker_life_if_no_use=5,
                        max_num_of_id=15):
        '''
                    :param iou_threshold: a float number between 0.0 and 1.0.
                                The iou_threshold is a threshold of intersection over union.
                    :param iou_threshold_of_tracker: a float number between 0.0 and 1.0.
                                The iou_threshold_of_tracker is a threshold of intersection over union
                                that reassigned a new tracker when less than the threshold..
                    :param tracker_life_if_no_use: an unsigned integer
                                The tracker_life_if_no_use is a life times which tracker stay along when it's no used.
                    :param max_num_of_id: an unsigned integer
                                The max_num_of_id is a max number for random range from 0 to max_num_of_id.
                    '''
        self.pre_cars = list()
        self.cars = list()
        self.pred_boxes = dict()
        self.tm = TrackerManager()
        self.trackerId_to_times_map = dict()

        self.IOU_THRESHOLD = iou_threshold
        self.IOU_THRESHOLD_OF_TRAKCER = iou_threshold_of_tracker
        self.TRACKER_LIFE_IF_NO_USE = tracker_life_if_no_use
        self.MAX_NUM_OF_ID = max_num_of_id

    def track (self, boxes_list, frame):
        '''
                    :param boxes_list:  List of boxes. box is a tuple of four float number between 0.0 and 1.0.
                                                    Four float  numbers seperately mean ymin, xmin, ymax, xmax
                    :param frame:  np_array of images.
                    :return: Return box_to_id_map and tracker_box_to_id_map
                                box_to_id_map is a dictionary of key of box and value of id of string type.
                                tracker_box_to_id_map is a dictionary of key of box and value of id of string type.
                    '''
        self.cars = self.generate_cars(boxes_list)
        # self.cars now have bounding box but no id.
        # Following codes will specified unique id for each car.

        # The id_list takes the id of detected cars form the frame.
        # The boxes of detected cars include from boxes_list
        # or prediction of tracker
        id_list = list()

        # Step 1:
        #   We specified id for the car which iou of car box and
        #   previous car box is greater than IOU_THRESHOLD.
        if len(self.pre_cars) > 0:
            for car in self.cars:
                for pre_car in self.pre_cars:
                    iou = self.iou(car.box, pre_car.box)
                    if iou > self.IOU_THRESHOLD:
                        car.id = pre_car.id
                        id_list.append(car.id)
                        break
        # The ids of true-positive cars that is not in id_list have these reasons:
        #   1. The car has no box in boxes_list, but has previous box (self.pre_car[i].box).
        #       This means Tensorflow Object Detection doesn't detect this object in this frame.
        #       We need to fix it through predicted box from tracker.
        #   2. The car has box in boxes_list, but has no previous box (self.pre_car[i].box).
        #       This means this car is new detected from Tensorflow Object Detection.
        #   3. The car has box in boxes_list, and has previous box (self.pre_car[i].box),
        #       but iou of car.box and pre_car.box is less than IOU_THRESHOLD.
        #       This means the car run too fast,
        #       therefore, the motion distance of frame (t) and frame (t-1) are too long.

        # Step 2:
        #   1. We take predicted boxes from tracker
        #   2. We filter the predicted boxes from eliminating the its id
        #       which is in id_list.
        pred_boxes = None
        if len(self.tm.trackers) > 0:
            pred_boxes = self.tm.update(frame)
            pred_boxes = self._transfer_predicted_boxes(pred_boxes, frame.shape[1], frame.shape[0])
            pred_boxes = self.refine_tm_and_pred_boxes(pred_boxes, frame)
            miss_ids = [id for id in pred_boxes.keys() if id not in id_list]
            # Case 1: iou (pre_car.box, car.box) > IOU_THRESHOLD
            #   We predict its id by using predicted box
            #   if iou (predict_box, car.box) > IOU_THRESHOLD.
            for id in miss_ids:
                ok, box = pred_boxes[id]
                for car in self.cars:
                    if self.iou(car.box, box) > self.IOU_THRESHOLD:
                        car.id = id
                        id_list.append(car.id)
                        break

            # Case 2: use predicted box
            #   1) You can add predicted boxes that is not in boxes_list
            #       to true-positive results. Also, you need to fix this code.
            #       We will use (2).
            #   2) We recommand not to draw on frame in deploy mode.
            #       This can know the Tensorflow Object Detection Accuracy.
            #       In the next frame, we use predicted box to specify id
            #       if the iou of car.box and pre_car.box is less than IOU_THRESHOLD.

            # refresh miss_ids
            miss_ids = [id for id in miss_ids if id not in id_list]
            id_list += miss_ids # To keep the id to prevent from specifying the duplicated from new id.

            # trackerId_to_times_map is used to record the ids with times
            #   which the times mean the number of missed detection.
            self.update_trackerId_to_times_map (miss_ids)

            # If the tracker miss over certain times, we will delete the tracker from tm.
            self.update_tm_from_trackerId_to_times_map()

        # Step 3:
        #   New id for new detected car, and add it to tracker for prediction.
        for car in self.cars:
            if car.id == None:
                while (True):
                    car.id = random.randint (0, self.MAX_NUM_OF_ID)
                    if car.id not in id_list:
                        id_list.append(car.id)
                        box = self._prepare_box_for_track(car.box, frame.shape[1], frame.shape[0])
                        self.tm.add(cv2.TrackerCSRT_create(), frame, box, car.id)
                        break

        self.update_pre_cars()

        box_to_id_map = dict()
        for car in self.cars:
            box_to_id_map[car.box] = str(car.id)

        tracker_box_to_id_map = dict()
        if pred_boxes != None:
            for id in pred_boxes:
                ok, box = pred_boxes[id]
                tracker_box_to_id_map[box] = str(id)

        return box_to_id_map, tracker_box_to_id_map

    def iou (self, bb1, bb2):
        '''
                    bounding box is a tuple of four float numbers between 0.0 and 1.0.
                    The four float numbers stand for ymin, xmin, ymax, xmax
                    :param bb1: a tuple of four float numbers between 0.0 and 1.0.
                    :param bb2:  a tuple of four float numbers between 0.0 and 1.0.
                    :return: a float number between 0.0 and 1.0,
                                    which stand for the result of  intersection over union
                    '''
        region1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        region2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
        intersection_xmin = max (bb1[0], bb2[0])
        intersection_ymin = max (bb1[1], bb2[1])
        intersection_xmax = min (bb1[2], bb2[2])
        intersection_ymax = min (bb1[3], bb2[3])
        if intersection_xmin > intersection_xmax or \
            intersection_ymin > intersection_ymax:
            return 0.0
        intersection_region = (intersection_xmax - intersection_xmin) * \
                              (intersection_ymax - intersection_ymin)
        union = region1 + region2 - intersection_region
        return intersection_region * 1.0 / union

    def generate_cars (self, boxes_list):
        '''
                :param boxes_list: a list of boxes
                :return: a list of Car class
                '''
        cars = list()
        for box in boxes_list:
            cars.append (self._Car(box))
        return cars

    def update_pre_cars (self):
        self.pre_cars = self.cars

    def refine_tm_and_pred_boxes (self, pred_boxes, frame):
        '''
                    :param pred_boxes: a dictionary of key of integer id and value of a tuple of Boolean ok and  tuple box.
                    :param frame: np_array of image
                    :return: a dictionary of key of integer id and value of a tuple of Boolean ok and  tuple box.
                                This dictionary mean refined pred_boxes.
                    '''
        delete_id = list()
        new_pred_boxes = dict()
        for id in pred_boxes:
            ok, box = pred_boxes[id]
            if not ok:
                delete_id.append(id)
                self.tm.detele(id)
            elif 0.0 > box[0] or box[0] > 1.0 or \
                0.0 > box[1] or box[1] > 1.0 or \
                0.0 > box[2] or box[2] > 1.0 or \
                0.0 > box[3] or box[3] > 1.0:
                delete_id.append(id)
                self.tm.detele(id)
            else:
                car = [car for car in self.cars if car.id == id]
                if len(car) == 1:
                    car = car[0]
                    if self.iou(box, car.box) < self.IOU_THRESHOLD_OF_TRAKCER:
                        delete_id.append(id)
                        self.tm.detele(id)
                        box = self._prepare_box_for_track(car.box, frame.shape[1], frame.shape[0])
                        self.tm.add(cv2.TrackerCSRT_create(), frame, box, id)
                        new_pred_boxes[id] = (True, car.box)
                elif len(car) > 1:
                    print ("Error: duplicated id")

        for id in delete_id:
            del pred_boxes[id]
        return {**pred_boxes, **new_pred_boxes}

    def update_trackerId_to_times_map (self, miss_ids):
        '''
                    :param miss_ids: a list of integer
                    '''
        # Delete item that is not in miss_ids
        delete_ids = [id for id in self.trackerId_to_times_map \
                      if id not in miss_ids]
        for id in delete_ids:
            del self.trackerId_to_times_map[id]

        # Add or increase to tracker_id_map_to_no_detected_car
        for id in miss_ids:
            if self.trackerId_to_times_map.get(id) is None:
                self.trackerId_to_times_map[id] = 1
            else:
                self.trackerId_to_times_map[id] += 1

    def update_tm_from_trackerId_to_times_map (self):
        delete_ids = [id for id in self.trackerId_to_times_map \
                      if self.trackerId_to_times_map[id] > self.TRACKER_LIFE_IF_NO_USE]
        for id in delete_ids:
            self.tm.detele(id)

    def _prepare_box_for_track (self, box, width, height):
        '''
                    Transfer input box (normalized) to output box (unnormalized)
                    input box: (ymin, xmin, ymax, xman)
                    output box: (xmin, ymin, width_box, height_box)

                    :param box: a tuple of four float numbers between 0 and 1
                                width: unsigned integer
                                height: unsigned integer
                    :return: return output box  (xmin, ymin, width_box, height_box)
                    '''
        return (box[1] * width,
                box[0] * height,
                (box[3] - box[1]) * width,
                (box[2] - box[0]) * height)

    def __prepare_box_for_tensorflow (self, box, width, height):
        '''
                            Transfer input box (unnormalized) to output box (normalized)
                            intput box: (xmin, ymin, width_box, height_box)
                            output box: (ymin, xmin, ymax, xman)

                            :param box: a tuple of four float numbers between 0 and 1
                                        width: unsigned integer
                                        height: unsigned integer
                            :return: return output box  (ymin, xmin, ymax, xman)
                    '''
        return (box[1] * 1.0 / height,
                box[0] * 1.0 / width,
                (box[1] + box[3]) * 1.0 / height,
                (box[0] + box[2]) * 1.0 / width)

    def _transfer_predicted_boxes (self, pred_boxes, width, height):
        '''
                :param pred_boxes:  a dictionary of key of integer id and value of a tuple of Boolean ok and  tuple box.
                :param width: unsigned integer
                                        It means image width.
                :param height: unsigned integer
                                        It means image height.
                :return: a dictionary of key of integer id and value of a tuple of Boolean ok and  tuple box.
                '''
        for id in pred_boxes:
            ok, box = pred_boxes[id]
            box = self.__prepare_box_for_tensorflow(box, width, height)
            pred_boxes[id] = (ok, box)
        return pred_boxes


class TrackerManager (object):

    def __init__ (self):
        ''''
                        Variables:
                            trackers: a dictionary. Key is an unsigned integer, and value is a tracker.
                '''
        self.trackers = dict()

    def add (self, tracker, frame, box, id):
        ''''
                Args:
                    tracker: tracker of from opencv contrib method cv2.Tracker_create().
                                 Such as cv2.TrackerMedianFlow_create().
                                 Others like BOOSTING', 'MIL','KCF', 'TLD', 'GOTURN'
                    frame: np_array of float32 or flaot64
                    box: tuple of 4 integer. ex: (0,0,1,1)
                    id: unsigned integer
                    box_normalized: Boolean value

                Returns:
                    Return True if success; otherwise, False.
                '''
        if not tracker.init (frame, box):
            print ('Error: Cannot init tracker')
            return False
        if id in self.trackers.keys():
            print ('Error: Duplicated id')
            return False
        self.trackers[id] = tracker
        return True

    def detele (self, id):
        '''
                    Args:
                        id: unsigned integer
                '''
        self.trackers[id].clear()
        del self.trackers[id]

    def update (self, frame):
        '''
                    Args:
                        frame: np_array of float32 or flaot64

                    Returns:
                        Return a dictionary of key of id, and value of tuple of status which is ok or not and a bounding box.
                        Type of status and bounding box are Boolean and tuples of 4 integers
                '''
        result = dict()
        for id in self.trackers:
            tracker = self.trackers[id]
            ok, bbox = tracker.update (frame)
            result[id] = (ok, bbox)
        return result