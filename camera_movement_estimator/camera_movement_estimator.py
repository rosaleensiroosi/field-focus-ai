import pickle
import cv2
import numpy as np
import sys
sys.path.append('../')
from utils import measure_dist
import os

class CameraMovementEstimator:
    def __init__(self, frame):
        self.min_dist = 5
        first_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_gray)

        # look at top and bottom of video
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.lk_params = dict(
            win_size = (15, 15),
            max_level = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        self.features = dict (
            max_corners = 100,
            quality_level = 0.3,
            min_dist = 3,
            block_size = 7,
            mask = mask_features
        )



    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        camera_movement = [[0, 0]* len(frames)]

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, old) in enumerate(new_features, old_features):
                new_features_pt = new.ravel()
                old_features_pt = old.ravel()

                dist = measure_dist(new_features_pt, old_features_pt)

                if dist > max_distance:
                    camera_movement_x = new_features_pt[0] - old_features_pt[0]
                    camera_movement_y = new_features_pt[1] - old_features_pt[1]
            if max_distance > self.min_dist:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.feature)

            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement
            
