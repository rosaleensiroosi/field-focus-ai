import supervision as sv  # import supervision library
import pickle  # import pickle for serializing and deserializing data
import os  # import os for interacting with operating system
from ultralytics import YOLO  # import YOLO class from ultralytics library
import sys  # import sys for system-specific parameters and functions
import numpy as np # import numpy for triangle points
sys.path.append('../')  # add parent directory to system path
from utils import get_bbox_width, get_center_of_bbox, get_foot_position  # import utility functions
import cv2  # import OpenCV for computer vision tasks
import pandas as pd
 
# define Tracker class
class Tracker:
    # initialize Tracker class with model path
    def __init__(self, model_path):
        self.model = YOLO(model_path)  # load YOLO model
        self.tracker = sv.ByteTrack()  # initialize ByteTrack tracker

    def interpolate_ball_positions(self, ball_poss):
        ball_poss = [x.get(1, {}).get('bbox',[]) for x in ball_poss]
        df_ball_poss = pd.DataFrame(ball_poss, columns=['x1', 'y1', 'x2', 'y2'])

        # interpolate missing values
        df_ball_poss = df_ball_poss.interpolate()
        df_ball_poss = df_ball_poss.bfill()

        ball_poss = [{1: {"bbox": x}} for x in df_ball_poss.to_numpy().tolist()]

        return ball_poss

    # define detect_frames method
    def detect_frames(self, frames):
        batch_size = 20  # set batch size for processing frames
        detections = []  # initialize empty list for detections
        for i in range(0, len(frames), batch_size):
            # process frames in batches
            detections_batch = self.model.predict(frames[i: i + batch_size], conf=0.1)
            # append detections for current batch
            detections += detections_batch
        return detections  # return all detections

    # define get_object_tracks method
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        # check if should read tracks from stub file
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)  # load tracks from stub file
            return tracks  # return loaded tracks

        detections = self.detect_frames(frames)  # detect objects in frames

        # initialize tracks for players, referees, and ball
        tracks = {
            "players": [{} for _ in range(len(frames))],
            "referees": [{} for _ in range(len(frames))],
            "ball": [{} for _ in range(len(frames))]
        }

        # iterate through frames
        for frame_num in range(len(frames)):
            if frame_num < len(detections):
                detection = detections[frame_num]  # get detection for current frame
                cls_names = detection.names  # get class names
                cls_names_inv = {v: k for k, v in cls_names.items()}  # create inverse mapping of class names

                # convert detections to supervision format
                detection_sv = sv.Detections.from_ultralytics(detection)

                # replace 'goalkeeper' class with 'player' class
                for obj_ind, class_id in enumerate(detection_sv.class_id):
                    if cls_names[class_id] == "goalkeeper":
                        detection_sv.class_id[obj_ind] = cls_names_inv["player"]

                # update tracker with current detections
                detection_w_tracks = self.tracker.update_with_detections(detection_sv)

                # process each detection with track
                for frame_detection in detection_w_tracks:
                    bbox = frame_detection[0].tolist()  # get bounding box
                    cls_id = frame_detection[3]  # get class id
                    track_id = frame_detection[4]  # get track id

                    # add bounding box to appropriate track list
                    if cls_id == cls_names_inv['player']:
                        tracks["players"][frame_num][track_id] = {"bbox": bbox}
                    if cls_id == cls_names_inv['referee']:
                        tracks["referees"][frame_num][track_id] = {"bbox": bbox}

                # add ball detections to track list
                for frame_detection in detection_sv:
                    bbox = frame_detection[0].tolist()  # get bounding box
                    cls_id = frame_detection[3]  # get class id
                    if cls_id == cls_names_inv['ball']:
                        tracks["ball"][frame_num][1] = {"bbox": bbox}

        # save tracks to stub file if path is provided
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks  # return tracks

    # define draw_ellipse method
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])  # get bottom y-coordinate of bounding box

        x_center, _ = get_center_of_bbox(bbox)  # get center x-coordinate of bounding box
        width = get_bbox_width(bbox)  # get width of bounding box

        # draw ellipse on frame
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=225,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        # draw rectangle and text for track id, if provided
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = y2 - rectangle_height // 2
        y2_rect = y2 + rectangle_height // 2

        if track_id is not None:
            # draw rectangle
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            x1_text = x1_rect + 12
            if track_id > 90:
                x1_text -= 10

            # draw track id text
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame  # return annotated frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1]) 
        x, _ = get_center_of_bbox(bbox)
        tri_pts = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20]
        ])
        cv2.drawContours(frame, [tri_pts], 0, color, -1)
        cv2.drawContours(frame, [tri_pts], 0, (0, 0, 0), 2)

        return frame
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # draw semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1250, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_frame = team_ball_control[:frame_num + 1]

        # get # of time each team has ball
        t1_num_frames = team_ball_control_frame[team_ball_control_frame == 1].shape[0]
        t2_num_frames = team_ball_control_frame[team_ball_control_frame == 2].shape[0]
        t1 = t1_num_frames / (t1_num_frames + t2_num_frames)
        t2 = t2_num_frames / (t1_num_frames + t2_num_frames)

        cv2.putText(frame, f"white team ball control: {t1 * 100: .2f}%", (1300, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"green team ball control: {t2 * 100: .2f}%", (1300, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    # define draw_annotations method
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []  # initialize empty list for output frames
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()  # make copy of current frame

            # get tracks for current frame
            player_dict = tracks["players"][frame_num] if frame_num < len(tracks["players"]) else {}
            referee_dict = tracks["referees"][frame_num] if frame_num < len(tracks["referees"]) else {}
            ball_dict = tracks["ball"][frame_num]

            # draw ellipses for players
            for track_id, player in player_dict.items():
                colour = player.get("team_colour", (0, 0, 0))
                frame = self.draw_ellipse(frame, player["bbox"], colour, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            # draw ellipses for referees
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (207,215,253), track_id)

            # draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (252,248,249))

            # draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)  # add annotated frame to output list

        return output_video_frames  # return annotated video frames
