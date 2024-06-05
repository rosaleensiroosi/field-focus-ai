import cv2  # import OpenCV for computer vision tasks
import numpy as np  # import numpy for numerical operations
from utils import read_video, save_video  # import read_video and save_video functions from utils module
from trackers import Tracker  # import Tracker class from trackers module
from team_assigner import TeamAssigner  # import TeamAssigner class from team_assigner module
from player_ball_assigner import PlayerBallAssigner  # import PlayerBallAssigner class from player_ball_assigner module

def main():
    # read video frames from input video file
    video_frames = read_video('input-videos/08fd33_4.mp4')

    # initialize tracker with model path
    tracker = Tracker('models/best.pt')
    # get object tracks from video frames, read from stub if available
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,  # set to False to force processing of video frames
                                       stub_path='stubs/track_stubs.pkl')
    
    # interpolate ball positions in tracks
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    # save cropped image of a player from the first frame
    for track_id, player in tracks['players'][0].items():  # look at frame 0
        bbox = player['bbox']  # get bounding box of player
        frame = video_frames[0]  # get first frame

        # crop bounding box from frame
        cropped_img = frame[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])]

        # save cropped image
        cv2.imwrite(f'output-videos/cropped_img.jpg', cropped_img)
        break  # save only first image

    # assign player teams using TeamAssigner
    team_assigner = TeamAssigner()
    team_assigner.assign_team_colour(video_frames[0], tracks['players'][0])

    # assign team to each player in each frame
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            # get team for player based on the frame and bounding box
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            # update player track with team and team colour
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_colour'] = team_assigner.team_colours[team]

    # assign ball acquisition using PlayerBallAssigner
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        # get bounding box of the ball in the current frame
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        # assign ball to closest player
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            # mark player as having the ball
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            # record team in control of the ball
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # retain the previous team in control if no player is assigned
            team_ball_control.append(team_ball_control[-1])
    
    # convert team ball control list to numpy array
    team_ball_control = np.array(team_ball_control)

    # draw annotations on video frames
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # save output video with annotations
    save_video(output_video_frames, 'output-videos/output-video.mp4')

if __name__ == '__main__':
    main()
