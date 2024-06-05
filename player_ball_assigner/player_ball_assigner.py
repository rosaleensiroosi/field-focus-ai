import sys
sys.path.append('../')
from utils import get_center_of_bbox, measure_dist

class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 70  # max distance to assign ball to player
    
    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)  # get ball center position
        min_distance = float('inf')  # init min distance with infinity
        assigned_player = -1  # init assigned player as -1 (no player)
        
        for player_id, player in players.items():
            player_bbox = player['bbox']  # get player bbox
            # calc distances from ball to left and right sides of player
            distance_left = measure_dist((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_dist((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)  # use min distance
            
            # if distance is within max limit and smallest found, assign player
            if distance < self.max_player_ball_distance and distance < min_distance:
                min_distance = distance
                assigned_player = player_id

        return assigned_player  # return id of assigned player
