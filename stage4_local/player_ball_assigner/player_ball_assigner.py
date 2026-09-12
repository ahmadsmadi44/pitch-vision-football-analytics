import sys

sys.path.append('/content')
from utils.bbox_utils import get_center_of_bbox


class PlayerBallAssigner:
    def __init__(self, max_player_ball_distance=70):
        self.max_player_ball_distance = max_player_ball_distance

    def assign_ball_to_player(self, players, ball_bbox):
        ball_x, ball_y = get_center_of_bbox(ball_bbox)
        minimum_distance = float("inf")
        assigned_player = -1

        for player_id, player in players.items():
            bbox = player["bbox"]
            distance_left = ((bbox[0] - ball_x) ** 2 + (bbox[-1] - ball_y) ** 2) ** 0.5
            distance_right = ((bbox[2] - ball_x) ** 2 + (bbox[-1] - ball_y) ** 2) ** 0.5
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player
