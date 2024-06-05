from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colours = {}  # dictionary to store team colours
        self.player_team_dict = {}  # dictionary to store player team assignments
    
    def get_clustering_model(self, image):
        # perform k-means clustering on the 2d reshaped image
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(image.reshape(-1, 3))
        return kmeans  # return the k-means model

    def get_player_colour(self, frame, bbox):
        # extract the player's bounding box from the frame
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        # extract the top half of the player's bounding box
        top_half_image = image[0:int(image.shape[0] / 2), :]
        
        # get clustering model for the top half of the image
        kmeans = self.get_clustering_model(top_half_image)
        # reshape the labels to the image shape
        clustered_image = kmeans.labels_.reshape(top_half_image.shape[:2])
        
        # determine the non-player cluster based on corner clusters
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        # determine the player cluster
        player_cluster = 1 - non_player_cluster
        
        # get the player color from the k-means cluster centers
        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color  # return the player color

    def assign_team_colour(self, frame, player_detections):
        # extract player colors and perform k-means clustering to assign team colors
        player_colors = [self.get_player_colour(frame, player["bbox"]) for player in player_detections.values()]
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10).fit(player_colors)
        
        # store the k-means model and team colors
        self.kmeans = kmeans
        self.team_colours = {1: kmeans.cluster_centers_[0], 2: kmeans.cluster_centers_[1]}

    def get_player_team(self, frame, player_bbox, player_id):
        # if player id already has a team assigned, return it
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        # get player color
        player_color = self.get_player_colour(frame, player_bbox)
        # predict team id using k-means and increment by 1
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1

        # hard-code team assignments for specific goalkeepers
        if player_id == 152 or player_id == 133:
            team_id = 1
        elif player_id == 376 or player_id == 383 or player_id == 336:
            team_id = 2

        # store and return the player team id
        self.player_team_dict[player_id] = team_id
        return team_id