 # football-analysis-AI-tool
this computer vision / machine learning project uses
1. YOLO to detect players, referees, and the ball,
2. k-means for pixel segmentation (and to group players by their teams), and
3. optical flow for motion tracking.

# goals
the aim of this project is to:
1. read .mp4 video,
2. detect the different players, referees, and the singular football present in a video using YOLO,
3. train the model to improve its performance and accuracy using Jupyter (model is saved as best.pt),
4. assign players to their respective teams based on the cluster analysis performed on their t-shirts using k-means,
5. interpolate ball position when ball is not detected in frame,
6. calculate each teams' acquisition percentage in real time by detecting who is in current possession of the ball, and
7. save output video as an .mp4 video file.

# model
as the model (best.pt) is over 100 MB, it has been uploaded via [drive](https://drive.google.com/file/d/12tdbyDr6NzRrEb5-A5fEiqGOWxPY1mpm/view?usp=sharing) as it is too large for github. :(

# before and after 
a comparison between the before and after, respectively:

<img width="400" alt="Screenshot 2024-05-22 at 4 08 01 PM" src="https://github.com/rosaleens/football-analysis-AI-tool/assets/100162862/bff429c2-de29-4aab-be2e-8682e1aed6f5"> <img width="400" alt="Screenshot 2024-05-22 at 4 08 17 PM" src="https://github.com/rosaleens/football-analysis-AI-tool/assets/100162862/68dc9778-17ec-487f-81f3-3e930de44f1a">



# demonstration
[demonstration before and after video](https://youtu.be/TDzZ2b1MxGw)

# credit
project is inspired by: https://www.youtube.com/watch?v=neBZ6huolkg&t=13139s
