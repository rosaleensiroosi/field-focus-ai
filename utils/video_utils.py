import cv2

def read_video(video_path):
    video_frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(frame)
    cap.release()
    return video_frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # output vid format = mp4
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0])) 
    for frame in output_video_frames: # writes frame to video writer
        out.write(frame)
    out.release()