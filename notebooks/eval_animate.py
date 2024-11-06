import cv2
import imageio
import os
import re
from pathlib import Path
from moviepy.editor import ImageSequenceClip

def extract_video_number(filename):
    match = re.search(r'video_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return float('inf')

def create_videos_from_frames(input_directory, output_gif, output_mp4, frame_duration=0.5):
    video_files = [f for f in os.listdir(input_directory) if f.endswith('.mp4')]
    video_files.sort(key=extract_video_number)

    if not video_files:
        print("No MP4 files found in the directory")
        return

    frames = []

    for video_file in video_files:
        video_path = os.path.join(input_directory, video_file)
        print(f"Processing: {video_file}")

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()

    if not frames:
        print("No frames were extracted from the videos")
        return

    print("Creating GIF...")
    imageio.mimsave(
        output_gif,
        frames,
        duration=frame_duration,
        loop=0
    )
    print(f"GIF created successfully: {output_gif}")

    print("Creating MP4...")
    clip = ImageSequenceClip(frames, fps=1/frame_duration)
    clip.write_videofile(output_mp4, fps=1/frame_duration)
    print(f"MP4 created successfully: {output_mp4}")

if __name__ == "__main__":
    input_dir = "/content/.../eval"  # Replace with your directory path
    output_gif = "eval.gif"
    output_mp4 = "eval.mp4"
    create_videos_from_frames(input_dir, output_gif, output_mp4)
