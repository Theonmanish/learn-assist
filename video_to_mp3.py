# converts the videos to mp3
import os
import subprocess
files = os.listdir("videos")
os.makedirs("audios", exist_ok=True)
for file in files:
    if " #" not in file or " [" not in file or " -" not in file:
        print(f"Skipping file with unexpected name format: {file}")
        continue
    tutorial_number = file.split(" [")[0].split(" #")[1]
    file_name = file.split(" -")[1].split(" -")[0]
    print(tutorial_number,file_name)
    subprocess.run(
        ["ffmpeg", "-i", os.path.join("videos", file),
         os.path.join("audios", f"{tutorial_number}_{file_name}.mp3")],
        check=True,
    )