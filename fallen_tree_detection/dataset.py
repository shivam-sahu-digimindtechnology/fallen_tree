from huggingface_hub import hf_hub_download
import shutil
import os
#  https://huggingface.co/Musawer14/fight_detection_yolov8/blob/main/Yolo_nano_weights.pt
repo_id = "Musawer14/fight_detection_yolov8"
# filename = "yolo_small_weights.pt"
filename = "Yolo_nano_weights.pt"

# Download model
downloaded_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename
)



print("Downloaded from cache:")
print(downloaded_path)

# Copy to current project folder
destination = os.path.join(os.getcwd(), filename)

shutil.copy2(downloaded_path, destination)

print("\nModel copied successfully!")
print("Model path:")
print(destination)
print("File exists:", os.path.exists(destination))