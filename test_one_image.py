import cv2
from web.backend.services.pipeline_runner import runner
import sys

image_path = r"D:\mdm_proj\Digitization-of-ECG-Image\train\129883643\129883643-0001.png"
image = cv2.imread(image_path)
if image is None:
    print("Could not load image")
    sys.exit(1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Running stage 0")
try:
    normalised, keypoint, homo = runner.run_stage0(image)
    print("Keypoints:", len(keypoint))
except Exception as e:
    print("Stage 0 failed:", e)

print("Running stage 1")
try:
    rectified, grid = runner.run_stage1(normalised)
    print("Grid:", grid.shape)
except Exception as e:
    print("Stage 1 failed:", e)

print("Running stage 2")
try:
    leads = runner.run_stage2(rectified)
    print("Leads:", leads.keys())
except Exception as e:
    print("Stage 2 failed:", e)
