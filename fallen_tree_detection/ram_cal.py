
import torch
import time
from ultralytics import YOLO

MODEL_PATH = "best.pt"
IMAGE_PATH = "test.jpg"
DEVICE = 0  # NVIDIA GPU 0

if not torch.cuda.is_available():
    print("CUDA GPU not available!")
    raise SystemExit(1)

def mb(value):
    return value / (1024 ** 2)

torch.cuda.init()
torch.cuda.synchronize(DEVICE)

print("GPU:", torch.cuda.get_device_name(DEVICE))
print("Total VRAM:",
      round(torch.cuda.get_device_properties(DEVICE).total_memory / 1024**3, 2),
      "GB")

# Baseline before loading the model
torch.cuda.reset_peak_memory_stats(DEVICE)
baseline = torch.cuda.memory_allocated(DEVICE)

# Load model on GPU
start = time.perf_counter()
model = YOLO(MODEL_PATH)
model.to(f"cuda:{DEVICE}")
torch.cuda.synchronize(DEVICE)
load_time = time.perf_counter() - start

after_load = torch.cuda.memory_allocated(DEVICE)
load_peak = torch.cuda.max_memory_allocated(DEVICE)

print("\n--- MODEL LOAD ---")
print(f"Load time: {load_time:.3f} sec")
print(f"Allocated after load: {mb(after_load):.2f} MB")
print(f"Load memory increase: {mb(after_load - baseline):.2f} MB")
print(f"Peak allocated during load: {mb(load_peak):.2f} MB")

# Warm-up: first inference may allocate extra memory
model.predict(
    source=IMAGE_PATH,
    imgsz=640,
    device=DEVICE,
    verbose=False
)
torch.cuda.synchronize(DEVICE)

# Reset peak measurement for inference
torch.cuda.reset_peak_memory_stats(DEVICE)
before_inference = torch.cuda.memory_allocated(DEVICE)

start = time.perf_counter()
for _ in range(20):
    model.predict(
        source=IMAGE_PATH,
        imgsz=640,
        device=DEVICE,
        verbose=False
    )
torch.cuda.synchronize(DEVICE)
elapsed = time.perf_counter() - start

inference_peak = torch.cuda.max_memory_allocated(DEVICE)
reserved_peak = torch.cuda.max_memory_reserved(DEVICE)
current_allocated = torch.cuda.memory_allocated(DEVICE)

print("\n--- INFERENCE ---")
print(f"20 inference calls: {elapsed:.3f} sec")
print(f"Average time/call: {elapsed / 20:.4f} sec")
print(f"Allocated before inference: {mb(before_inference):.2f} MB")
print(f"Peak allocated VRAM: {mb(inference_peak):.2f} MB")
print(f"Peak reserved VRAM: {mb(reserved_peak):.2f} MB")
print(f"Currently allocated: {mb(current_allocated):.2f} MB")
