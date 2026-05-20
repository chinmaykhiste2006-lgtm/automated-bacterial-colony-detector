from ultralytics import YOLO

model = YOLO("models/unet/best.pt")

print("TASK:", model.task)
print("MODEL:", model)