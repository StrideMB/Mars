import torch

model = torch.load("/home/zxh/mars/run_root/zxh/yolov8n.pt")
print(model)
print(model.keys())