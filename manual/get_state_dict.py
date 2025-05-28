from ultralytics import YOLO
import torch

# 加载官方预训练模型
model = YOLO('./resources/pretrained/yolov8n.pt')

# 模型参数字典
state_dict = model.model.state_dict()  # 注意：Ultralytics中YOLO对象的模型参数是model.model

# 保存state_dict到pth文件
torch.save(state_dict, './resources/pretrained/yolov8n_state_dict.pth')

print('state_dict保存完成')
