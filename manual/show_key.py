import torch
 
if __name__ == "__main__":
 
    weights_files_pretrained = './resources/pretrained/yolov8n_state_dict.pth' # 权重文件路径
    weights_files_mine = '/home/zxh/mars/run_root/zxh/c1.nano.full/__cache__/best_weights.pth' # 权重文件路径
    weights_pretrained= torch.load(weights_files_pretrained) # 加载权重文件
    weights_mine = torch.load(weights_files_mine) # 加载权重文件

    print("Pretrained weights keys:\n")
    for key, v in weights_pretrained.items():
        print(key)

    print("\nMy weights keys:\n")
    for key, v in weights_mine.items():
        print(key)
    
    weights_modified_path = './resources/pretrained/backbone/backbone_nano.pth'
    weights_modified = torch.load(weights_modified_path) # 加载修改后的权重文件
    print("modified weights keys:\n")
    for key, v in weights_modified.items():
        print(key)
    