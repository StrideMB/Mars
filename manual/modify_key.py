import torch

def load_new_keys(path):
    with open(path, 'r') as f:
        return [l.strip() for l in f if l.strip()]

if __name__ == "__main__":
    # 1. 加载原始权重
    src = './resources/pretrained/yolov8n_state_dict.pth'
    dst = './resources/pretrained/backbone/backbone_nano.pth'
    state_dict = torch.load(src, map_location='cpu')

    # 2. 读取 new_keys.txt（请按顺序把你列出的所有 key 放进去）
    new_keys = load_new_keys('new_keys.txt')

    # 3. 校验并重命名
    old_keys = list(state_dict.keys())
    print(len(old_keys), len(new_keys))
    # assert len(old_keys) == len(new_keys), "old_keys 与 new_keys 数量不一致！"
    new_state = {new_k: state_dict[old_k] for old_k, new_k in zip(old_keys, new_keys)}

    # 4. 保存
    torch.save(new_state, dst)
    print(f"重命名完成，已保存到 {dst}")