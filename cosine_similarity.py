import torch
import torch.nn.functional as F
import time
torch.set_grad_enabled(False)

def get_data():
    data = torch.randn(2048, 768, device="cuda:0")   # 2048 vectors, each is 768-dimensional
    return data

# implementation 1
s_time = time.time()
for _ in range(100):
    data = get_data()
    similarity = F.cosine_similarity(data.unsqueeze(0), data.unsqueeze(1), dim=-1)
print("time for implementation 1:", time.time() - s_time)


# implementation 2
s_time = time.time()
for _ in range(100):
    data = get_data()
    data_normed = data / torch.linalg.vector_norm(data, dim=-1, keepdim=True).clamp(min=1e-8)
    similarity = data_normed @ data_normed.T
print("time for implementation 2:", time.time() - s_time)

# On one H100
# time for implementation 1: 4.3520119190216064
# time for implementation 2: 1.3465051651000977
