import time
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
# device=torch.device('cuda:0')
device=torch.device('cpu')

# 設置圖像讀取和轉換
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

# 載入自己的圖像數據集
# dataset = ImageFolder(root=r'datasets\train\img', transform=transform)
dataset = ImageFolder(root=r'C:\develop\CRNN_Chinese_Characters_Rec_data\datasets', transform=transform)

# 計算平均值和標準差
data_loader = torch.utils.data.DataLoader(dataset, batch_size=50000, shuffle=False, num_workers=0)
mean = 0.
std = 0.
total_samples = 0

stime = time.time()
for inputs, _ in data_loader:
    inputs = inputs.to(device)
    batch_samples = inputs.size(0)
    inputs = inputs.view(batch_samples, inputs.size(1), -1)
    mean += inputs.mean(2).sum(0)
    std += inputs.std(2).sum(0)
    total_samples += batch_samples
    etime = time.time()
    print(mean/total_samples, std/total_samples, total_samples/batch_samples, etime-stime)     # 0.8281, 0.1979 
    stime = etime

mean /= total_samples
std /= total_samples

print('Mean:', mean)
print('Std:', std)


# Mean: tensor([0.8281])
# Std: tensor([0.1979])
