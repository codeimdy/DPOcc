import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # Fully connected layer from input_dim to output_dim

    def forward(self, x):
        B, C, H, W, _ = x.shape  # Extract dimensions
        x = x.view(B, C, H, W, -1)  # Flatten the last dimensions
        x = self.fc(x)  # [2, 128, 128, 128, 3] -> [2, 128, 128, 128, 64]
        x = F.relu(x)  # ReLU activation (dimension unchanged) [2, 128, 128, 128, 64]
        return x.view(B, C, H, W, -1)  # Restore dimensions

class SparseConvConstructor(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(SparseConvConstructor, self).__init__()
        self.conv = nn.Conv3d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2)  # 3D Convolution

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)  # Rearrange dimensions: [2, 64, 128, 128, 128] -> [2, 128, 128, 128, 128]
        x = self.conv(x)  # Apply convolution
        return x.permute(0, 2, 3, 4, 1)  # Restore dimensions

class VoxelToPillar(nn.Module):
    def __init__(self):
        super(VoxelToPillar, self).__init__()

    def forward(self, x):
        min_values = x.min(dim=-1, keepdim=True)[0]  # Min along D [2, 128, 128, 128, 1]
        max_values = x.max(dim=-1, keepdim=True)[0]  # Max along D [2, 128, 128, 128, 1]
        mean_values = x.mean(dim=-1, keepdim=True)  # Mean along D [2, 128, 128, 128, 1]
        return torch.cat([min_values, max_values, mean_values], dim=-1)  # Concat along the last dim [2, 128, 128, 128, 3]

class DetectionNetwork(nn.Module):
    def __init__(self):
        super(DetectionNetwork, self).__init__()
        self.mlp = MLP(input_dim=3, output_dim=64)  # Input to MLP is 3, output is 64
        self.sparse_conv = SparseConvConstructor(input_dim=64, output_dim=128, kernel_size=3)  # Input to Conv3d is 64, output is 128
        self.voxel_to_pillar = VoxelToPillar()  # Handles dimensional reduction and feature extraction

    def forward(self, x):
        x = self.voxel_to_pillar(x)  # [2, 128, 128, 128, 16] -> [2, 128, 128, 128, 3]
        x = self.mlp(x)  # [2, 128, 128, 128, 3] -> [2, 128, 128, 128, 64]
        x = self.sparse_conv(x)  # [2, 128, 128, 128, 64] -> [2, 128, 128, 128, 128]
        final_conv = nn.Conv3d(128, 16, kernel_size=1)  # 1x1x1 convolution to reduce channel dimension
        output = final_conv(x.permute(0, 4, 1, 2, 3))  # Rearrange and apply convolution
        output = output.permute(0, 2, 3, 4, 1)
        return output
def main():
    input_tensor = torch.randn(2, 128, 128, 128, 16)  # Initial input dimensions
    model = DetectionNetwork()
    output = model(input_tensor)  # Calculate model output for given input

    print(output.shape)  # Print the shape of the output tensor

if __name__ == '__main__':
    main()
