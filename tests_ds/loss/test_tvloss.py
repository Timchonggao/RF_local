import torch

def total_variation_loss(data, wx=1.0, wy=1.0, wz=1.0, return_grad=False):
    """
    计算形状为 [128, 128, 128, 22] 的张量的全变差损失。
    
    参数：
        data (torch.Tensor): 输入张量，形状为 [128, 128, 128, 22]
        wx, wy, wz (float): x, y, z 方向的权重
        return_grad (bool): 是否返回梯度张量
    
    返回：
        loss (torch.Tensor): TV 损失值（标量）
        grad (torch.Tensor, optional): TV 损失对 data 的梯度，形状与 data 相同
    """
    # 检查输入
    if data.shape != (128, 128, 128, 22):
        raise ValueError(f"Expected data shape [128, 128, 128, 22], got {data.shape}")
    if not data.is_cuda and not data.is_cpu:
        raise ValueError("data must be a CPU or CUDA tensor")
    if not data.is_contiguous():
        data = data.contiguous()

    # 确保 data 需要梯度（如果需要计算梯度）
    if return_grad and not data.requires_grad:
        data = data.requires_grad_(True)

    # 获取维度
    sz_i, sz_j, sz_k, sz_c = data.shape  # [128, 128, 128, 22]

    # 初始化 TV 损失
    tv_loss = 0.0

    # x 方向差值 (depth)
    if sz_i > 1:
        diff_x = data[1:, :, :, :] - data[:-1, :, :, :]  # u[i,j,k,c] - u[i-1,j,k,c]
        tv_loss += wx * torch.abs(diff_x).sum()

    # y 方向差值 (height)
    if sz_j > 1:
        diff_y = data[:, 1:, :, :] - data[:, :-1, :, :]  # u[i,j,k,c] - u[i,j-1,k,c]
        tv_loss += wy * torch.abs(diff_y).sum()

    # z 方向差值 (width)
    if sz_k > 1:
        diff_z = data[:, :, 1:, :] - data[:, :, :-1, :]  # u[i,j,k,c] - u[i,j,k-1,c]
        tv_loss += wz * torch.abs(diff_z).sum()

    # 归一化（模仿 CUDA 代码）
    tv_loss = tv_loss / 6.0

    if return_grad:
        # 计算梯度
        grad = torch.autograd.grad(
            tv_loss,
            data,
            create_graph=False,
            retain_graph=False
        )[0]
        return tv_loss, grad
    else:
        return tv_loss

# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    data = torch.randn(128, 128, 128, 22, device='cuda', requires_grad=True)

    # 计算 TV 损失
    loss = total_variation_loss(data, wx=1.0, wy=1.0, wz=1.0)
    print(f"TV Loss: {loss.item()}")

    # 计算 TV 损失和梯度
    loss, grad = total_variation_loss(data, wx=1.0, wy=1.0, wz=1.0, return_grad=True)
    print(f"TV Loss: {loss.item()}")
    print(f"Gradient shape: {grad.shape}")
    print(f"Sample gradient values: {grad[0, 0, 0, :5]}")
