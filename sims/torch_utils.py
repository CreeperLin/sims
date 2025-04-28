import torch


@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = [q[..., i] for i in [0, 1, 2, 3]]
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = qw * qw - qx * \
                qx - qy * qy + qz * qz
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    # pitch = torch.where(
    # torch.abs(sinp) >= 1, copysign(torch.pi / 2.0, sinp), torch.asin(sinp))
    sinp = torch.clip(sinp, -1, 1)
    pitch = torch.asin(sinp)
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = qw * qw + qx * \
                qx - qy * qy - qz * qz
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack((roll, pitch, yaw), dim=-1)
