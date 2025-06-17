import torch.nn as nn
import torch.nn.functional as F



class NN_FCBNRL_MM(nn.Module):
    def __init__(self, in_dim, out_dim, channel=8, use_RL=True):
        super(NN_FCBNRL_MM, self).__init__()
        m_l = []
        m_l.append(
            nn.Linear(
                in_dim,
                out_dim,
            )
        )
        if use_RL:
            m_l.append(nn.LeakyReLU(0.1))
        m_l.append(nn.BatchNorm1d(out_dim))

        self.block = nn.Sequential(*m_l)

    def forward(self, x):
        return self.block(x)
class ResidualBlock(nn.Module):
    """
         Every ResidualBlock needs to ensure that the dimensions of input and output remain unchanged
         So the number of channels of the convolution kernel is set to the same
    """
    def __init__(self, channel):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

    def forward(self, x):
        """
                 There is a jump connection in ResidualBlock;
                 When the second convolution result is obtained, the input of the residual block needs to be added,
                 Then activate the result to achieve jump connection ==> to avoid the disappearance of the gradient
                 In the derivation, because the original input x is added, the gradient is: dy + 1, near 1
        """
        y = F.relu(self.conv1(x))
        y = self.conv2(y)

        return F.relu(x + y)


class ResNetMnist(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.res_block_1 = ResidualBlock(16)
        self.res_block_2 = ResidualBlock(32)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Flatten()
        self.fc2 = NN_FCBNRL_MM(512,out_dim)

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        in_size = x.size(0)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = self.res_block_1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.res_block_2(x)
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x