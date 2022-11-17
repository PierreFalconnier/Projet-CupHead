from torch import nn
import copy
import math

# A MODIFIER POUR AVOIR L'inputsize en hyperparametre
class CupHeadNet(nn.Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.c, self.h, self.w = input_dim

        # if self.h != 84:
        #     raise ValueError(f"Expecting input height: 84, got: {self.h}")
        # if self.w != 84:
        #     raise ValueError(f"Expecting input width: 84, got: {self.w}")

        if self.h != self.w:
            raise ValueError(f"Expecting a square image")

        # paramètres du model

        self.kernel_size_list = [8,4,3]
        self.stride_size_list = [4,2,1]
        self.in_channels_list = [self.c,32,64]
        self.out_channels_list = [32,64,64]

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels_list[0], out_channels=self.out_channels_list[0], kernel_size=self.kernel_size_list[0], stride=self.stride_size_list[0]),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.in_channels_list[1], out_channels=self.out_channels_list[1], kernel_size=self.kernel_size_list[1], stride=self.stride_size_list[1]),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.in_channels_list[2], out_channels=self.out_channels_list[2], kernel_size=self.kernel_size_list[2], stride=self.stride_size_list[2]),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.compute_linear_dimensions(), 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        # self.online = nn.Sequential(
        #     nn.Conv2d(in_channels=self.c, out_channels=32, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(self.compute_linear_dimensions(), 512),
        #     nn.ReLU(),
        #     nn.Linear(512, output_dim),
        # )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
    
    def compute_linear_dimensions(self):
        h = self.h
        for k in range(len(self.kernel_size_list)):
            h = math.floor(1+(h+2*0-1*(self.kernel_size_list[k]-1)-1)/self.stride_size_list[k])
        linear_dimension = h*h*self.out_channels_list[-1]
        print(linear_dimension) 
        return linear_dimension
    



if __name__=='__main__':
    import torch
    from agent import CupHead


    SHAPE = (1,84,84)
    DIM_OUT = 5
    MODEL = CupHeadNet(SHAPE,DIM_OUT).float()
    print(MODEL.compute_linear_dimensions) 
    T = torch.rand(SHAPE)
    print(MODEL(T, "online"))