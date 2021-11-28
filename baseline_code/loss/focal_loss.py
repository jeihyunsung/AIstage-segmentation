import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FOCALLOSS(nn.Module):
    def __init__(self, alpha, gamma, reduction = 'mean', eps=1e-8, classes=11):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.classes = classes
    
    def forward(self, input, target):
        if input.size(0) != target.size(0):
            raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

        n = input.size(0) # B
        out_size = (n,) + input.size()[2:]

        if target.size()[1:] != input.size()[2:]:
            raise ValueError(f'Expected target size {out_size}, got {target.size()}')

        if not input.device == target.device:
            raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

        if isinstance(self.alpha, float):
            alpha = self.alpha
            pass
        elif isinstance(self.alpha, np.ndarray):
            alpha = torch.from_numpy(np.array([self.alpha for _ in range(n)])).to(input.device)
            #self.alpha = torch.from_numpy(self.alpha).to(input.device)
            # alpha : (B, C, H, W)
            alpha = alpha.view(-1, len(alpha[0]), 1, 1).expand_as(input)
        elif isinstance(self.alpha, torch.Tensor):
            # alpha : (B, C, H, W)
            alpha = self.alpha.view(-1, len(self.alpha), 1, 1).expand_as(input)
            alpha.to(input.device)

        input_soft = F.softmax(input, dim=1) + self.eps

        #target_one_hot = F.one_hot(target, num_classes=self.classes)     
        shape = target.shape   
        one_hot = torch.zeros((shape[0], self.classes)+shape[1:], device=input.device)
        target_one_hot = one_hot.scatter_(1, target.unsqueeze(1), 1.0)

        weight = torch.pow(1.0 - input_soft, self.gamma)
        focal = -alpha * weight * torch.log(input_soft)
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)

        if self.reduction == 'none':
            # loss : (B, H, W)
            loss = loss_tmp
        elif self.reduction == 'mean':
            # loss : scalar
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            # loss : scalar
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")
        return loss


if  __name__ == "__main__":
    focal_loss = FOCALLOSS(alpha=0.75, gamma=2, classes=11)
    x = torch.randn([2, 11, 512, 512])
    y = torch.randint(0, 10, [2,512,512])
    output = focal_loss(x,y)
    print(output.size())