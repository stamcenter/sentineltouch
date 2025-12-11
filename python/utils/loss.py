import torch
import math

class ArcFace(torch.nn.Module):
    def __init__(self, sp=64.0, sn=64.0, m=0.5, **kwargs):
        super(ArcFace, self).__init__()
        self.sp = sp / sn  # sn will be multiplied again
        self.sn = sn
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

    def forward(self, cosine: torch.Tensor, label):
        cosine = cosine.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]

        cos_theta = cosine[index]
        
        # print(f"cos_theta: {cos_theta}")
        # print("cos_theta shape:", cos_theta.shape)
        # print("labels min/max:", label.min().item(), label.max().item())

        target_logit = cos_theta[torch.arange(0, cos_theta.size(0)), label[index]].view(
            -1, 1
        )
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = (target_logit * self.cos_m - sin_theta * self.sin_m).to(
            cosine.dtype
        )  # cos(target+margin)

        cosine[index] = cosine[index].scatter(
            1, label[index, None], cos_theta_m * self.sp
        )
        cosine.mul_(self.sn)

        return cosine
    
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distances, labels):
        loss_contrastive = torch.mean((1-labels) * torch.pow(distances, 2) +
                                      (labels) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2))
        return loss_contrastive