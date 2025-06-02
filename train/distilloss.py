import torch
#from overrides import override # this could be removed since Python 3.12
import torch.nn.functional as F
from train.loss import DetectionLoss


class DistillationDetectionLoss(object):
    def __init__(self, mcfg, model):
        self.mcfg = mcfg
        self.histMode = False
        self.detectionLoss = DetectionLoss(mcfg, model)
        self.cwdLoss = CWDLoss(self.mcfg.device, self.mcfg.temperature1)
        self.respLoss = ResponseLoss(self.mcfg.device, self.mcfg.nc, self.mcfg.teacherClassIndexes, self.mcfg.temperature2)
        #raise NotImplementedError("DistillationDetectionLoss::__init__")

    #@override
    def __call__(self, rawPreds, batch):
        """
        rawPreds[0] & rawPreds[1] shape: (
            (B, regMax * 4 + nc, 80, 80),
            (B, regMax * 4 + nc, 40, 40),
            (B, regMax * 4 + nc, 20, 20),
            (B, 128 * w, 160, 160),
            (B, 256 * w, 80, 80),
            (B, 512 * w, 40, 40),
            (B, 512 * w * r, 20, 20),
            (B, 512 * w, 40, 40),
            (B, 256 * w, 80, 80),
            (B, 512 * w, 40, 40),
            (B, 512 * w * r, 20, 20),
        )
        """
        spreds = rawPreds[0]
        tpreds = rawPreds[1]

        sresponse, sfeats = spreds[:3], spreds[3:]
        tresponse, tfeats = tpreds[:3], tpreds[3:]

        loss = torch.zeros(3, device=self.mcfg.device)  # original, cwd distillation, response distillation
        loss[0] = self.detectionLoss(sresponse, batch) * self.mcfg.distilLossWeights[0]  # original
        loss[1] = self.cwdLoss(sfeats, tfeats) * self.mcfg.distilLossWeights[1]  # cwd distillation
        loss[2] = self.respLoss(sresponse, tresponse) * self.mcfg.distilLossWeights[2]  # response distillation

        return loss.sum()

class CWDLoss(object):
    def __init__(self, device, temperature=1.0):
        self.device = device
        self.temperature = temperature
    
    def __call__(self, sfeats, tfeats):
        """
        sfeats: list of tensors, each tensor shape: (B, C, H, W)
        tfeats: list of tensors, each tensor shape: (B, C, H, W)
        """
        loss = 0.0

        for sfeat, tfeat in zip(sfeats, tfeats):
            # reshape to (B, C, H * W)
            B, C, H, W = sfeat.shape
            s_flat = sfeat.view(B, C, -1)
            t_flat = tfeat.view(B, C, -1)

            # softmax over channel dimension
            s_dist = F.log_softmax(s_flat / self.temperature, dim=1)
            t_dist = F.softmax(t_flat / self.temperature, dim=1)

            kl = F.kl_div(s_dist, t_dist, reduction='batchmean') * self.temperature ** 2
            loss += kl

        return loss / len(sfeats)

class ResponseLoss(object):
    def __init__(self, device, nc, teacherClassIndexes, temperature=1.0):
        self.device = device
        self.nc = nc
        self.teacherClassIndexes = teacherClassIndexes
        self.temperature = temperature


    def __call__(self, sresponse, tresponse):
        """
        sresponse: list of tensors, each tensor shape: (B, regMax * 4 + nc, H, W)
        tresponse: list of tensors, each tensor shape: (B, regMax * 4 + nc, H, W)
        """
        loss = 0.0
        cls_start = -self.nc

        for sfeat, tfeat in zip(sresponse, tresponse):
            # shape: (B, nc, H, W)
            slogits = sfeat[:, cls_start:, :, :]
            tlogits = tfeat[:, cls_start:, :, :]

            # only distill for shared classes
            if self.teacherClassIndexes is not None:
                slogits = slogits[:, self.teacherClassIndexes, :, :]
                tlogits = tlogits[:, self.teacherClassIndexes, :, :]
            
            # reshape to (B * H * W, C)
            B, C, H, W = slogits.shape
            slogits = slogits.permute(0, 2, 3, 1).reshape(-1, C)
            tlogits = tlogits.permute(0, 2, 3, 1).reshape(-1, C)

            # apply temperature scaling
            slogits = slogits / self.temperature
            tlogits = tlogits / self.temperature

            # softmax -> KL divergence
            t_soft = F.softmax(tlogits, dim=1)
            s_log_soft = F.log_softmax(slogits, dim=1)

            kl = F.kl_div(s_log_soft, t_soft, reduction='batchmean') * self.temperature ** 2
            loss += kl

        return loss / len(sresponse)
        