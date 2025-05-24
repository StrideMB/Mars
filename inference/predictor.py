import torch
from misc.bbox import bboxDecode, nonMaxSuppression


class DetectionPredictor(object):
    def __init__(self, mcfg, model):
        self.mcfg = mcfg
        self.model = model
        self.model.setInferenceMode(True)

    def predictRaw(self, images):
        #print("images min:", images.min().item(), "max:", images.max().item())
        #if torch.isnan(images).any():
        #    print("Input image contains NaN!")

        
        with torch.no_grad():
            preds = self.model(images)

        #if any([torch.isnan(p).any() for p in preds]):
        #    print("Model output contains NaN!")

        #for name, param in self.model.named_parameters():
        #    if torch.isnan(param).any():
        #        print(f"Layer {name} has NaN weights!")
        #        break

        batchSize = preds[0].shape[0]
        no = self.mcfg.nc + self.mcfg.regMax * 4

        predBoxDistribution, predClassScores = torch.cat([xi.view(batchSize, no, -1) for xi in preds], 2).split((self.mcfg.regMax * 4, self.mcfg.nc), 1)

        #print("Raw predClassScores: min =", predClassScores.min().item(),
        #  "max =", predClassScores.max().item())
        
        predBoxDistribution = predBoxDistribution.permute(0, 2, 1).contiguous() # (batchSize, 8400, regMax * 4)
        predClassScores = predClassScores.sigmoid().permute(0, 2, 1).contiguous() # (batchSize, 8400, nc)
        
        #print("After sigmoid: predClassScores: min =", predClassScores.min().item(),
        #  "max =", predClassScores.max().item())
        
        # generate predicted bboxes
        predBboxes = bboxDecode(self.model.anchorPoints, predBoxDistribution, self.model.proj, xywh=False) # (batchSize, 8400, 4)
        predBboxes = predBboxes * self.model.anchorStrides

        results = nonMaxSuppression(
            predClassScores=predClassScores,
            predBboxes=predBboxes,
            scoreThres=0.2,
            iouThres=0.4,
            maxDetect=50,
        )

        return results
