import os
from config import mconfig


def mcfg(tags):
    mcfg = mconfig.ModelConfig()
    #projectRootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #pretrainedFile = os.path.join(projectRootDir, "resources/pretrained/backbone", "backbone_{}.pth".format(mcfg.phase))
    #mcfg.pretrainedBackboneUrl = "file://{}".format(pretrainedFile)

    mcfg.phase = "nano" # DO NOT MODIFY
    mcfg.trainSplitName = "train" # DO NOT MODIFY
    mcfg.validationSplitName = "validation" # DO NOT MODIFY
    mcfg.testSplitName = "test" # DO NOT MODIFY

    # data setup
    mcfg.imageDir = "/home/zxh/mars/mar20/images"
    mcfg.annotationDir = "/home/zxh/mars/mar20/annotations"
    mcfg.classList = ["A{}".format(x) for x in range(1, 21)] # DO NOT MODIFY
    mcfg.subsetMap = { # DO NOT MODIFY
        "train": "/home/zxh/mars/mar20/splits/v5/train.txt",
        "validation": "/home/zxh/mars/mar20/splits/v5/validation.txt",
        "test": "/home/zxh/mars/mar20/splits/v5/test.txt",
        "small": "/home/zxh/mars/mar20/splits/v5/small.txt",
    }

    if "full" in tags:
        mcfg.modelName = "base"
        mcfg.maxEpoch = 500
        #mcfg.backboneFreezeEpochs = [x for x in range(0, 100)]
        mcfg.backboneFreezeEpochs = []

    if "teacher" in tags:
        mcfg.modelName = "base"
        mcfg.maxEpoch = 200
        mcfg.backboneFreezeEpochs = [x for x in range(0, 100)]
        mcfg.trainSelectedClasses = ["A{}".format(x) for x in range(1, 11)] # DO NOT MODIFY

    if "distillation" in tags:
        mcfg.modelName = "distillation"
        mcfg.checkpointModelFile = "/home/zxh/mars/ame/c1.nano.teacher/__cache__/best_weights.pth"
        mcfg.teacherModelFile = "/home/zxh/mars/ame/c1.nano.teacher/__cache__/best_weights.pth"
        mcfg.distilLossWeights = (1.0, 0.05, 0.001)
        mcfg.maxEpoch = 100
        mcfg.backboneFreezeEpochs = [x for x in range(0, 25)]
        mcfg.epochValidation = False # DO NOT MODIFY
        mcfg.trainSplitName = "small" # DO NOT MODIFY
        mcfg.teacherClassIndexes = [x for x in range(0, 10)] # DO NOT MODIFY

    return mcfg
