import numpy as np
from PIL import Image
from misc import misc
from misc.img import rescale2Target, hsvAdjust
from misc.bbox import rescaleBoxes


class ImageTransformedInfo(object):
    def __init__(self, oriWidth, oriHeight, scaledWidth, scaledHeight, targetWidth, targetHeight, xoffset, yoffset, flip):
        self.oriWidth = oriWidth
        self.oriHeight = oriHeight
        self.scaledWidth = scaledWidth
        self.scaledHeight = scaledHeight
        self.targetWidth = targetWidth
        self.targetHeight = targetHeight
        self.xoffset = xoffset
        self.yoffset = yoffset
        self.flip = flip
        self.imgFile = None


class DataAugmentationProcessor(object):
    def __init__(self, inputShape, jitter=0.3, rescalef=(0.25, 2), flipProb=0.5, huef=0.1, satf=0.7, valf=0.4):
        self.inputShape = inputShape
        self.jitter = jitter
        self.rescalef = rescalef
        self.flipProb = flipProb
        self.huef = huef
        self.satf = satf
        self.valf = valf

    def processSimple(self, image, boxList):
        # rescale image
        targetHeight, targetWidth = self.inputShape
        oriWidth, oriHeight = image.size
        scaleFactor = min(targetWidth / oriWidth, targetHeight / oriHeight)
        scaledWidth = int(oriWidth * scaleFactor)
        scaledHeight = int(oriHeight * scaleFactor)
        newImage, xoffset, yoffset = rescale2Target(image, scaledWidth, scaledHeight, targetWidth, targetHeight)
        imageData = np.array(newImage, np.float32)
        # rescale boxes accordingly
        boxList = rescaleBoxes(boxList, oriWidth, oriHeight, scaledWidth, scaledHeight, targetWidth, targetHeight, xoffset, yoffset, False)
        transformInfo = ImageTransformedInfo(oriWidth, oriHeight, scaledWidth, scaledHeight, targetWidth, targetHeight, xoffset, yoffset, False)
        return imageData, boxList, transformInfo

    def processEnhancement(self, image, boxList):
        # rescale image with jitter
        targetHeight, targetWidth = self.inputShape
        oriWidth, oriHeight = image.size
        scalef = misc.randAB(self.rescalef[0], self.rescalef[1])
        oriAspectRatio = oriWidth / oriHeight
        newAspectRatio = oriAspectRatio * misc.randAB(1 - self.jitter, 1 + self.jitter) / misc.randAB(1 - self.jitter, 1 + self.jitter)
        if newAspectRatio < 1:
            scaledHeight = int(scalef * targetHeight)
            scaledWidth = int(newAspectRatio * scaledHeight)
        else:
            scaledWidth = int(scalef * targetWidth)
            scaledHeight = int(scaledWidth / newAspectRatio)
        newImage, xoffset, yoffset = rescale2Target(image, scaledWidth, scaledHeight, targetWidth, targetHeight)

        # flip image
        flipFlag = misc.randAB(0, 1) < self.flipProb
        if flipFlag:
            newImage = newImage.transpose(Image.FLIP_LEFT_RIGHT)

        # HSV adjustment
        imageData = hsvAdjust(newImage, self.huef, self.satf, self.valf)

        # rescale boxes accordingly
        boxList = rescaleBoxes(boxList, oriWidth, oriHeight, scaledWidth, scaledHeight, targetWidth, targetHeight, xoffset, yoffset, flipFlag)
        tinfo = ImageTransformedInfo(oriWidth, oriHeight, scaledWidth, scaledHeight, targetWidth, targetHeight, xoffset, yoffset, flipFlag)
        tinfo = []

        return imageData, boxList, tinfo
