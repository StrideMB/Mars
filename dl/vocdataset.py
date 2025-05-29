import os
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from functools import partial
import pathlib
import numpy as np
from functools import partial
from misc.log import log
from misc import img, xml
from dl.aug import DataAugmentationProcessor
# add
import cv2
import PIL.Image as Image


class VocDataset(Dataset):
    @staticmethod
    def collate(batch):
        """
        Used by PyTorch DataLoader class (collate_fn)
        """
        images  = []
        labels  = []
        tinfos = []
        rawImages = []

        for i, data in enumerate(batch):
            img = data[0]
            label = data[1]
            images.append(img)
            label[:, 0] = i # enrich image index in batch
            labels.append(label)
            if len(data) > 3:
                tinfo = data[2]
                tinfos.append(tinfo)
                rawImage = data[3]
                rawImages.append(rawImage)

        images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
        labels  = torch.from_numpy(np.concatenate(labels, 0)).type(torch.FloatTensor)

        if len(rawImages) > 0:
            return images, labels, tinfos, rawImages

        return images, labels

    @staticmethod
    def workerInit(seed, workerId):
        workerSeed = workerId + seed
        random.seed(workerSeed)
        np.random.seed(workerSeed)
        torch.manual_seed(workerSeed)

    @staticmethod
    def getDataLoader(mcfg, splitName, isTest, fullInfo, selectedClasses=None):
        if splitName not in mcfg.subsetMap:
            raise ValueError("Split not found in mcfg: {}".format(splitName))

        dataset = VocDataset(
            imageDir=mcfg.imageDir,
            annotationDir=mcfg.annotationDir,
            classList=mcfg.classList,
            inputShape=mcfg.inputShape,
            subset=mcfg.subsetMap[splitName],
            isTest=isTest,
            fullInfo=fullInfo,
            suffix=mcfg.suffix,
            splitName=splitName,
            selectedClasses=selectedClasses,
        )
        return DataLoader(
            dataset,
            shuffle=True,
            batch_size=mcfg.batchSize,
            num_workers=mcfg.dcore,
            pin_memory=True,
            drop_last=False,
            sampler=None,
            collate_fn=VocDataset.collate,
            worker_init_fn=partial(VocDataset.workerInit, mcfg.seed)
        )

    def __init__(self, imageDir, annotationDir, classList, inputShape, subset, isTest, fullInfo, suffix, splitName, selectedClasses):
        super(VocDataset, self).__init__()
        self.imageDir = imageDir
        self.annotationDir = annotationDir
        self.classList = classList
        self.inputShape = inputShape
        self.augp = DataAugmentationProcessor(inputShape=inputShape)
        self.isTest = isTest
        self.fullInfo = fullInfo
        self.suffix = suffix
        self.splitName = splitName
        self.selectedClasses = selectedClasses

        if subset is None:
            self.imageFiles = [os.path.join(imageDir, x) for x in os.listdir(imageDir) if pathlib.Path(x).suffix == self.suffix]
        else:
            self.imageFiles = [os.path.join(imageDir, x) for x in subset]
            for imFile in self.imageFiles:
                if not os.path.exists(imFile):
                    raise ValueError("Image file in subset not exists: {}".format(imFile))
        if len(self.imageFiles) == 0:
            raise ValueError("Empty image directory: {}".format(imageDir))

        self.annotationFiles = [os.path.join(annotationDir, "{}.xml".format(pathlib.Path(x).stem)) for x in self.imageFiles]
        for annFile in self.annotationFiles:
            if not os.path.exists(annFile):
                raise ValueError("Annotation file not exists: {}".format(annFile))

        log.inf("VOC dataset [{}] initialized from {} with {} images".format(self.splitName, imageDir, len(self.imageFiles)))
        if self.selectedClasses is not None:
            log.inf("VOC dataset [{}] set with selected classes: {}".format(self.splitName, self.selectedClasses))

    def postprocess(self, imageData, boxList):
        imageData = imageData / 255.0
        imageData = np.transpose(np.array(imageData, dtype=np.float32), (2, 0, 1))
        boxList = np.array(boxList, dtype=np.float32)
        labels = np.zeros((boxList.shape[0], 6)) # add one dim (5 + 1 = 6) as image batch index (VocDataset.collate)
        if boxList.shape[0] > 0:
            boxList[:, [0, 2]] = boxList[:, [0, 2]] / self.inputShape[1]
            boxList[:, [1, 3]] = boxList[:, [1, 3]] / self.inputShape[0]
            labels[:, 1] = boxList[:, -1]
            labels[:, 2:] = boxList[:, :4]
        return imageData, labels

    def __len__(self):
        return len(self.imageFiles)

    def __getitem__(self, index):
        ii = index % len(self.imageFiles)
        imgFile = self.imageFiles[ii]
        image = img.loadRGBImage(imgFile)
        annFile = self.annotationFiles[ii]
        boxList = xml.XmlBbox.loadXmlObjectList(annFile, self.classList, selectedClasses=self.selectedClasses, asArray=True)

        if self.isTest:
            imageData, boxList, tinfo = self.augp.processSimple(image, boxList)
        else:
            #use_mosaic = np.random.rand() < 0.5
            use_mosaic = True  # disable mosaic for now
            if use_mosaic:
                imageData, boxList = self.mosaic(ii)
            else:
                imageData, boxList, tinfo = self.augp.processEnhancement(image, boxList)
                Image.fromarray(imageData.astype(np.uint8)).save("enhancement.jpg")
            
            
        imageData, labels = self.postprocess(imageData, boxList)
        if not self.fullInfo:
            return imageData, labels

        tinfo.imgFile = imgFile
        return imageData, labels, tinfo, image

        ## mixup augmentation
            #if np.random.rand() < 0.5:
                #mix_idx = np.random.randint(0, len(self.imageFiles))
                #mix_img = img.loadRGBImage(self.imageFiles[mix_idx])
                #mix_box = xml.XmlBbox.loadXmlObjectList(self.annotationFiles[mix_idx], self.classList, selectedClasses=self.selectedClasses, asArray=True)
                #mix_img_data, mix_box, _ = self.augp.processEnhancement(mix_img, mix_box)

                ## mixup image
                #alpha = 0.2
                #lam = np.random.beta(alpha, alpha)
                #imageData = lam * imageData + (1 - lam) * mix_img_data
                #boxList = np.concatenate((boxList, mix_box), axis=0)
    
    def mosaic(self, index):
        input_h, input_w = self.inputShape
        s = input_h  # 假设输入是正方形，使用高度作为基准
    
        # 随机生成mosaic中心点
        yc = int(random.uniform(0, 2 * s))
        xc = int(random.uniform(0, 2 * s))
    
        # 创建2倍大小的画布
        mosaic_img = None
        mosaic_boxes = []
    
        # 获取3个额外的随机索引
        indices = [index] + [random.randint(0, len(self.imageFiles) - 1) for _ in range(3)]
    
        for i in range(4):
            # 加载图像和标注
            idx = indices[i]
            img_path = self.imageFiles[idx]
            ann_path = self.annotationFiles[idx]
        
            img_ = img.loadRGBImage(img_path)
            box_ = xml.XmlBbox.loadXmlObjectList(ann_path, self.classList, 
                                               selectedClasses=self.selectedClasses, 
                                               asArray=True)
        
            # 数据增强处理
            img_, box_, _ = self.augp.processEnhancement(img_, box_)
            box_ = box_.astype(np.float32)  # 确保是float类型
        
            h, w = img_.shape[:2]
        
            # 计算在mosaic中的位置
            if i == 0:  # top left
                if mosaic_img is None:
                    mosaic_img = np.full((s * 2, s * 2, img_.shape[2]), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        
            # 将图像放置到mosaic画布上
            mosaic_img[y1a:y2a, x1a:x2a] = img_[y1b:y2b, x1b:x2b]
        
            # 计算padding并更新box坐标
            padw = x1a - x1b
            padh = y1a - y1b
        
            if box_.shape[0] > 0:
                # 更新box坐标：加上padding偏移
                box_[:, [0, 2]] += padw  # x坐标
                box_[:, [1, 3]] += padh  # y坐标
                mosaic_boxes.append(box_)
    
        # 合并所有boxes
        if mosaic_boxes:
            mosaic_boxes = np.concatenate(mosaic_boxes, axis=0)
            # 限制box坐标在mosaic图像范围内
            mosaic_boxes[:, 0:4:2] = np.clip(mosaic_boxes[:, 0:4:2], 0, s * 2)  # x坐标
            mosaic_boxes[:, 1:4:2] = np.clip(mosaic_boxes[:, 1:4:2], 0, s * 2)  # y坐标
        else:
            mosaic_boxes = np.zeros((0, 5), dtype=np.float32)
    
        # 将mosaic图像resize回原始输入大小
        mosaic_img = cv2.resize(mosaic_img, (input_w, input_h))
        #Image.fromarray(mosaic_img.astype(np.uint8)).save("mosaic.jpg")
    
        # 相应地缩放box坐标
        scale_x = input_w / (s * 2)
        scale_y = input_h / (s * 2)
        mosaic_boxes[:, [0, 2]] *= scale_x
        mosaic_boxes[:, [1, 3]] *= scale_y
    
        return mosaic_img, mosaic_boxes