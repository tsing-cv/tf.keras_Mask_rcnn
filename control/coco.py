# -*- coding: utf-8 -*-
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Author      : tsing-cv
#   Created date: 2019-02-20 18:32:26
#
#================================================================
import os
import sys
import time
import numpy as np
import imgaug
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

sys.path.append(ROOT_DIR) 
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils
from tools import visualize


# FIXME arguments-------------------------------------------
# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2017"
AUTO_DOWNLOAD = False
# ----------------------------------------------------------


############################################################
#  Configurations
############################################################
class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco" # "coco" or other name( for your own dataset name )

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # background + numoflabels


    TRAIN_BN = False



############################################################
#  Dataset
############################################################
class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
                  return_coco=False, auto_download=False):
        """Load a subset of the COCO dataset.
        Params:
            Dir  dataset_dir:   The root directory of the COCO dataset.
            Str  subset:        What to load (train, val, minival, valminusminival)
            Time year:          What dataset year to load (2014, 2017) as a string, not an integer
            List class_ids:     If provided, only loads images that have the given classes.
            Bool return_coco:   If True, returns the COCO object.
            Bool auto_download: Automatically download and unzip MS-COCO images and annotations
        """
        
        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)

        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/images/{}{}".format(dataset_dir, subset, year)
        print ("Images path\n\t{}\nAnnotations path\n\t{}".format(
            os.path.abspath("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year)),
            os.path.abspath(image_dir)))

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            # about coco, please refer https://github.com/waleedka/coco/tree/master/PythonAPI/pycocotools
            class_ids = sorted(coco.getCatIds())


        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
            
        else:
            # All images
            image_ids = list(coco.imgs.keys())


        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(source      = "coco", 
                           image_id    = i,
                           path        = os.path.join(image_dir, coco.imgs[i]['file_name']),
                           width       = coco.imgs[i]["width"],
                           height      = coco.imgs[i]["height"],
                           annotations = coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def auto_download(self, dataDir, dataType, dataYear):
        """Download the COCO dataset/annotations if requested.
        Params:
            dataDir: The root directory of the COCO dataset.
            dataType: What to load (train, val, minival, valminusminival)
            dataYear: What dataset year to load (2014, 2017) as a string, not an integer
        Note:
            For 2014, use "train", "val", "minival", or "valminusminival"
            For 2017, only "train" and "val" annotations are available
        """
        # Setup paths and file names
        if dataType == "minival" or dataType == "valminusminival":
            imgDir     = "{}/{}{}".format(dataDir, "val", dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
            imgURL     = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
        else:
            imgDir     = "{}/{}{}".format(dataDir, dataType, dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
            imgURL     = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)
        # print("Image paths:"); print(imgDir); print(imgZipFile); print(imgURL)

        # Create main folder if it doesn't exist yet
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)

        # Download images if not available locally
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)
            print("Downloading images to " + imgZipFile + " ...")
            with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
            print("Unzipping " + imgZipFile)
            with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
                zip_ref.extractall(dataDir)
            print("... done unzipping")
        print("Will use images in " + imgDir)

        # Setup annotations data paths
        annDir = "{}/annotations".format(dataDir)
        if dataType == "minival":
            annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
            annFile    = "{}/instances_minival2014.json".format(annDir)
            annURL     = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
            unZipDir   = annDir
        elif dataType == "valminusminival":
            annZipFile = "{}/instances_valminusminival2014.json.zip".format(dataDir)
            annFile    = "{}/instances_valminusminival2014.json".format(annDir)
            annURL     = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
            unZipDir   = annDir
        else:
            annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
            annFile    = "{}/instances_{}{}.json".format(annDir, dataType, dataYear)
            annURL     = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
            unZipDir   = dataDir
        # print("Annotations paths:"); print(annDir); print(annFile); print(annZipFile); print(annURL)

        # Download annotations if not available locally
        if not os.path.exists(annDir):
            os.makedirs(annDir)
        if not os.path.exists(annFile):
            if not os.path.exists(annZipFile):
                print("Downloading zipped annotations to " + annZipFile + " ...")
                with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
                    shutil.copyfileobj(resp, out)
                print("... done downloading.")
            print("Unzipping " + annZipFile)
            with zipfile.ZipFile(annZipFile, "r") as zip_ref:
                zip_ref.extractall(unZipDir)
            print("... done unzipping")
        print("Will use annotations in " + annFile)


    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(self.__class__, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id("coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"], image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(self.__class__, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(self.__class__, self).image_reference(image_id)


    # The following two functions are from pycocotools with a few changes.
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        Return: 
            binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        Return: 
            binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

############################################################
#  Show pipeline
############################################################
def get_ax(rows=1, cols=1, size=16):
    """返回一个在该notebook中用于所有可视化的Matplotlib Axes array。
    提供一个中央点坐标来控制graph的尺寸。
    
    调整attribute的尺寸来控制渲染多大的图像
    Params:
        rows: subset rows number
        cols: subset cols number
        size: figure size
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def show_rpn_predict(model, image, gt_class_id, gt_bbox):
    """Show rpn processing, draw filtered anchors
    Params:
        Class   model:       mrcnn model
        Ndarray image:       image
        Int     gt_class_id: class id of groundtruth
        List    gt_bbox:     bbox of groundtruth
    """
    # 生成RPN trainig targets
    # target_rpn_match=1是positive anchors, -1是negative anchors
    # 0是neutral anchors.
    target_rpn_match, target_rpn_bbox = modellib.build_rpn_targets(
        image.shape, model.anchors, gt_class_id, gt_bbox, model.config)
    modellib.log("target_rpn_match", target_rpn_match)
    modellib.log("target_rpn_bbox", target_rpn_bbox)
    
    positive_anchor_ix = np.where(target_rpn_match[:] == 1)[0]
    negative_anchor_ix = np.where(target_rpn_match[:] == -1)[0]
    neutral_anchor_ix = np.where(target_rpn_match[:] == 0)[0]
    positive_anchors = model.anchors[positive_anchor_ix]
    negative_anchors = model.anchors[negative_anchor_ix]
    neutral_anchors = model.anchors[neutral_anchor_ix]
    modellib.log("positive_anchors", positive_anchors)
    modellib.log("negative_anchors", negative_anchors)
    modellib.log("neutral anchors", neutral_anchors)
    
    #将refinement deltas应用于positive anchors
    refined_anchors = utils.apply_box_deltas(
        positive_anchors,
        target_rpn_bbox[:positive_anchors.shape[0]] * model.config.RPN_BBOX_STD_DEV)
    modellib.log("refined_anchors", refined_anchors, )
    
    #显示refinement (点)之前的positive anchors和refinement (线)之后的positive anchors.
    ax = get_ax(1, 2)
    visualize.draw_boxes(image, boxes=positive_anchors, 
                         refined_boxes=refined_anchors, 
                         title="Positive and Refined anchors", ax=ax[0])


    pillar = model.keras_model.get_layer("ROI").output  # node to start searching from

    # TF 1.4 and 1.9 introduce new versions of NMS. Search for all names to support TF 1.3~1.10
    nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression:0")
    if nms_node is None:
        nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV2:0")
    if nms_node is None: #TF 1.9-1.10
        nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV3:0")

    rpn = model.run_graph([image], [
        ("rpn_class", model.keras_model.get_layer("rpn_class").output),
        ("pre_nms_anchors", model.ancestor(pillar, "ROI/pre_nms_anchors:0")),
        ("refined_anchors", model.ancestor(pillar, "ROI/refined_anchors:0")),
        ("refined_anchors_clipped", model.ancestor(pillar, "ROI/refined_anchors_clipped:0")),
        ("post_nms_anchor_ix", nms_node),
        ("proposals", model.keras_model.get_layer("ROI").output),
    ])

    # Show top anchors by score (before refinement)
    limit = 50
    sorted_anchor_ids = np.argsort(rpn['rpn_class'][:,:,1].flatten())[::-1]
    visualize.draw_boxes(image, boxes=model.anchors[sorted_anchor_ids[:limit]], title="Top 100 anchors", ax=ax[1])

    # Show top anchors with refinement. Then with clipping to image boundaries
    limit = 25
    ax = get_ax(2, 2)
    pre_nms_anchors = utils.denorm_boxes(rpn["pre_nms_anchors"][0], image.shape[:2])
    refined_anchors = utils.denorm_boxes(rpn["refined_anchors"][0], image.shape[:2])
    refined_anchors_clipped = utils.denorm_boxes(rpn["refined_anchors_clipped"][0], image.shape[:2])
    visualize.draw_boxes(image, boxes=pre_nms_anchors[:limit],
                        refined_boxes=refined_anchors[:limit],
                        title="Top 50 anchors and refined anchors",
                        ax=ax[0, 0])
    visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[:limit], title="Top 50 refined anchors", ax=ax[0, 1])

    # Show refined anchors after non-max suppression
    limit = 25
    ixs = rpn["post_nms_anchor_ix"][:limit]
    visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[ixs], title="Top 50 refined anchors after nms", ax=ax[1, 0])

    # Show final proposals
    # These are the same as the previous step (refined anchors 
    # after NMS) but with coordinates normalized to [0, 1] range.
    limit = 25
    # Convert back to image coordinates for display
    h, w = config.IMAGE_SHAPE[:2]
    proposals = rpn['proposals'][0, :limit] * np.array([h, w, h, w])
    visualize.draw_boxes(image, refined_boxes=proposals, title="Top 50 proposals", ax=ax[1, 1])

    # Measure the RPN recall (percent of objects covered by anchors)
    # Here we measure recall for 3 different methods:
    # - All anchors
    # - All refined anchors
    # - Refined anchors after NMS
    iou_threshold = 0.7

    recall, positive_anchor_ids = utils.compute_recall(model.anchors, gt_bbox, iou_threshold)
    print("All Anchors ({:5})       Recall: {:.3f}  Positive anchors: {}".format(
        model.anchors.shape[0], recall, len(positive_anchor_ids)))

    recall, positive_anchor_ids = utils.compute_recall(rpn['refined_anchors'][0], gt_bbox, iou_threshold)
    print("Refined Anchors ({:5})   Recall: {:.3f}  Positive anchors: {}".format(
        rpn['refined_anchors'].shape[1], recall, len(positive_anchor_ids)))

    recall, positive_anchor_ids = utils.compute_recall(proposals, gt_bbox, iou_threshold)
    print("Post NMS Anchors ({:5})  Recall: {:.3f}  Positive anchors: {}".format(
        proposals.shape[0], recall, len(positive_anchor_ids)))

def show_rois_refinement(model, dataset, config, image):
    """Show rcnn predict
    Params:
        Class   model:   mrcnn model
        Class   dataset: 
        Class   config:  
        Ndarray image:
    """
    # Get input and output to classifier and mask heads.
    mrcnn = model.run_graph([image], [
        ("proposals", model.keras_model.get_layer("ROI").output),
        ("probs", model.keras_model.get_layer("mrcnn_class").output),
        ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
        ("masks", model.keras_model.get_layer("mrcnn_mask").output),
        ("detections", model.keras_model.get_layer("mrcnn_detection").output),
    ])


    # Get detection class IDs. Trim zero padding.
    det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
    det_count = np.where(det_class_ids == 0)[0][0]
    det_class_ids = det_class_ids[:det_count]
    detections = mrcnn['detections'][0, :det_count]

    print("{} detections: {}".format(
        det_count, np.array(dataset.class_names)[det_class_ids]))

    captions = ["{} {:.3f}".format(dataset.class_names[int(c)], s) if c > 0 else ""
                for c, s in zip(detections[:, 4], detections[:, 5])]
    visualize.draw_boxes(
        image, 
        refined_boxes=utils.denorm_boxes(detections[:, :4], image.shape[:2]),
        visibilities=[2] * len(detections),
        captions=captions, title="Detections",
        ax=get_ax())
    
    
    # Proposals的坐标是规范化的坐标. 将它们缩放到图像坐标.
    h, w = config.IMAGE_SHAPE[:2]
    proposals = np.around(mrcnn["proposals"][0] * np.array([h, w, h, w])).astype(np.int32)
    
    # 每个proposal的Class ID, score, and mask
    roi_class_ids = np.argmax(mrcnn["probs"][0], axis=1)
    roi_scores = mrcnn["probs"][0, np.arange(roi_class_ids.shape[0]), roi_class_ids]
    roi_class_names = np.array(dataset.class_names)[roi_class_ids]
    roi_positive_ixs = np.where(roi_class_ids > 0)[0]
    
    #有多少ROIs和空行?
    print("{} Valid proposals out of {}".format(np.sum(np.any(proposals, axis=1)), proposals.shape[0]))
    print("{} Positive ROIs".format(len(roi_positive_ixs)))
    
    # Class数量
    print(list(zip(*np.unique(roi_class_names, return_counts=True))))
    
    #显示一个随机样本的proposals.
    #分类为背景的Proposals是点，其他的显示它们的类名和置信分数.
    limit = 200
    ixs = np.random.randint(0, proposals.shape[0], limit)
    captions = ["{} {:.3f}".format(dataset.class_names[c], s) if c > 0 else ""
                for c, s in zip(roi_class_ids[ixs], roi_scores[ixs])]
    ax = get_ax(1, 2)
    visualize.draw_boxes(image, boxes=proposals[ixs],
                        visibilities=np.where(roi_class_ids[ixs] > 0, 2, 1),
                        captions=captions, title="ROIs Before Refinement",
                        ax=ax[0])

    #指定类别的bounding box偏移.
    roi_bbox_specific = mrcnn["deltas"][0, np.arange(proposals.shape[0]), roi_class_ids]
    modellib.log("roi_bbox_specific", roi_bbox_specific)
    
    #应用bounding box变换
    #形状: [N, (y1, x1, y2, x2)]
    refined_proposals = utils.apply_box_deltas(
        proposals, roi_bbox_specific * config.BBOX_STD_DEV).astype(np.int32)
    modellib.log("refined_proposals", refined_proposals)
    
    #显示positive proposals
    # ids = np.arange(roi_boxes.shape[0])  #显示所有
    limit = 5
    ids = np.random.randint(0, len(roi_positive_ixs), limit)  #随机显示样本
    captions = ["{} {:.3f}".format(dataset.class_names[c], s) if c > 0 else ""
                for c, s in zip(roi_class_ids[roi_positive_ixs][ids], roi_scores[roi_positive_ixs][ids])]
    visualize.draw_boxes(image, boxes=proposals[roi_positive_ixs][ids],
                        refined_boxes=refined_proposals[roi_positive_ixs][ids],
                        visibilities=np.where(roi_class_ids[roi_positive_ixs][ids] > 0, 1, 0),
                        captions=captions, title="ROIs After Refinement",
                        ax=ax[1])

    #去掉那些被分类为背景的boxes
    keep = np.where(roi_class_ids > 0)[0]
    print("Keep {} detections:\n{}".format(keep.shape[0], keep))
    
    #去掉低置信度的检测结果
    keep = np.intersect1d(keep, np.where(roi_scores >= model.config.DETECTION_MIN_CONFIDENCE)[0])
    print("Remove boxes below {} confidence. Keep {}:\n{}".format(
        config.DETECTION_MIN_CONFIDENCE, keep.shape[0], keep))
    #为每一个类别做NMS
    pre_nms_boxes = refined_proposals[keep]
    pre_nms_scores = roi_scores[keep]
    pre_nms_class_ids = roi_class_ids[keep]
    
    nms_keep = []
    for class_id in np.unique(pre_nms_class_ids):
        #选择该类的检测结果
        ixs = np.where(pre_nms_class_ids == class_id)[0]
        #做NMS
        class_keep = utils.non_max_suppression(pre_nms_boxes[ixs], 
                                                pre_nms_scores[ixs],
                                                config.DETECTION_NMS_THRESHOLD)
        #映射索引
        class_keep = keep[ixs[class_keep]]
        nms_keep = np.union1d(nms_keep, class_keep)
        print("{:22}: {} -> {}".format(dataset.class_names[class_id][:20], 
                                    keep[ixs], class_keep))
    
    keep = np.intersect1d(keep, nms_keep).astype(np.int32)
    print("\nKept after per-class NMS: {}\n{}".format(keep.shape[0], keep))
    
    #显示最终的检测结果
    ixs = np.arange(len(keep))  # Display all
    # ixs = np.random.randint(0, len(keep), 10)  # Display random sample
    captions = ["{} {:.3f}".format(dataset.class_names[c], s) if c > 0 else ""
                for c, s in zip(roi_class_ids[keep][ixs], roi_scores[keep][ixs])]
    visualize.draw_boxes(
        image, boxes=proposals[keep][ixs],
        refined_boxes=refined_proposals[keep][ixs],
        visibilities=np.where(roi_class_ids[keep][ixs] > 0, 1, 0),
        captions=captions, title="Detections after NMS",
        ax=get_ax())

def show_activate_layers(model, config, image):
    # TODO
    #获取一些示例层的activations
    activations = model.run_graph([image], [
        ("input_image",        model.keras_model.get_layer("input_image").output),
        ("res4w_out",          model.keras_model.get_layer("res4w_out").output),  # for resnet100
        ("rpn_bbox",           model.keras_model.get_layer("rpn_bbox").output),
        ("roi",                model.keras_model.get_layer("ROI").output),
    ])
    
    #输入图像 (规范化的)
    _ = plt.imshow(modellib.unmold_image(activations["input_image"][0],config))
    
    # Backbone feature map
    visualize.display_images(np.transpose(activations["res4w_out"][0,:,:,:4], [2, 0, 1]))



############################################################
#  COCO Evaluation
############################################################
def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Evaluate coco dataset results
    
    Params:
        Class-Child dataset:   inhere follow the Datset class
        List        image_ids: images id list
        List        rois:      [[xmin, ymin, xmax, ymax]]
        List        class_ids: list of labels
        List        scores:    evaluate confidence
        Array       masks:     masks ndarray 
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score    = scores[i]
            bbox     = np.around(rois[i], 1)
            mask     = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
                }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, config, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    Params:
        dataset:   A Dataset object with valiadtion data
        eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
        limit:     if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    APs = []
    for i, image_id in enumerate(image_ids):
        # Load image
        # image = dataset.load_image(image_id)
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        info = dataset.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                            dataset.image_reference(image_id)))
        
        # visualize.display_images(np.transpose(gt_mask, [2, 0, 1]), titles='gt_mask', cmap="Blues")
        gt_instance = visualize.display_instances(image       = image, 
                                    boxes       = gt_bbox, 
                                    masks       = gt_mask, 
                                    class_ids   = gt_class_id,
                                    class_names = dataset.class_names,
                                    title       = "Gt instance",
                                    auto_show   = False)
        # plt.imsave('../gt_instance/{}.jpg'.format(image_id), gt_instance)
        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)
        # Show instance
        visualize.display_instances(image       = image, 
                                    boxes       = r['rois'], 
                                    masks       = r['masks'], 
                                    class_ids   = r["class_ids"],
                                    class_names = dataset.class_names,
                                    scores      = r['scores'],
                                    title       = "Predict instance")

        # show rpn and rois procession
        # show_rpn_predict(model=model, image=image, gt_class_id=gt_class_id, gt_bbox=gt_bbox)
        # show_rois_refinement(model=model, dataset=dataset, config=config, image=image)
        # show_activate_layers(model=model, config=config, image=image)
        #画出precision-recall的曲线
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                r['rois'], r['class_ids'], r['scores'], r['masks'])
        # visualize.plot_precision_recall(AP, precisions, recalls)
        # # 显示confusion matrix
        # visualize.plot_overlaps(gt_class_id, r['class_ids'], r['scores'],
        #                         overlaps, dataset.class_names)
        
        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)
        APs.append(AP)

    print("mAP @ IoU=50: ", np.mean(APs))
    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("\nPrediction time: {:0.6}. Average waste {:0.6}s/image".format(t_prediction, t_prediction / len(image_ids)))
    print("Total time: {:0.6}".format(time.time() - t_start))



# *************************************************************************
# -------------------------------------------------------------------------
#                             Control
# -------------------------------------------------------------------------
# *************************************************************************
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train model on MS COCO.')
    parser.add_argument("command", 
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=False,
                        default="../Coco",
                        metavar="coco path",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model', required=False,
                        default="coco",
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    args = parser.parse_args()
    print("\n\n---------------------Step1: Command Line Parameters--------------------------")
    print("Command:    \t", args.command)
    print("Model:      \t", os.path.abspath(args.model))
    print("Dataset:    \t", os.path.abspath(args.dataset))
    print("Year:       \t", DEFAULT_DATASET_YEAR)
    print("Logs:       \t", DEFAULT_LOGS_DIR)

    
    
    print("\n\n-------------------------Step2: Config Parameters----------------------------")
    # Configurations
    if args.command == "train":
        config = CocoConfig()
    else:
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.2 # 0
            DETECTION_MAX_INSTANCES = 50
        config = InferenceConfig()
    config.display()

    
    print("\n\n-----------------------Step3: Load Graph and Weghts--------------------------")
    # Create model
    mode = "training" if args.command == "train" else "inference"
    model = modellib.MaskRCNN(mode=mode, config=config,
                              model_dir=DEFAULT_LOGS_DIR)
    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
        if not os.path.exists(model_path):
            utils.download_trained_weights(COCO_MODEL_PATH)
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    print("Model_path\n\t{}".format(model_path))
    exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask", "rpn_model"] if config.NAME != 'coco' and args.command == "train" else None
    model.load_weights(model_path, by_name=True, exclude=exclude)

    # 显示所有训练的weights 
    # print("Display Weight Statiscal Graph") 
    # visualize.display_weight_stats(model)
    
    print("\n\n----------------------Step4: Execute Train or Eval--------------------------")
    # Train or evaluate
    if args.command == "train":
        # Train dataset
        dataset_train = CocoDataset()
        dataset_train.load_coco(args.dataset, "train", year=DEFAULT_DATASET_YEAR, auto_download=AUTO_DOWNLOAD)
        if DEFAULT_DATASET_YEAR in '2014':
            dataset_train.load_coco(args.dataset, "valminusminival", year=DEFAULT_DATASET_YEAR, auto_download=AUTO_DOWNLOAD)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val" if DEFAULT_DATASET_YEAR in '2017' else "minival"
        dataset_val.load_coco(args.dataset, val_type, year=DEFAULT_DATASET_YEAR, auto_download=AUTO_DOWNLOAD)
        dataset_val.prepare()

        # Image Augmentation
        # pass

        # *** This training schedule is an example. Update to your needs ***


        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads',
                    augmentation=None)
                    

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        if config.BACKBONE == "mobilenet224v1":
            stage_2_layers = '11M+'
        else:
            stage_2_layers = '4+'

        print("Fine tune {} stage {} and up".format(config.BACKBONE, stage_2_layers))        
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers=stage_2_layers,
                    augmentation=None)
                    

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=300,
                    layers='all',
                    augmentation=None)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val" if DEFAULT_DATASET_YEAR in '2017' else "minival"
        coco = dataset_val.load_coco(args.dataset, val_type, year=DEFAULT_DATASET_YEAR, return_coco=True, auto_download=AUTO_DOWNLOAD)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, config, "bbox", limit=int(args.limit))

    else:
        print("'{}' is not recognized. Use 'train' or 'evaluate'".format(args.command))
