import sys
sys.path.append("./YCBVTrainValTest/YCBVideoTest")
import os.path as osp
import torch
import numpy as np
from gorilla.evaluation import DatasetEvaluator, DatasetEvaluators
from ycb_video_sem_eval import (evaluate)


class YCBVideoSemanticEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """
    def __init__(self, logger=None):
        """
        Args:
            num_classes, ignore_label: deprecated argument
        """
        self.logger = logger
        self.reset()

    def reset(self):
        self._predictions = {}
        self._gt = {}

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model. It is a list of dicts.
            outputs: the outputs of a model. It is either list of semantic
                     segmentation predictions or list of dicts with key
                     "sem_seg" that contains semantic segmentation prediction
                     in the same format.
        """
        for input, output in zip(inputs, outputs):
            scene_name = input["scene_name"]  # todo modify it
            semantic_gt = input["labels"].cpu().numpy()
            semantic_pred = output["semantic_pred"].cpu().numpy()
            self._gt[scene_name] = semantic_gt
            self._predictions[scene_name] = semantic_pred

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):
        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        matches = {}
        for scene_name in self._gt.keys():
            matches[scene_name] = {}
            matches[scene_name]["semantic_gt"] = self._gt[scene_name]
            matches[scene_name]["semantic_pred"] = self._predictions[scene_name]

        evaluate(matches, self.logger)


# TODO: add YCB-Video Instance Evaluator
YCBVideoEvaluator = DatasetEvaluators([YCBVideoSemanticEvaluator])
