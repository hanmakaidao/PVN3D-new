import os
import sys
import inspect
import numpy as np
import logging

try:
    from itertools import izip
except ImportError:
    izip = zip

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

CLASS_LABELS = [
    "background", "002_master_chef_can", "003_cracker_box", "004_sugar_box",
    "005_tomato_soup_can", "006_mustard_bottle", "007_tuna_fish_can",
    "008_pudding_box", "009_gelatin_box", "010_potted_meat_can", "011_banana",
    "019_pitcher_base", "021_bleach_cleanser", "024_bowl", "025_mug",
    "035_power_drill", "036_wood_block", "037_scissors", "040_large_marker",
    "051_large_clamp", "052_extra_large_clamp", "061_foam_brick",
]
VALID_CLASS_IDS = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
UNKNOWN_ID = np.max(VALID_CLASS_IDS) + 1


def get_iou(label_id, confusion):
    if label_id not in VALID_CLASS_IDS:
        return float("nan")
    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false negatives
    fn = np.longlong(confusion[label_id, :].sum()) - tp
    # false positives
    not_ignored = [i for i in VALID_CLASS_IDS if not i == label_id]
    fp = np.longlong(confusion[not_ignored, label_id].sum())

    denom = (tp + fp + fn)
    if denom == 0:
        return float("nan")
    return (float(tp) / denom, tp, denom)


# generate confusion matrix for mIoU
def evaluate_scan(data, confusion):
    pred_ids = data["semantic_pred"]
    gt_ids = data["semantic_gt"]
    # sanity checks
    if not pred_ids.shape == gt_ids.shape:
        message = "{}: number of predicted values does not match number of vertices".format(
            pred_ids.shape)
        sys.stderr.write("ERROR: " + str(message) + "\n")
        sys.exit(2)

    np.add.at(confusion, (gt_ids, pred_ids), 1)


def evaluate(matches, logger=None):
    if logger is not None:
        assert isinstance(logger, logging.Logger)
    max_id = UNKNOWN_ID  # 22

    # formulate confusion matrix
    confusion = np.zeros((max_id + 1, max_id + 1), dtype=np.ulonglong)

    def info(message):
        if logger is not None:
            logger.info(message)
        else:
            print(message)

    message = "evaluating {} scans...".format(len(matches))
    info(message)

    # add all test data to confusion matrix
    for i, (scene, data) in enumerate(matches.items()):
        evaluate_scan(data, confusion)
        sys.stdout.write("\rscans processed: {}".format(i + 1))
        sys.stdout.flush()
    info("")

    class_ious = {}

    # get IoU of each class
    for i in range(len(VALID_CLASS_IDS)):
        label_name = CLASS_LABELS[i]
        label_id = VALID_CLASS_IDS[i]
        class_ious[label_name] = get_iou(label_id, confusion)

    # print results
    info("classes          IoU")
    info("----------------------------")
    mean_iou = 0
    for i in range(1, len(VALID_CLASS_IDS)):
        label_name = CLASS_LABELS[i]
        info("{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})".format(
            label_name, class_ious[label_name][0], class_ious[label_name][1],
            class_ious[label_name][2]))
        mean_iou += class_ious[label_name][0]
    mean_iou = mean_iou / len(VALID_CLASS_IDS)
    info("mean: {:>5.3f}".format(mean_iou))
