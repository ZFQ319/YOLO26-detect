import os
import numpy as np
from shapely.geometry import Polygon

pred_dir = "E:/ZFQ_Projects/Python_Projects/YOLO26/runs/obb/val2/labels"
gt_dir = "E:/ZFQ_Projects/Python_Projects/Datasets_detect/SSDD/obb/labels/test/"

small_iou = []
medium_iou = []
large_iou = []


def polygon_iou(poly1, poly2):
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area

    if union == 0:
        return 0

    return inter / union


for file in os.listdir(gt_dir):

    gt_path = os.path.join(gt_dir, file)
    pred_path = os.path.join(pred_dir, file)

    if not os.path.exists(pred_path):
        continue

    gt_boxes = np.loadtxt(gt_path).reshape(-1, 9)
    pred_boxes = np.loadtxt(pred_path).reshape(-1, 9)

    for gt in gt_boxes:

        cls, x1, y1, x2, y2, x3, y3, x4, y4 = gt

        gt_poly = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

        gt_area = gt_poly.area*640*640

        best_iou = 0

        for pred in pred_boxes:

            cls, px1, py1, px2, py2, px3, py3, px4, py4 = pred

            pred_poly = Polygon([(px1, py1), (px2, py2), (px3, py3), (px4, py4)])

            iou = polygon_iou(gt_poly, pred_poly)

            if iou > best_iou:
                best_iou = iou

        if gt_area < 16 * 16:
            small_iou.append(best_iou)

        elif gt_area < 32 * 32:
            medium_iou.append(best_iou)

        else:
            large_iou.append(best_iou)

print("Small IoU:", np.mean(small_iou))
print("Medium IoU:", np.mean(medium_iou))
print("Large IoU:", np.mean(large_iou))
