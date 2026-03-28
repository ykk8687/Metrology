import cv2
import numpy as np

class MeasureUtils:
    """
    影像評估與量測工具類別，提供 Dice、Boundary IoU、眼睛距離等靜態方法。
    """
    @staticmethod
    def calc_eye_distance(eye_boxes):
        """計算單個遮罩內兩眼中心點的歐氏距離
        eye_boxes: list of two boxes (x1, y1, x2, y2)，須恰好兩個眼睛框
        return: float, 兩眼中心點距離；若條件不符合，回傳 -1
        """
        if len(eye_boxes) != 2:
            return -1
        c1 = eye_boxes[0]
        c2 = eye_boxes[1]
        return np.linalg.norm(np.array(c1)-np.array(c2))

    @staticmethod
    def dice_score(mask_pred, mask_gt):
        """計算二值遮罩的 Dice 指標
        mask_pred, mask_gt: 二值 numpy 陣列，值為 0 或非0
        return: float in [0,1]
        """
        pred = (mask_pred > 0).astype(np.uint8)
        gt = (mask_gt > 0).astype(np.uint8)
        inter = np.logical_and(pred, gt).sum()
        union = pred.sum() + gt.sum()
        if union == 0:
            return 1.0 if inter == 0 else 0.0
        return 2.0 * inter / float(union)

    @staticmethod
    def mask_to_boundary(mask, dilation_ratio=0.02):
        """
        Convert binary mask to boundary mask.
        :param mask (numpy array, uint8): binary mask
        :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
        :return: boundary mask (numpy array)
        """
        h, w = mask.shape
        img_diag = np.sqrt(h ** 2 + w ** 2)
        dilation = int(round(dilation_ratio * img_diag))
        if dilation < 1:
            dilation = 1
        # Pad image so mask truncated by the image border is also considered as boundary.
        new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        kernel = np.ones((3, 3), dtype=np.uint8)
        new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
        mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
        # G_d intersects G in the paper.
        return mask - mask_erode

    @staticmethod
    def boundary_iou(gt, dt, dilation_ratio=0.02):
        """
        Compute boundary iou between two binary masks.
        :param gt (numpy array, uint8): binary mask
        :param dt (numpy array, uint8): binary mask
        :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
        :return: boundary iou (float)
        """
        gt_boundary = MeasureUtils.mask_to_boundary(gt, dilation_ratio)
        dt_boundary = MeasureUtils.mask_to_boundary(dt, dilation_ratio)
        intersection = ((gt_boundary * dt_boundary) > 0).sum()
        union = ((gt_boundary + dt_boundary) > 0).sum()
        boundary_iou = intersection / union
        return boundary_iou
