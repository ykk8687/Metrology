import cv2
import numpy as np
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from coco_utils import CocoImageSelector
from config import CONFIG
from ground_detector import GroundDetector
from image_process import ImageProcessor
from measure import MeasureUtils
from sam_segment import GenericSegmenter

def find_eye_masks_for_animal(masks_and_rois, eye_boxes, segmenter, image):
    """
    對每個動物的 bbox，找出與之相交的眼睛 bbox，並用 SAM 分割出眼睛遮罩。
    masks_and_rois: list of (mask, bbox)，動物的遮罩與對應的 bbox
    eye_boxes: list of (x1, y1, x2, y2)，眼睛的 bbox
    segmenter: GenericSegmenter，已初始化的 SAM 分割器
    image: numpy.ndarray，輸入影像
    return: list of list of eye masks，對應每個動物的眼睛遮罩清單
    """
    all_eye_masks = []
    for mask, bbox in masks_and_rois:
        x1, y1, x2, y2 = bbox
        # 找出與此動物 bbox 相交的眼睛 bbox
        inter_eyes = []
        for ex1, ey1, ex2, ey2 in eye_boxes:
            if max(x1, ex1) < min(x2, ex2) and max(y1, ey1) < min(y2, ey2):
                inter_eyes.append((ex1, ey1, ex2, ey2))
        # 以 segmenter 對相交的眼睛 bbox 逐一分割出眼睛輪廓遮罩
        eye_masks_for_this = []
        
        if inter_eyes:
            eye_masks_pairs = segmenter.segment(image, inter_eyes)
            for eye_mask, eye_box in eye_masks_pairs:
                ex1, ey1, ex2, ey2 = eye_box
                m_full = np.zeros(image.shape[:2], dtype=np.uint8)
                try:
                    sub = eye_mask[ey1:ey2, ex1:ex2]
                    if sub.shape == (ey2-ey1, ex2-ex1):
                        m_full[ey1:ey2, ex1:ex2] = (sub > 0)
                    else:
                        m_full[ey1:ey2, ex1:ex2] = (eye_mask > 0)
                except Exception:
                    m_full[ey1:ey2, ex1:ex2] = (eye_mask > 0)
                eye_masks_for_this.append(m_full)
        
        # 限制每個動物只有兩個眼睛遮罩，超過部分以面積排序取前兩個
        if len(eye_masks_for_this) > 2:
            areas = [int(np.count_nonzero(m)) for m in eye_masks_for_this]
            idxs = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)[:2]
            eye_masks_for_this = [eye_masks_for_this[i] for i in idxs]
        all_eye_masks.append(eye_masks_for_this)

    return all_eye_masks

def prepare_gt_mask(img_id, image):
    """
    從 COCO annotation 建立 per-annotation ground-truth masks 的清單
    img_id: int，COCO 圖片 ID
    image: numpy.ndarray，輸入影像（用於取得尺寸）
    return: list of numpy.ndarray，對應每個 annotation 的二值遮罩
    """
    ann_masks = []
    coco = COCO(CONFIG['coco_ann_file'])
    ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    height, width = image.shape[:2]
    if anns is not None:
        for ann in anns:
            seg = ann.get('segmentation')
            mask_ann = np.zeros((height, width), dtype=np.uint8)
            if seg is None:
                ann_masks.append(mask_ann)
                continue
            if isinstance(seg, list):
                for poly in seg:
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                    if pts.size > 0:
                        cv2.fillPoly(mask_ann, [pts], 1)
            ann_masks.append(mask_ann)
    return ann_masks

def main():
    """主流程：從 COCO 下載/載入影像，使用 GroundingDINO 找出貓的 bbox、再用 SAM 對 bbox 分割輪廓，結合眼睛偵測資源，並輸出結果與評估指標。

    會產出：
    - pred masks、eye masks 的視覺化與儲存
    - per-annotation Dice 與 boundary IoU 的評估結果（若 boundary_iou 可用）
    - GT 掩模與各遮罩的 debug 圖檔
    """
    # 1. 篩選COCO圖片
    selector = CocoImageSelector(CONFIG)
    img_id, image, animal_bboxes, img_path = selector.select_one_image(choosen_id=CONFIG['choosen_id'])

    # 2. 使用 GroundingDINO 找出貓的 bbox，並用通用分割器做逐物件分割
    ground = GroundDetector(CONFIG)
    cat_boxes = ground.detect(image, prompt=CONFIG['dino_target_1'])
    segmenter = GenericSegmenter(CONFIG)
    masks_and_rois = segmenter.segment(image, cat_boxes)

    # 3. GroundingDINO 找出眼睛 bbox，並改用 SAM 對每個眼睛 bbox 分割成輪廓遮罩
    eye_boxes = ground.detect(image, prompt=CONFIG['dino_target_2'])

    all_eye_masks = find_eye_masks_for_animal(masks_and_rois, eye_boxes, segmenter, image)
    
    # 4. Per-annotation Dice 計算與 Ground-truth Mask 的進一步輸出
    # 4.1 建立 annotation, ground-truth mask
    ann_masks = prepare_gt_mask(img_id, image)
    
    # 4.2 計算 annotation 與 mask 的最佳匹配 Dice 與 Boundary IoU
    per_ann_dice = []
    per_ann_boundary = []
    if ann_masks:
        for gt_mask in ann_masks:
            best = 0.0
            best_biou = 0.0
            for pred_mask, _ in masks_and_rois:
                try:
                    d = MeasureUtils.dice_score(pred_mask, gt_mask)
                    biou = MeasureUtils.boundary_iou(gt_mask, pred_mask.astype(np.uint8))

                    if d > best:
                        best = d
                    if biou > best_biou:
                        best_biou = biou

                except Exception:
                    continue
            per_ann_dice.append(best)
            per_ann_boundary.append(best_biou)

        for i, val in enumerate(per_ann_dice, start=1):
            print(f"[Dice-Ann] annotation {i}: {val:.4f}")
        for i, val in enumerate(per_ann_boundary, start=1):
            print(f"[BoundaryAnn] annotation {i}: {val:.4f}")

    # 5. 結果可視化與儲存
    processor = ImageProcessor(CONFIG['mask_colors'], CONFIG['results_dir'], debug_masks=True)
    processor.visualize_and_save(image, masks_and_rois, all_eye_masks, img_path, gt_masks=ann_masks if 'ann_masks' in locals() else None)

if __name__ == "__main__":
    main()
