import numpy as np

from groundingdino.util.inference import Model


class EyeDetector:
    """EyeDetector：基於 GroundingDINO 的眼睛偵測介面
    用途：初始化時載入模型，detect(image, prompt=None) 可於 ROI 圖像上偵測眼睛，並回傳眼睛的 bbox (x1, y1, x2, y2)。
    輸入：config(dct) 包含 dino_config、dino_weights、device、dino_prompt 等參數
    輸出：list[tuple] 形式的 (x1, y1, x2, y2)，或依實作回傳的遮罩格式
    """
    def __init__(self, config):
        """
        建構子，初始化GroundingDINO模型。
        config: dict，包含模型路徑、權重、device、閾值、prompt等
        """
        self.model = Model(
            config['dino_config'],
            config['dino_weights'],
            device=config['device']
        )
        self.box_threshold = config.get('dino_box_threshold', 0.35)
        self.text_threshold = config.get('dino_text_threshold', 0.25)
        self.prompt = config.get('dino_prompt', 'cat eye')

    def detect(self, roi_img, prompt=None):
        """
        偵測ROI中的眼睛位置。
        roi_img: numpy.ndarray，ROI影像
        prompt: str，偵測提示詞，預設為config設定
        return: list of (x1, y1, x2, y2)
        """
        if prompt is None:
            prompt = self.prompt
        detections, phrases = self.model.predict_with_caption(
            image=roi_img,
            caption=prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )
        print("'masks' in detections:", 'masks' in detections)
        # 將輸出統一為 mask list（以 ROI 尺寸為單位）以便主流程映射回全影像
        masks_out = []
        # 兼容不同型別的 detections，優先處理 Detections.xyxy 的情況
        if hasattr(detections, 'xyxy'):
            boxes = detections.xyxy
            if hasattr(boxes, 'cpu'):
                boxes = boxes.cpu().numpy()
            else:
                boxes = np.asarray(boxes)
            h, w = roi_img.shape[:2]
            masks_out = []
            for box in boxes:
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
                m = np.zeros((h, w), dtype=np.uint8)
                m[y1:y2, x1:x2] = 1
                masks_out.append(m)
            return masks_out
        if isinstance(detections, dict) and 'masks' in detections:
            for m in detections['masks']:
                m = np.asarray(m)
                if m.ndim == 2:
                    masks_out.append(m)
        elif isinstance(detections, list):
            # 若為 list 形式輸出，過濾掉 bbox 並收集 mask
            for item in detections:
                # 忽略 bbox 格式的輸出
                if isinstance(item, (list, tuple)) and len(item) == 4:
                    continue
                arr = np.asarray(item)
                if arr.ndim == 2:
                    masks_out.append(arr)
        # Debug: show what masks were detected (for troubleshooting)
        try:
            shapes = [getattr(m, 'shape', None) for m in masks_out]
            print(f"[EyeDetector] Detected {len(masks_out)} eye mask(s); shapes: {shapes}")
        except Exception:
            pass
        return masks_out
