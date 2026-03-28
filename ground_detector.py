import numpy as np

from groundingdino.util.inference import Model


class GroundDetector:
    """GroundDetector：封裝 GroundingDINO 模型，提供物件邊界框與遮罩的偵測介面。
    主要用途：用於找出目標的 bbox 與對應的 mask，以及支援二次分析（如眼睛 bbox／分割等）。
    """
    def __init__(self, config):
        self._download_if_missing(config['dino_checkpoint_url'], config['dino_checkpoint'])
        self.model = Model(
            config['dino_config'],
            config['dino_checkpoint'],
            device=config.get('device', 'cpu')
        )
        self.box_threshold = config.get('dino_box_threshold', 0.35)
        self.text_threshold = config.get('dino_text_threshold', 0.25)
        self.prompt = config.get('dino_prompt', 'cat')
    
    def _download_if_missing(self, url, save_path):
        from pathlib import Path

        import requests
        path = Path(save_path)

        if path.is_file():
            print("model already exists")
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        print("downloading dino model from ", url)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()

            total = int(r.headers.get("content-length", 0))

            with open(path, "wb") as f:
                downloaded = 0

                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)

                    print(
                        f"\r{downloaded/1024/1024:.1f}MB / {total/1024/1024:.1f}MB",
                        end=""
                    )

        print("\nDownload complete")
    def detect(self, image, prompt=None):
        """對整張影像執行偵測，回傳 { 'boxes': [...], 'masks': [...] }。
        boxes: list of (x1, y1, x2, y2) in image coordinates
        masks: list of 二值遮罩，尺寸與 image 相同，1 表示偵測到的區域
        """
        if prompt is None:
            prompt = self.prompt
        detections, phrases = self.model.predict_with_caption(
            image=image,
            caption=prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )

        height, width = image.shape[:2]
        boxes_out = []

        boxes = detections.xyxy
        if hasattr(boxes, 'cpu'):
            boxes = boxes.cpu().numpy()
        else:
            boxes = np.asarray(boxes)
        for box in boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(width, x2); y2 = min(height, y2)
            boxes_out.append((x1, y1, x2, y2))
        
        return boxes_out
