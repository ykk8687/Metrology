import numpy as np
from segment_anything import SamPredictor, sam_model_registry


class GenericSegmenter:
    """GenericSegmenter：基於 SAM 的通用分割介面
    用途：初始化 SAM 模型，segment(image, object_bboxes) 回傳 list of (mask, bbox)
    輸入：config(dict)、image: numpy.ndarray、object_bboxes: list of (x1, y1, x2, y2)
    輸出：list of (mask, bbox)，mask 為該 bbox 的分割遮罩
    """
    def __init__(self, config):
        """
        建構子，初始化SAM模型。
        config: dict，包含模型型號、checkpoint、device等
        """
        self._download_if_missing(config['sam_checkpoint_url'], config['sam_checkpoint'])
        sam = sam_model_registry[config['sam_model_type']](checkpoint=config['sam_checkpoint'])
        sam.to(config['device'])
        self.predictor = SamPredictor(sam)
        self.device = config['device']

    def _download_if_missing(self, url, save_path):
        from pathlib import Path

        import requests
        path = Path(save_path)

        if path.is_file():
            print("model already exists")
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        print("downloading sam model from ", url)
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

    def segment(self, image, object_bboxes):
        """
        分割多個物件bbox。
        image: numpy.ndarray，輸入影像
        object_bboxes: list of (x1, y1, x2, y2)
        return: list of (mask, bbox)
        """
        self.predictor.set_image(image)
        masks = []
        for bbox in object_bboxes:
            input_box = np.array(bbox)
            mask, _, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False
            )
            masks.append((mask[0], bbox))
        
        return masks[:len(object_bboxes)]