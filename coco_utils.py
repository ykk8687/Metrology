from pycocotools import coco
from pycocotools.coco import COCO
import os
import cv2

class CocoImageSelector:
    """
    CocoImageSelector類別，負責COCO圖片選擇與自動下載。
    用途：自動檢查本地有無COCO圖片與標註，若無則自動下載，並可篩選出有兩隻貓或兩隻動物的圖片。
    輸入：config(dict)
    輸出：select_one_image()回傳一張合格圖片與其bbox。
    """
    def __init__(self, config):
        """
        建構子，初始化COCO路徑與自動檢查下載。
        config: dict，包含路徑等
        """
        self.ann_file = config['coco_ann_file']
        self.img_dir = config['coco_img_dir']
        self.config = config
        self._ensure_data_ready()
        self.coco = COCO(self.ann_file)

    def _ensure_data_ready(self):
        """
        檢查本地有無COCO圖片與標註，若無則自動下載。
        無參數，無回傳。
        """
        if not (os.path.exists(self.ann_file) and os.path.isdir(self.img_dir) and len(os.listdir(self.img_dir)) > 0):
            print("未偵測到本地圖片與標註，將自動從COCO下載符合條件的圖片與標註...")
            import requests
            from tqdm import tqdm
            ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            ann_zip = "annotations_trainval2017.zip"
            if not os.path.exists(ann_zip):
                print("下載COCO標註...（檔案較大，請耐心等候）")
                with requests.get(ann_url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))
                    with open(ann_zip, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc=ann_zip) as pbar:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            import zipfile
            with zipfile.ZipFile(ann_zip, 'r') as zip_ref:
                zip_ref.extractall("./annotations_tmp")
            ann_path = "./annotations_tmp/annotations/instances_val2017.json"
            coco = COCO(ann_path)
            img_ids = self.get_multi_cat_images(coco)
            if len(img_ids) == 0:
                img_ids = self.get_multi_animal_images(coco)
            print("篩選到 {} 張圖片，開始下載...".format(len(img_ids)))
            if len(img_ids) == 0:
                raise RuntimeError("找不到符合條件的圖片")
            self._download_with_tqdm(coco, img_ids, self.img_dir, self.ann_file)
            print("下載完成！")
        else:
            print("偵測到本地已存在圖片與標註，直接載入...")

    def _download_with_tqdm(self, coco, img_ids, save_dir, ann_save_path=None):
        """
        下載圖片並顯示進度條。
        coco: COCO物件
        img_ids: list
        save_dir: str
        ann_save_path: str
        return: None
        """
        os.makedirs(save_dir, exist_ok=True)
        imgs = coco.loadImgs(img_ids)
        from tqdm import tqdm
        for img in tqdm(imgs, desc="下載圖片", unit="img"):
            url = img['coco_url']
            fname = os.path.join(save_dir, img['file_name'])
            if not os.path.exists(fname):
                import requests
                r = requests.get(url, timeout=30)
                with open(fname, 'wb') as f:
                    f.write(r.content)
        
        if ann_save_path:
            self._download_coco_images_and_anns(coco, img_ids, ann_save_path)
    
    def _download_coco_images_and_anns(self, coco, img_ids, ann_save_path):
        """
        根據 img_ids 過濾 images/annotations，並寫出 filtered_annotations.json。
        coco: COCO 物件
        img_ids: list of int，欲保留的圖片 id
        save_dir: str，圖片資料夾（未用到，保留參數）
        ann_save_path: str，輸出 json 路徑
        """
        # 過濾 images
        images = [img for img in coco.dataset['images'] if img['id'] in img_ids]
        # 過濾 annotations
        annotations = [ann for ann in coco.dataset['annotations'] if ann['image_id'] in img_ids]
        # categories 全部保留
        categories = coco.dataset['categories']
        filtered = {
            'images': images,
            'annotations': annotations,
            'categories': categories
        }
        import json
        with open(ann_save_path, 'w', encoding='utf-8') as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)

    def get_multi_cat_images(self, coco=None):
        """
        篩選出有兩隻貓以上的圖片。
        coco: COCO物件，若為None則用self.coco
        return: list
        """
        if coco is None:
            coco = self.coco
        cat_id = coco.getCatIds(catNms=['cat'])
        img_ids = coco.getImgIds(catIds=cat_id)
        result = []
        for img_id in img_ids:
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_id, iscrowd=None)
            if len(ann_ids) >= 2:
                result.append(img_id)
        return result

    def get_multi_animal_images(self, coco=None):
        """
        篩選出有兩隻動物以上的圖片。
        coco: COCO物件，若為None則用self.coco
        return: list
        """
        if coco is None:
            coco = self.coco
        animal_ids = list(range(15, 26))
        img_ids = coco.getImgIds(catIds=animal_ids)
        result = []
        for img_id in img_ids:
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=animal_ids, iscrowd=None)
            if len(ann_ids) >= 2:
                result.append(img_id)
        return result

    def select_one_image(self, choosen_id=0):
        """
        選出一張合格圖片，回傳img_id, image, animal_bboxes, img_path。
        choosen_id: int，選擇的圖片索引，預設為0
        return: (int, numpy.ndarray, list, str)
        """
        img_ids = self.get_multi_cat_images()
        if len(img_ids) == 0:
            img_ids = self.get_multi_animal_images()
        if len(img_ids) == 0:
            raise RuntimeError('找不到符合條件的圖片')
        img_id = img_ids[choosen_id]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        animal_ids = list(range(15, 26))
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=animal_ids, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        animal_bboxes = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            animal_bboxes.append((int(x), int(y), int(x+w), int(y+h)))
        return img_id, image, animal_bboxes, img_path