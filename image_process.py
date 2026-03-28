import os
import cv2
import numpy as np

class ImageProcessor:
    """
    ImageProcessor類別：負責分割結果的可視化、距離計算與結果儲存。
    用途：overlay_mask 疊加遮罩，draw_eye_boxes_and_lines 畫眼睛框與中心，visualize_and_save 計算距離與儲存。
    輸入：mask_colors(list)、results_dir(str)
    輸出：visualize_and_save無回傳
    """
    def __init__(self, mask_colors, results_dir, debug_masks: bool = False):
        """初始化遮罩與輸出路徑
        mask_colors: list of tuple, 遮罩顏色
        results_dir: str, 結果輸出資料夾
        debug_masks: bool, 是否輸出 debug 遮罩到 results_images/debug_masks
        """
        self.mask_colors = mask_colors
        self.results_dir = results_dir
        self.debug_masks = debug_masks
        if self.debug_masks:
            self._debug_dir = os.path.join(self.results_dir, 'debug_masks')
            os.makedirs(self._debug_dir, exist_ok=True)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        # ensure debug dir path exists when debugging is enabled
        if self.debug_masks:
            self._debug_dir = getattr(self, "_debug_dir", os.path.join(self.results_dir, 'debug_masks'))
            os.makedirs(self._debug_dir, exist_ok=True)

    def save_mask(self, mask, path: str):
        if mask is None:
            return
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        import cv2
        cv2.imwrite(path, (mask > 0).astype(np.uint8) * 255)
        # Ensure _debug_dir exists when debug mode is on
        if self.debug_masks:
            os.makedirs(self._debug_dir, exist_ok=True)

    def save_mask(self, mask, path: str):
        """Save a binary mask to the given path as a PNG (0/255)."""
        if mask is None:
            return
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        try:
            cv2.imwrite(path, (mask > 0).astype(np.uint8) * 255)
        except Exception:
            pass

    def overlay_mask(self, image, mask, color, alpha=0.5):
        """在影像上疊加半透明遮罩
        image: numpy.ndarray, 原圖
        mask: numpy.ndarray, 二值遮罩 (0/1 或 0/255)
        color: tuple, 遮罩顏色
        alpha: float, 透明度
        return: numpy.ndarray, 疊加後影像
        """
        overlay = image.copy()
        overlay[mask > 0] = (np.array(color) * alpha + overlay[mask > 0] * (1 - alpha)).astype(np.uint8)
        return overlay


    def visualize_and_save(self, image, masks_and_rois, all_eye_masks, img_path, gt_masks=None):
        """
        計算距離、畫線、標註、儲存結果
        image: numpy.ndarray
        masks_and_rois: list of (mask, bbox)
        all_eye_boxes: list of list of (x1, y1, x2, y2)
        img_path: str，原圖路徑
        gt_masks: optional list[numpy.ndarray], per-annotation ground-truth masks
        return: None
        """
        vis_img = image.copy()
        # 疊加動物遮罩與眼睛遮罩，並取得每隻貓的眼睛中心點
        all_eye_centers = []
        # 疊加動物遮罩
        for idx, (mask, bbox) in enumerate(masks_and_rois):
            vis_img = self.overlay_mask(vis_img, mask, self.mask_colors[idx%len(self.mask_colors)], alpha=0.4)
        # 疊加眼睛遮罩並計算中心
        for idx, (mask, bbox) in enumerate(masks_and_rois):
            eye_centers = []
            if all_eye_masks is not None and idx < len(all_eye_masks):
                for eye_mask in all_eye_masks[idx]:
                    if eye_mask is not None and np.any(eye_mask > 0):
                        vis_img = self.overlay_mask(vis_img, eye_mask, (0,0,255), alpha=0.4)
                        # 以 connectedComponentsWithStats 找出質心
                        mask_uint8 = (eye_mask > 0).astype(np.uint8)
                        _, _, _, centroids = cv2.connectedComponentsWithStats(mask_uint8)
                        if centroids.shape[0] > 1:
                            last = centroids[-1]
                            eye_centers.append( (int(last[0]), int(last[1])) )
            all_eye_centers.append(eye_centers)
        # Save per-eye masks for debugging (eye masks come from main_pipeline via all_eye_masks)
        if self.debug_masks and all_eye_masks is not None:
            base = os.path.splitext(os.path.basename(img_path))[0]
            for a_idx, eye_list in enumerate(all_eye_masks):
                for e_idx, eye_mask in enumerate(eye_list):
                    if eye_mask is not None:
                        dbg_path = os.path.join(self._debug_dir, f"{base}_animal{a_idx}_eye{e_idx}.png")
                        self.save_mask(eye_mask, dbg_path)
        # (1) 每隻動物雙眼距離，畫線與標註，並輸出距離資訊
        from measure import MeasureUtils
        for idx, centers in enumerate(all_eye_centers):
            if len(centers) == 2:
                d = MeasureUtils.calc_eye_distance(centers)
                cv2.line(vis_img, centers[0], centers[1], (0,255,255), 1)
                mid = (int((centers[0][0]+centers[1][0])/2), int((centers[0][1]+centers[1][1])/2))
                cv2.putText(vis_img, "{:.1f}px".format(d), mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                print(f"[DEBUG] Animal {idx+1} intra-eye distance: {d:.1f} px (centers: {centers})")
            else:
                print(f"[DEBUG] Intra-eye distance skipped for animal {idx+1} due to center_count={len(centers)}; centers={centers}")

        # (2) 不同動物右眼距離，畫線與標註
        right_eye_centers = []
        for centers in all_eye_centers:
            if len(centers) >= 2:
                # 右眼定義為 x 值較小的那一個
                right = min(centers, key=lambda p: int(p[0]))
                right_eye_centers.append(right)
        
        if len(right_eye_centers) == 2:
            cv2.line(vis_img, right_eye_centers[0], right_eye_centers[1], (255,0,255), 1)
            d = MeasureUtils.calc_eye_distance(right_eye_centers)
            mid = (int((right_eye_centers[0][0]+right_eye_centers[1][0])/2), int((right_eye_centers[0][1]+right_eye_centers[1][1])/2))
            cv2.putText(vis_img, "{:.1f}px".format(d), mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
            print(f"[DEBUG] Right-eye distance between animals: {d:.1f} px")
        out_path = os.path.join(self.results_dir, os.path.splitext(os.path.basename(img_path))[0] + "_result.jpg")
        # Debug：同時儲存每個動物的原始遮罩內容，便於檢查遮罩是否有內容
        if self.debug_masks:
            base = os.path.splitext(os.path.basename(img_path))[0]
            for i, (m, bbox) in enumerate(masks_and_rois):
                dbg_path = os.path.join(self._debug_dir, f"{base}_animal{i}_mask.png")
                cv2.imwrite(dbg_path, (m > 0).astype(np.uint8) * 255)
        cv2.imwrite(out_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        print("結果已儲存到 {}".format(out_path))
        # Save GT masks if provided
        if self.debug_masks and gt_masks is not None:
            base = os.path.splitext(os.path.basename(img_path))[0]
            for ann_idx, gt_mask in enumerate(gt_masks):
                if gt_mask is None:
                    continue
                gt_path = os.path.join(self._debug_dir, f"{base}_gt_ann{ann_idx}.png")
                self.save_mask(gt_mask, gt_path)
