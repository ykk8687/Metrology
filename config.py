"""
集中管理所有可自定義參數
"""

CONFIG = {
    # COCO資料集
    'coco_ann_file': './filtered_annotations.json',
    'coco_img_dir': './filtered_images',
    # SAM
    'sam_checkpoint': './weights/sam_vit_h_4b8939.pth',
    'sam_model_type': 'vit_h',
    'sam_checkpoint_url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    # GroundingDINO
    'dino_config': './GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
    'dino_checkpoint': './weights/groundingdino_swint_ogc.pth',
    'dino_checkpoint_url': 'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth',
    'dino_box_threshold': 0.35,
    'dino_text_threshold': 0.25,
    'dino_target_1': 'cat',
    'dino_target_2': 'eyes',

    # 通用
    'device': 'cuda',
    # 0~16
    'choosen_id': 10,
    # 結果
    'results_dir': './results_images',
    # mask顏色
    'mask_colors': [
        (255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)
    ]
}
