# -*- coding: utf-8 -*-
import os
import json
import shutil
import cv2
import xml.etree.ElementTree as ET

SRC_IMG_DIR = '/home/jwt/CenterNet/data/forensic/images/train2024'
SRC_XML_DIR = '/home/jwt/CenterNet/data/forensic/annotations/xmls'

OUT_ROOT = '/home/jwt/CenterNet/data/forensic_liquid'
OUT_TRAIN_IMG = os.path.join(OUT_ROOT, 'images', 'train2024')
OUT_VAL_IMG = os.path.join(OUT_ROOT, 'images', 'val2024')
OUT_ANN_DIR = os.path.join(OUT_ROOT, 'annotations')

TARGET_CLASS = 'liquid_drawing'
CATEGORY = {'id': 1, 'name': 'liquid_drawing'}

def find_image(filename, xml_stem):
    candidates = []
    if filename:
        candidates.append(filename)

    for ext in ['jpg', 'jpeg', 'png', 'webp']:
        candidates.append(xml_stem + '.' + ext)

    for name in candidates:
        p = os.path.join(SRC_IMG_DIR, name)
        if os.path.exists(p):
            return name, p

    return None, None

def parse_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    xml_stem = os.path.splitext(os.path.basename(xml_path))[0]

    filename_node = root.find('filename')
    filename = filename_node.text.strip() if filename_node is not None and filename_node.text else ''

    img_name, img_path = find_image(filename, xml_stem)
    if img_path is None:
        return None

    img = cv2.imread(img_path)
    if img is None:
        print('[跳过] 图片无法读取:', img_path)
        return None

    h, w = img.shape[:2]
    anns = []

    for obj in root.findall('object'):
        name_node = obj.find('name')
        if name_node is None or not name_node.text:
            continue

        cls_name = name_node.text.strip()
        if cls_name != TARGET_CLASS:
            continue

        bnd = obj.find('bndbox')
        if bnd is None:
            continue

        xmin = float(bnd.find('xmin').text)
        ymin = float(bnd.find('ymin').text)
        xmax = float(bnd.find('xmax').text)
        ymax = float(bnd.find('ymax').text)

        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(0, min(xmax, w - 1))
        ymax = max(0, min(ymax, h - 1))

        bw = xmax - xmin
        bh = ymax - ymin

        if bw <= 1 or bh <= 1:
            continue

        anns.append({
            'category_id': 1,
            'bbox': [xmin, ymin, bw, bh],
            'area': bw * bh,
            'iscrowd': 0
        })

    if len(anns) == 0:
        return None

    return {
        'file_name': img_name,
        'img_path': img_path,
        'width': w,
        'height': h,
        'annotations': anns
    }

def build_coco(records, out_json):
    images = []
    annotations = []
    ann_id = 1

    for img_id, rec in enumerate(records, start=1):
        images.append({
            'id': img_id,
            'file_name': rec['file_name'],
            'width': rec['width'],
            'height': rec['height']
        })

        for ann in rec['annotations']:
            new_ann = dict(ann)
            new_ann['id'] = ann_id
            new_ann['image_id'] = img_id
            annotations.append(new_ann)
            ann_id += 1

    coco = {
        'images': images,
        'annotations': annotations,
        'categories': [CATEGORY]
    }

    with open(out_json, 'w') as f:
        json.dump(coco, f, indent=2)

    print('[保存]', out_json)
    print('images =', len(images))
    print('annotations =', len(annotations))
    print('categories =', coco['categories'])

def main():
    os.makedirs(OUT_TRAIN_IMG, exist_ok=True)
    os.makedirs(OUT_VAL_IMG, exist_ok=True)
    os.makedirs(OUT_ANN_DIR, exist_ok=True)

    xmls = [
        os.path.join(SRC_XML_DIR, f)
        for f in sorted(os.listdir(SRC_XML_DIR))
        if f.lower().endswith('.xml')
    ]

    records = []

    for xml_path in xmls:
        rec = parse_xml(xml_path)
        if rec is not None:
            records.append(rec)

    if len(records) == 0:
        print('没有筛选到 liquid_drawing 数据。')
        return

    # 清空旧图片，避免旧文件干扰
    for d in [OUT_TRAIN_IMG, OUT_VAL_IMG]:
        for f in os.listdir(d):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                os.remove(os.path.join(d, f))

    # 与 micro_paper 同一标准：train 和 val 使用同一批图片
    for rec in records:
        shutil.copy2(rec['img_path'], os.path.join(OUT_TRAIN_IMG, rec['file_name']))
        shutil.copy2(rec['img_path'], os.path.join(OUT_VAL_IMG, rec['file_name']))

    build_coco(records, os.path.join(OUT_ANN_DIR, 'instances_train2024.json'))
    build_coco(records, os.path.join(OUT_ANN_DIR, 'instances_val2024.json'))

    print('\n完成 forensic_liquid 数据集构建。')

if __name__ == '__main__':
    main()
