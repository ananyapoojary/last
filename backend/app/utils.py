from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np
from fuzzywuzzy import fuzz

ocr = PaddleOCR(use_angle_cls=True, lang='en', det_db_score_mode='fast', layout=True)

def compute_iou(box1, box2):
    box1 = np.array(box1).reshape(-1, 2)
    box2 = np.array(box2).reshape(-1, 2)
    x_min1, y_min1 = np.min(box1, axis=0)
    x_max1, y_max1 = np.max(box1, axis=0)
    x_min2, y_min2 = np.min(box2, axis=0)
    x_max2, y_max2 = np.max(box2, axis=0)

    inter_xmin = max(x_min1, x_min2)
    inter_ymin = max(y_min1, y_min2)
    inter_xmax = min(x_max1, x_max2)
    inter_ymax = min(y_max1, y_max2)

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def is_similar(text1, text2, threshold=85):
    return fuzz.ratio(text1.strip().lower(), text2.strip().lower()) >= threshold

def preprocess_variants(img):
    variants = [img]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    adaptive_bgr = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
    variants.append(adaptive_bgr)

    bilateral = cv2.bilateralFilter(adaptive_bgr, 9, 75, 75)
    morph = cv2.morphologyEx(bilateral, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8))
    gamma = 1.5
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    corrected = cv2.LUT(morph, table)
    variants.append(corrected)

    return variants

def process_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError("Image not found or unreadable")

    all_results = []
    for variant in preprocess_variants(image):
        result = ocr.ocr(variant, cls=True)[0]
        if result:
            all_results.extend(result)

    final_results = []
    for box, (text, score) in sorted(all_results, key=lambda x: x[1][1], reverse=True):
        matched = False
        for i, (existing_box, (existing_text, existing_score)) in enumerate(final_results):
            if is_similar(text, existing_text) and compute_iou(box, existing_box) > 0.5:
                matched = True
                if score > existing_score:
                    final_results[i] = (box, (text, score))
                break
        if not matched:
            final_results.append((box, (text, score)))

    boxes = [item[0] for item in final_results]
    txts = [item[1][0] for item in final_results]
    scores = [item[1][1] for item in final_results]

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    image_with_boxes = draw_ocr(image, boxes, txts, scores, font_path=font_path)

    output_path = "app/output.png"
    cv2.imwrite(output_path, image_with_boxes)

    return txts, output_path
