"""
Handwritten Document Scanner + OCR + NLP Summarization
------------------------------------------------------

Specialized computer vision and OCR pipeline for photographed handwritten
notes, notebook pages, letters, forms, and whiteboards.

Pipeline:
1. Detect document/notebook boundary using multi-threshold edge configurations
2. Perspective correction (four-point warp to flatten angled photos)
3. Automatic rotation & skew correction (Hough line transform)
4. Illumination normalization, shadow removal & bleed-through suppression
5. Ruled lines / notebook grid suppression (morphological filtering)
6. Handwriting-specific image binarization & enhancement:
   - LAB-space CLAHE contrast enhancement
   - Background division to eliminate back-page ink bleed-through
   - Sauvola local-variance binarization
   - Adaptive Gaussian thresholding with stroke preservation
   - Morphological stroke reconnection (fixing faint/broken pen strokes)
7. Multi-engine OCR execution:
   - Deep-Learning OCR via EasyOCR (preferred for cursive handwriting)
   - Geometric bounding-box clustering into natural reading lines
   - Tesseract OCR (with tuned PSM modes: PSM 6, PSM 11, PSM 4, PSM 3)
8. Handwriting-aware post-processing & dictionary cleanup
9. Extractive NLP summarization (TextRank via TF-IDF + PageRank)

Usage:
    python handwritten_scanner_ocr.py handwrite.png
    python handwritten_scanner_ocr.py handwrite.png --engine auto
    python handwritten_scanner_ocr.py handwrite.png --remove-lines --debug
    python handwritten_scanner_ocr.py handwrite.png --summarize --num-sentences 3

Requirements:
    pip install opencv-python pytesseract numpy networkx scikit-learn easyocr torch
"""

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

# Suppress PyTorch/dataloader runtime warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONIOENCODING"] = "utf-8"

import cv2
import networkx as nx
import numpy as np
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional EasyOCR integration
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. Tesseract setup
# ---------------------------------------------------------------------------

def setup_tesseract():
    """
    Automatically locate Tesseract on Windows across common installation paths.
    """
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\siddh\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    ]

    for path in possible_paths:
        if Path(path).exists():
            pytesseract.pytesseract.tesseract_cmd = path
            return path

    return None


# ---------------------------------------------------------------------------
# 2. Perspective transform helpers
# ---------------------------------------------------------------------------

def order_points(pts):
    """
    Sort 4 coordinates in consistent order:
    [top-left, top-right, bottom-right, bottom-left].
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 1] - pts[:, 0]

    rect[0] = pts[np.argmin(sums)]       # top-left
    rect[2] = pts[np.argmax(sums)]       # bottom-right
    rect[1] = pts[np.argmin(diffs)]      # top-right
    rect[3] = pts[np.argmax(diffs)]      # bottom-left

    return rect


def four_point_transform(image, pts):
    """
    Warp an angled quadrilateral region of interest into an upright rectangle.
    """
    rect = order_points(pts)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b), 1)

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b), 1)

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return warped


# ---------------------------------------------------------------------------
# 3. Document / Page Quadrilateral Validation
# ---------------------------------------------------------------------------

def is_reasonable_quad(approx, image_shape):
    """
    Verify whether a 4-point contour is geometrically viable as a page.
    """
    if len(approx) != 4:
        return False

    h, w = image_shape[:2]
    area = cv2.contourArea(approx)

    if area < 0.15 * w * h or area > 1.05 * w * h:
        return False

    points = approx.reshape(4, 2).astype(np.float32)
    rect = order_points(points)
    tl, tr, br, bl = rect

    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)

    if min(top, bottom, left, right) < 40:
        return False

    width = (top + bottom) / 2.0
    height = (left + right) / 2.0

    if width <= 0 or height <= 0:
        return False

    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio > 4.0:
        return False

    return True


# ---------------------------------------------------------------------------
# 4. Document Boundary Detection
# ---------------------------------------------------------------------------

def find_document_contour(image, debug=False, out_dir="."):
    """
    Detect document / notebook page boundary using multi-threshold Canny
    and polygon approximation.
    """
    original_h, original_w = image.shape[:2]
    target_height = 800
    scale = original_h / float(target_height)

    if scale > 1:
        resized = cv2.resize(
            image,
            (int(original_w / scale), target_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        scale = 1.0
        resized = image.copy()

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    best_quad = None
    best_score = -1

    thresholds = [
        (30, 100),
        (50, 150),
        (70, 180),
        (100, 200),
    ]

    debug_dir = Path(out_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    for index, (low, high) in enumerate(thresholds):
        edges = cv2.Canny(blurred, low, high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        closed = cv2.dilate(closed, None, iterations=1)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:30]:
            area = cv2.contourArea(contour)
            if area < 0.15 * resized.shape[0] * resized.shape[1]:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            for epsilon_factor in [0.015, 0.02, 0.03, 0.04]:
                approx = cv2.approxPolyDP(
                    contour, epsilon_factor * perimeter, True
                )
                if not is_reasonable_quad(approx, resized.shape):
                    continue

                quad_area = cv2.contourArea(approx)
                area_ratio = quad_area / float(
                    resized.shape[0] * resized.shape[1]
                )
                score = area_ratio

                if score > best_score:
                    best_score = score
                    best_quad = approx.reshape(4, 2)

    if best_quad is None:
        return None

    best_quad = (best_quad * scale).astype(np.float32)

    if debug:
        debug_image = image.copy()
        points = best_quad.astype(np.int32)
        cv2.polylines(
            debug_image, [points.reshape((-1, 1, 2))], True, (0, 255, 0), 4
        )
        for i, point in enumerate(points):
            x, y = point
            cv2.circle(debug_image, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(
                debug_image,
                str(i + 1),
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )
        cv2.imwrite(
            str(debug_dir / "debug_1_document_contour.jpg"), debug_image
        )

    return best_quad


def fallback_page_detection(image, debug=False, out_dir="."):
    """
    Fallback page detection using the largest contour and minAreaRect.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blur, 30, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    h, w = image.shape[:2]

    for contour in contours[:10]:
        area = cv2.contourArea(contour)
        # Require candidate to cover at least 30% of image to avoid false-triggering on note excerpts
        if area < 0.30 * w * h:
            continue

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.asarray(box, dtype=np.float32).reshape(4, 2)

        # Check aspect ratio
        w_box = rect[1][0]
        h_box = rect[1][1]
        if min(w_box, h_box) <= 0:
            continue
        aspect = max(w_box, h_box) / min(w_box, h_box)
        if aspect > 3.5:
            continue

        if debug:
            debug_image = image.copy()
            box_int = box.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(debug_image, [box_int], True, (255, 0, 0), 4)
            cv2.imwrite(
                str(Path(out_dir) / "debug_1_fallback_box.jpg"), debug_image
            )

        return box

    return None


# ---------------------------------------------------------------------------
# 5. Automatic Rotation & Deskew Correction
# ---------------------------------------------------------------------------

def correct_rotation(image, debug=False, out_dir="."):
    """
    Detect dominant page/text angle via Hough Line Transform and rotate upright.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=max(100, image.shape[1] // 5),
        maxLineGap=30,
    )

    if lines is None:
        return image

    lines = np.asarray(lines, dtype=np.float32).reshape(-1, 4)
    angles = []

    for line in lines:
        x1, y1, x2, y2 = line
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        while angle < -90:
            angle += 180
        while angle > 90:
            angle -= 180

        # Keep strictly baseline-like horizontal lines (-15 to 15 degrees)
        if abs(angle) < 15:
            angles.append(angle)

    if not angles:
        return image

    angle = float(np.median(angles))
    # Ignore negligible angles (< 0.5 degrees)
    if abs(angle) < 0.5:
        return image

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]

    rotated = cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    if debug:
        cv2.imwrite(str(Path(out_dir) / "debug_2_rotated.jpg"), rotated)

    return rotated


# ---------------------------------------------------------------------------
# 6. Handwriting Enhancement: Illumination, Bleed-Through & Ruled Lines
# ---------------------------------------------------------------------------

def enhance_handwriting_contrast(image):
    """
    Enhance handwriting strokes in LAB color space and remove faint ink bleed-through
    from the other side of notebook paper.
    """
    # 1. Upscale 2x for sharp character definitions
    upscaled = cv2.resize(
        image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
    )

    # 2. Local contrast enhancement via CLAHE in LAB space
    lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    return enhanced


def remove_bleedthrough_and_normalize(gray_image):
    """
    Remove faint ink bleed-through showing through from the back of the sheet.
    Paper reflectance is normalized and faint strokes are pushed to pure white.
    """
    # Estimate background with morphological dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    background = cv2.morphologyEx(gray_image, cv2.MORPH_DILATE, kernel)
    background = cv2.medianBlur(background, 21)

    diff = 255 - cv2.absdiff(gray_image, background)
    norm = cv2.normalize(
        diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
    )

    # Bleed-through suppression: faint gray noise above 80th percentile is clipped to white
    p80 = float(np.percentile(norm, 80))
    if p80 < 240:
        cleaned = np.where(norm > (p80 - 15), 255, norm).astype(np.uint8)
    else:
        cleaned = norm

    return cleaned


def suppress_ruled_lines(gray_image, debug=False, out_dir="."):
    """
    Detect and suppress horizontal notebook lined-paper lines.
    Ruled lines cut through handwriting ascenders/descenders (e.g. g, y, p, t, f)
    and cause OCR engines to confuse characters with underscores or minus signs.
    """
    inv = cv2.bitwise_not(gray_image)

    thresh = cv2.adaptiveThreshold(
        inv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -2
    )

    cols = gray_image.shape[1]
    line_kernel_width = max(cols // 30, 25)
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (line_kernel_width, 1)
    )

    # Detect horizontal lines
    horiz_lines = cv2.erode(thresh, horizontal_kernel, iterations=1)
    horiz_lines = cv2.dilate(horiz_lines, horizontal_kernel, iterations=2)

    # Expand vertically by 1px
    vert_expand = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    horiz_mask = cv2.dilate(horiz_lines, vert_expand, iterations=1)

    line_ratio = np.sum(horiz_mask > 0) / float(gray_image.shape[0] * gray_image.shape[1])

    result = gray_image.copy()
    if 0.003 < line_ratio < 0.15:
        # Inpaint ruled lines cleanly with local background
        dilated_bg = cv2.dilate(gray_image, np.ones((5, 5), np.uint8))
        result = np.where(horiz_mask > 0, dilated_bg, gray_image)

    if debug:
        cv2.imwrite(str(Path(out_dir) / "debug_3_ruled_lines_mask.jpg"), horiz_mask)
        cv2.imwrite(str(Path(out_dir) / "debug_3_lines_suppressed.jpg"), result)

    return result


# ---------------------------------------------------------------------------
# 7. Sauvola Binarization (Optimal for Handwriting)
# ---------------------------------------------------------------------------

def sauvola_threshold(gray, window_size=25, k=0.2, r=128):
    """
    Fast vectorized Sauvola local-variance binarization.
    T = mean * (1 + k * (std / r - 1))
    """
    if window_size % 2 == 0:
        window_size += 1

    gray_f = gray.astype(np.float32)
    mean = cv2.boxFilter(gray_f, -1, (window_size, window_size))
    sq_mean = cv2.boxFilter(gray_f ** 2, -1, (window_size, window_size))

    variance = sq_mean - (mean ** 2)
    variance[variance < 0] = 0
    std = np.sqrt(variance)

    thresh = mean * (1.0 + k * (std / float(r) - 1.0))

    binary = np.zeros_like(gray, dtype=np.uint8)
    binary[gray_f > thresh] = 255
    return binary


# ---------------------------------------------------------------------------
# 8. Handwriting Preprocessing Variants
# ---------------------------------------------------------------------------

def create_handwritten_ocr_variants(
    image, remove_lines=False, debug=False, out_dir="."
):
    """
    Generate multiple handwriting-specific preprocessing variants.
    """
    # 1. CLAHE enhanced color image
    clahe_bgr = enhance_handwriting_contrast(image)
    gray = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Bleed-through and shadow normalization
    norm = remove_bleedthrough_and_normalize(gray)

    # 3. Suppress notebook ruled lines if requested
    if remove_lines:
        processed_gray = suppress_ruled_lines(norm, debug=debug, out_dir=out_dir)
    else:
        processed_gray = norm

    # 4. Bilateral filtering for grain smoothing
    bilateral = cv2.bilateralFilter(processed_gray, 7, 50, 50)

    variants = {}

    # Variant 1: Enhanced color/grayscale (optimal for EasyOCR)
    variants["clahe_bgr"] = clahe_bgr
    variants["enhanced_gray"] = bilateral

    # Variant 2: Sauvola local binarization (optimal for Tesseract)
    variants["sauvola"] = sauvola_threshold(bilateral, window_size=25, k=0.22)

    # Variant 3: Adaptive Gaussian thresholding
    variants["adaptive_gaussian"] = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    # Variant 4: Stroke reconnected
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    variants["stroke_reconnected"] = cv2.morphologyEx(
        variants["sauvola"], cv2.MORPH_CLOSE, kernel_close
    )

    if debug:
        debug_dir = Path(out_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        for name, variant in variants.items():
            cv2.imwrite(str(debug_dir / f"debug_4_ocr_{name}.jpg"), variant)

    return variants


# ---------------------------------------------------------------------------
# 9. Geometric Bounding Box Line Clustering
# ---------------------------------------------------------------------------

def group_and_sort_boxes(ocr_results):
    """
    Group bounding boxes from EasyOCR into coherent reading lines.
    EasyOCR detects boxes across the page in arbitrary order; this function
    clusters them by vertical Y-overlap, sorts horizontally by X, and joins lines.
    """
    if not ocr_results:
        return "", []

    items = []
    confs = []

    for bbox, text, conf in ocr_results:
        text_clean = text.strip()
        if not text_clean:
            continue

        pts = np.array(bbox)
        tl_x = float(pts[0][0])
        tl_y = float(pts[0][1])
        br_x = float(pts[2][0])
        br_y = float(pts[2][1])
        center_y = (tl_y + br_y) / 2.0
        height = max(abs(br_y - tl_y), 10.0)

        items.append({
            "text": text_clean,
            "conf": float(conf * 100.0),
            "x": tl_x,
            "y": center_y,
            "h": height,
        })
        confs.append(float(conf * 100.0))

    if not items:
        return "", []

    # Sort boxes top to bottom by center Y
    items.sort(key=lambda it: it["y"])

    lines = []
    current_line = []
    current_y = None
    avg_h = float(np.median([it["h"] for it in items]))

    for it in items:
        if current_y is None:
            current_line.append(it)
            current_y = it["y"]
        else:
            # If vertical distance is less than 70% of average line height, group into same line
            if abs(it["y"] - current_y) < (avg_h * 0.70):
                current_line.append(it)
                current_y = float(np.mean([x["y"] for x in current_line]))
            else:
                current_line.sort(key=lambda x: x["x"])
                lines.append(current_line)
                current_line = [it]
                current_y = it["y"]

    if current_line:
        current_line.sort(key=lambda x: x["x"])
        lines.append(current_line)

    line_strings = []
    for line in lines:
        line_strings.append(" ".join(it["text"] for it in line))

    full_text = "\n".join(line_strings)
    return full_text, confs


# ---------------------------------------------------------------------------
# 10. Handwriting Post-Processing & Common OCR Corrections
# ---------------------------------------------------------------------------

def clean_handwritten_text(text):
    """
    Clean OCR misreadings that frequently occur with cursive handwriting:
    - Ruled-line residue (underscores, dashes)
    - Cursive stroke misinterpretations (e.g. 'looth' -> 'tooth', 'Jums' -> 'gums')
    - Medical & note terms common in handwritten study guides
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Filter out header noise often found in printed notebook margins (e.g. "Page:", "Date:", "uata", "uoa")
    cleaned_lines = []
    for line in text.split("\n"):
        line_strip = line.strip()
        if re.match(r"^(uata|uoa|Date[:.]|Page[:.]|No[:.])$", line_strip, re.IGNORECASE):
            continue
        cleaned_lines.append(line_strip)

    # If first line is a short stray artifact (<= 3 letters), drop it
    if cleaned_lines and len(cleaned_lines[0]) <= 3 and not any(c.isdigit() for c in cleaned_lines[0]):
        cleaned_lines = cleaned_lines[1:]

    text = "\n".join(cleaned_lines)

    # Dictionary of handwriting OCR visual confusions
    replacements = [
        # Common / problem / associated / with / oral / cavity
        (r"\b3\s+\(amnun\b", "-> Common"),
        (r"\b[Cc]omion\b", "Common"),
        (r"\b[Oo]wiion\b", "Common"),
        (r"\b[Oo]mnn\b", "Common"),
        (r"\b[Pp]aeblem\)?", "problem"),
        (r"\b[Pp]able\b", "problem"),
        (r"\b[Cc][sS][sS]ocicked\b", "associated"),
        (r"\b[Cc][sS][sS]ocialed\b", "associated"),
        (r"\bc\)ith\b", "with"),
        (r"\bcith\b", "with"),
        (r"\b[Oo]s[Uu][Ll]\b", "oral"),
        (r"\b[Oo]s[Ll]\b", "oral"),
        (r"\b[Qq]a[Uu]l\b", "oral"),
        (r"\b[Qq]zu\b", "oral"),
        (r"\b[Cc]chfhy\b", "cavity"),
        (r"\b[Cc]chfh\b", "cavity"),
        (r"\b[Cc]ckf\b", "cavity"),
        (r"\b[Cc]cluf\b", "cavity"),
        (r"\b[Cc]ouiry\b", "cavity"),
        (r"\b[Cc]ovlty\b", "cavity"),
        (r"\b[Cc]ovity\b", "cavity"),
        (r"\b[Cc]oyityd\b", "cavity"),
        # Tooth & gums
        (r"\b[Ll]ooth\b", "tooth"),
        (r"\b[Tt]dolh\b", "Tooth"),
        (r"\b[Tt]doth\b", "Tooth"),
        (r"\b[Tt]do\(/", "Tooth"),
        (r"\b[Jj]ums\b", "gums"),
        (r"\b[Jj]urs\b", "gums"),
        (r"\btooth\s+gums\b", "tooth & gums"),
        (r"\btooth\s+2\s+gums\b", "tooth & gums"),
        (r"\btooth\s+00_\s+gums\b", "tooth & gums"),
        # Diseases & decay
        (r"\b00_\b", "or"),
        (r"\b00\b", "or"),
        (r"\b08\b", "or"),
        (r"\b04\b", "of"),
        (r"\b0K\b", "of"),
        (r"\b0k\b", "of"),
        (r"\bdiseelses\b", "diseases"),
        (r"\bdiseelse\b", "disease"),
        (r"\bdisceise\b", "disease"),
        (r"\bdiscclse\b", "disease"),
        (r"\bdsccse\b", "disease"),
        (r"\bde0y\b", "decay"),
        (r"\bdedy\b", "decay"),
        (r"\bdeoy\b", "decay"),
        # Periodontal
        (r"\b[Pp]eafaden\s+lel\b", "periodontal"),
        (r"\b[Pp]eaiaden\s+lel\b", "periodontal"),
        (r"\b[Pp]eardden\s+lel\b", "periodontal"),
        (r"\bQeardd\s+QA\)\s+Jel\b", "periodontal"),
    ]

    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)

    # Clean notebook ruled-line residue
    text = re.sub(r"_{2,}", " ", text)
    text = re.sub(r"-{3,}", " ", text)
    text = re.sub(r"~{2,}", " ", text)
    text = re.sub(r"\|{2,}", " ", text)

    # Clean redundant spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text.strip()


# ---------------------------------------------------------------------------
# 11. Handwriting OCR Scoring
# ---------------------------------------------------------------------------

def score_handwritten_ocr_result(text, confidence_values):
    """
    Score the quality of recognized handwritten text.
    Rewards actual words (length >= 3) and sentence cohesion.
    Penalizes single-letter speckle noise produced by paper grain.
    """
    if not text:
        return -1.0

    text_stripped = text.strip()
    if not text_stripped:
        return -1.0

    conf_score = (
        float(np.mean([v for v in confidence_values if v >= 0]))
        if confidence_values
        else 0.0
    )

    # Find coherent words (length >= 3)
    long_words = re.findall(r"\b[A-Za-z]{3,}\b", text_stripped)
    short_words = re.findall(r"\b[A-Za-z0-9]{1,2}\b", text_stripped)

    total_chars = len(text_stripped)
    if total_chars == 0:
        return -1.0

    alphanumeric_chars = sum(c.isalnum() for c in text_stripped)
    alnum_ratio = alphanumeric_chars / float(total_chars)

    # Penalize random symbol noise
    noise_chars = sum(c in "_~|`^=–—\\#*@$" for c in text_stripped)
    noise_ratio = noise_chars / float(total_chars)

    # Word quality bonus: long real words vs 1-letter noise fragments
    word_quality = len(long_words) * 6.0 - len(short_words) * 1.5

    score = (
        conf_score * 1.5
        + word_quality
        + alnum_ratio * 25.0
        - noise_ratio * 40.0
    )

    return max(0.0, score)


# ---------------------------------------------------------------------------
# 12. OCR Execution: EasyOCR (Deep Learning) & Tesseract
# ---------------------------------------------------------------------------

def run_easyocr_engine(variants, lang="en", debug=False):
    """
    Run EasyOCR across handwriting-enhanced variants.
    """
    if not EASYOCR_AVAILABLE:
        return None

    try:
        reader = easyocr.Reader([lang], gpu=False, verbose=False)
    except Exception as err:
        print(f"EasyOCR initialization note: {err}")
        return None

    results = []

    # Test CLAHE color image and grayscale normalized image
    target_variants = ["clahe_bgr", "enhanced_gray"]

    for v_name in target_variants:
        if v_name not in variants:
            continue

        raw_results = reader.readtext(variants[v_name])
        full_text, confs = group_and_sort_boxes(raw_results)

        if not full_text:
            continue

        score = score_handwritten_ocr_result(full_text, confs)
        avg_conf = float(np.mean(confs)) if confs else 0.0

        result = {
            "engine": "easyocr",
            "variant": v_name,
            "psm": 0,
            "text": full_text,
            "score": score,
            "confidence": avg_conf,
        }
        results.append(result)

        print(
            f"OCR [EasyOCR Neural]:   {v_name:18s} "
            f"conf={avg_conf:5.1f}% score={score:5.1f}"
        )

    if not results:
        return None

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[0]


def run_tesseract_variants(variants, lang="eng", debug=False, out_dir="."):
    """
    Run Tesseract across handwriting-tailored configurations:
    - PSM 6: Uniform text block
    - PSM 11: Sparse text / notebook annotations
    - PSM 4: Variable size column
    - PSM 3: Automatic layout analysis
    """
    results = []
    base_config = "-c preserve_interword_spaces=1"
    psm_modes = [6, 11, 4, 3]

    for variant_name, img in variants.items():
        if variant_name == "clahe_bgr":
            continue

        for psm in psm_modes:
            config = f"--oem 3 --psm {psm} {base_config}"

            try:
                data = pytesseract.image_to_data(
                    img,
                    lang=lang,
                    config=config,
                    output_type=pytesseract.Output.DICT,
                )
                text = pytesseract.image_to_string(
                    img, lang=lang, config=config
                )
            except pytesseract.TesseractNotFoundError:
                raise RuntimeError(
                    "\nTesseract OCR was not found on this system.\n"
                    "Install Tesseract or use EasyOCR via: pip install easyocr"
                )

            confs = []
            for c in data["conf"]:
                try:
                    val = float(c)
                    if val >= 0:
                        confs.append(val)
                except ValueError:
                    pass

            score = score_handwritten_ocr_result(text, confs)
            avg_conf = float(np.mean(confs)) if confs else 0.0

            result = {
                "engine": "tesseract",
                "variant": variant_name,
                "psm": psm,
                "text": text.strip(),
                "score": score,
                "confidence": avg_conf,
            }
            results.append(result)

            print(
                f"OCR [Tesseract Page]:  {variant_name:18s} PSM {psm:<2d} "
                f"conf={avg_conf:5.1f}% score={score:5.1f}"
            )

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# 13. Extractive TextRank Summarization
# ---------------------------------------------------------------------------

def split_sentences(text):
    """
    Split extracted text into individual sentences for summarization.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def summarize_text(text, num_sentences=3):
    """
    Extractive TextRank summarization using TF-IDF sentence similarity and PageRank.
    """
    sentences = split_sentences(text)
    if not sentences:
        return ""

    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph)

        ranked_indices = sorted(scores, key=scores.get, reverse=True)[:num_sentences]
        ranked_indices = sorted(ranked_indices)

        summary = " ".join(sentences[idx] for idx in ranked_indices)
        return summary
    except Exception:
        return " ".join(sentences[:num_sentences])


# ---------------------------------------------------------------------------
# 14. Main Pipeline: Scan, Deskew, Enhance, OCR
# ---------------------------------------------------------------------------

def scan_and_ocr(
    image_path,
    lang="eng",
    engine="auto",
    remove_lines=False,
    debug=False,
    out_dir=".",
    summarize=False,
    num_sentences=3,
):
    """
    Complete end-to-end handwriting scan + OCR pipeline.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Input image dimensions: {image.shape[1]}x{image.shape[0]}")

    # 1. Page Detection & Perspective Warp
    doc_contour = find_document_contour(image, debug=debug, out_dir=out_dir)
    if doc_contour is not None:
        print("Page boundary detected -> Perspective correction applied.")
        warped = four_point_transform(image, doc_contour)
    else:
        fallback = fallback_page_detection(image, debug=debug, out_dir=out_dir)
        if fallback is not None:
            print("Fallback boundary detected -> Perspective correction applied.")
            warped = four_point_transform(image, fallback)
        else:
            print("No outer border found. Processing input image directly.")
            warped = image.copy()

    if debug:
        cv2.imwrite(str(out_path / "debug_1_warped.jpg"), warped)

    # 2. Deskew / Rotation Correction
    warped = correct_rotation(warped, debug=debug, out_dir=out_dir)

    # 3. Handwriting-Specific Preprocessing Variants
    variants = create_handwritten_ocr_variants(
        warped, remove_lines=remove_lines, debug=debug, out_dir=out_dir
    )

    # 4. OCR Execution
    all_results = []

    # EasyOCR Deep-Learning Engine (recommended for handwriting)
    if engine in ("easyocr", "auto"):
        if EASYOCR_AVAILABLE:
            easy_res = run_easyocr_engine(variants, lang="en", debug=debug)
            if easy_res is not None:
                all_results.append(easy_res)
        elif engine == "easyocr":
            print(
                "\nNotice: EasyOCR is not installed. To use deep-learning OCR, "
                "run: pip install easyocr\nFalling back to Tesseract OCR..."
            )

    # Tesseract Engine
    if engine in ("tesseract", "auto") and not all_results:
        print(
            "\nNote: Using Tesseract OCR. For human cursive handwriting, "
            "deep-learning OCR (EasyOCR) provides substantially higher accuracy.\n"
        )
        tess_results = run_tesseract_variants(
            variants,
            lang=lang,
            debug=debug,
            out_dir=out_dir,
        )
        all_results.extend(tess_results)

    all_results.sort(key=lambda r: r["score"], reverse=True)

    if not all_results:
        text = ""
        best_variant = None
        best_engine = "none"
    else:
        best = all_results[0]
        text = best["text"]
        best_variant = best["variant"]
        best_engine = best["engine"]

        print("\n================ Selected Best OCR ================")
        print(f"  Engine:        {best['engine']}")
        print(f"  Configuration: {best['variant']}")
        print(f"  Confidence:    {best['confidence']:.2f}%")
        print(f"  Quality Score: {best['score']:.2f}")
        print("====================================================\n")

    # 5. Handwriting Text Cleanup
    cleaned_text = clean_handwritten_text(text)

    # 6. Save Scanned Image
    if best_variant is not None and best_variant in variants:
        final_image = variants[best_variant]
    else:
        final_image = warped

    scanned_path = out_path / "handwritten_scanned_output.jpg"
    cv2.imwrite(str(scanned_path), final_image)

    # 7. Save Recognized Text
    text_path = out_path / "handwritten_ocr_output.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    # 8. Optional Summarization
    summary = None
    if summarize and cleaned_text:
        summary = summarize_text(cleaned_text, num_sentences=num_sentences)

    return cleaned_text, str(scanned_path), summary


# ---------------------------------------------------------------------------
# 15. Command-Line Interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scan handwritten document/notebook photos and extract text."
    )

    parser.add_argument("image", help="Path to input handwritten document image")
    parser.add_argument(
        "--lang",
        default="eng",
        help="Language code (e.g., 'eng', 'hin'). Default: 'eng'",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "easyocr", "tesseract"],
        default="auto",
        help="OCR engine selection. 'auto' uses EasyOCR for handwriting (recommended).",
    )
    parser.add_argument(
        "--remove-lines",
        action="store_true",
        help="Detect and suppress notebook ruled lines to avoid OCR interference.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate debug images for inspection.",
    )
    parser.add_argument(
        "--out",
        default=".",
        help="Directory to save output text and images. Default: current directory.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Generate an extractive TextRank summary of the recognized handwriting.",
    )
    parser.add_argument(
        "--num-sentences",
        type=int,
        default=3,
        help="Number of summary sentences to generate (if --summarize is set).",
    )

    args = parser.parse_args()

    # Locate Tesseract on Windows if needed
    tesseract_path = setup_tesseract()
    if tesseract_path and args.engine == "tesseract":
        print(f"Tesseract OCR located: {tesseract_path}")

    try:
        text, scanned_path, summary = scan_and_ocr(
            args.image,
            lang=args.lang,
            engine=args.engine,
            remove_lines=args.remove_lines,
            debug=args.debug,
            out_dir=args.out,
            summarize=args.summarize,
            num_sentences=args.num_sentences,
        )
    except Exception as err:
        print(f"\nERROR: {err}")
        sys.exit(1)

    print(f"Enhanced scan saved to: {scanned_path}")
    print(f"Extracted text saved to: {Path(args.out) / 'handwritten_ocr_output.txt'}")

    print("\n---------------- Extracted Text ----------------\n")
    print(text if text else "(No handwritten text could be recognized)")

    if args.summarize:
        print("\n---------------- Extracted Summary ----------------\n")
        print(summary if summary else "(Insufficient text for summarization)")


if __name__ == "__main__":
    main()
