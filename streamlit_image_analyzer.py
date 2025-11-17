"""
Streamlit Image Analyzer for Social Media Campaigns
File: streamlit_image_analyzer.py

This file provides two modes:
1) Streamlit web app (if `streamlit` is installed)
2) CLI/fallback mode (if `streamlit` is NOT available) — useful for sandboxed environments

If you ran into "ModuleNotFoundError: No module named 'streamlit'", this version will *not* crash: it falls back to a CLI mode that runs the same analysis on a sample image or on an image path you provide.

Features (same as original):
- Upload image (single) / CLI path
- Compute brightness, contrast, sharpness, colorfulness
- Detect faces (OpenCV Haar cascades)
- OCR text-on-image (pytesseract fallback to easyocr if available)
- Dominant colors (KMeans)
- Composition score (rule-of-thirds, center of mass)
- Text density and readability estimate
- Platform recommendations (crop sizes for IG/FB/X)
- Engagement prediction score (simple heuristic model)
- Download results as JSON/CSV (Streamlit) or save JSON to disk (CLI)

Dependencies (for full features):
- numpy, pillow, opencv-python, scikit-learn, scikit-image
- Optional: pytesseract (plus system tesseract), easyocr
- Streamlit only needed for the web UI

"""

import sys
import os
import io
import math
import json
import argparse
from datetime import datetime

from PIL import Image
import numpy as np
import cv2

# Try to import streamlit; if not available, we'll run CLI mode.
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

# Optional OCR imports
try:
    import pytesseract
    PytesseractAvailable = True
except Exception:
    PytesseractAvailable = False

try:
    import easyocr
    EasyOCRAvailable = True
except Exception:
    EasyOCRAvailable = False

# ML imports for dominant color
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ----------------------- Utility functions -----------------------

def pil_to_cv2(img_pil):
    open_cv_image = np.array(img_pil.convert('RGB'))
    # Convert RGB to BGR
    return open_cv_image[:, :, ::-1].copy()


def cv2_to_pil(img_cv):
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def compute_brightness_contrast(img_pil):
    img_gray = np.array(img_pil.convert('L')).astype('float')
    brightness = np.mean(img_gray) / 255.0  # 0..1
    contrast = np.std(img_gray) / 128.0  # normalized roughly to 0..1
    return float(brightness), float(contrast)


def compute_sharpness(img_pil):
    img_cv = pil_to_cv2(img_pil)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(lap.var())
    sharp_norm = 1 - math.exp(-sharpness / 1000.0)
    return float(sharp_norm)


def colorfulness_score(img_pil):
    img = np.array(img_pil.convert('RGB')).astype('float')
    (R, G, B) = (img[:, :, 0], img[:, :, 1], img[:, :, 2])
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_rg = np.std(rg)
    std_yb = np.std(yb)
    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)
    std_root = math.sqrt(std_rg ** 2 + std_yb ** 2)
    mean_root = math.sqrt(mean_rg ** 2 + mean_yb ** 2)
    score = std_root + 0.3 * mean_root
    return float(1 - math.exp(-score / 40.0))


def dominant_colors(img_pil, k=4):
    img = img_pil.copy().convert('RGB')
    img = img.resize((300, 300))
    arr = np.array(img).reshape(-1, 3).astype(float)
    if not SKLEARN_AVAILABLE:
        # Fallback: return a simple histogram top colors
        pixels = arr.astype(int)
        pixels_tuple = [tuple(p) for p in pixels]
        from collections import Counter
        cnt = Counter(pixels_tuple)
        most = cnt.most_common(k)
        return [ (tuple(c[0]), c[1]) for c in most ]
    k = min(k, len(arr))
    kmeans = KMeans(n_clusters=k, random_state=0).fit(arr)
    centers = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)
    order = np.argsort(counts)[::-1]
    return [(tuple(centers[i].tolist()), int(counts[i])) for i in order]


def face_detection_cv(img_pil):
    img_cv = pil_to_cv2(img_pil)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        faces_list = [tuple(map(int, f)) for f in faces]
        return faces_list
    except Exception:
        return []


def ocr_text(img_pil):
    text = ""
    if PytesseractAvailable:
        try:
            text = pytesseract.image_to_string(img_pil)
            return text.strip()
        except Exception:
            text = ""
    if EasyOCRAvailable:
        try:
            reader = easyocr.Reader(['en'], gpu=False)
            result = reader.readtext(np.array(img_pil))
            text = " ".join([r[1] for r in result])
            return text.strip()
        except Exception:
            return ""
    return ""


def text_density(img_pil, ocr_str):
    if not ocr_str:
        return 0.0
    char_count = len(ocr_str)
    w, h = img_pil.size
    area = w * h
    density = char_count / (area / 10000.0)
    return float(min(1.0, density / 10.0))


def rule_of_thirds_score(img_pil):
    img_cv = pil_to_cv2(img_pil)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        return 0.0
    cx = np.mean(xs)
    cy = np.mean(ys)
    h, w = gray.shape
    thirds = [(w / 3.0, h / 3.0), (2 * w / 3.0, h / 3.0), (w / 3.0, 2 * h / 3.0), (2 * w / 3.0, 2 * h / 3.0)]
    dists = [math.hypot(cx - tx, cy - ty) for (tx, ty) in thirds]
    min_dist = min(dists)
    max_possible = math.hypot(w, h)
    score = 1.0 - (min_dist / max_possible)
    return float(max(0.0, min(1.0, score)))


def composition_clutter_score(img_pil):
    img_gray = np.array(img_pil.convert('L'))
    hist, _ = np.histogram(img_gray.flatten(), bins=256, range=(0, 255), density=True)
    hist = hist + 1e-12
    entropy = -np.sum(hist * np.log2(hist))
    return float(min(1.0, entropy / 8.0))


def readability_estimate(img_pil, ocr_str):
    if not ocr_str:
        return 1.0
    return 0.6


def platform_recommendations(width, height):
    recs = {
        'Instagram (feed portrait)': {'size': '1080x1350', 'recommended_crop': (4, 5)},
        'Instagram (square)': {'size': '1080x1080', 'recommended_crop': (1, 1)},
        'Facebook (post)': {'size': '1200x630', 'recommended_crop': (1.91, 1)},
        'Twitter/X (post)': {'size': '1200x675', 'recommended_crop': (16, 9)},
    }
    return recs


def engagement_prediction(metrics):
    weights = {
        'face': 0.2,
        'brightness': 0.05,
        'contrast': 0.05,
        'sharpness': 0.1,
        'colorfulness': 0.15,
        'rule_of_thirds': 0.15,
        'clutter': -0.1,
        'text_density': -0.05,
        'readability': 0.05,
    }
    score = 0.5
    score += weights['face'] * (1.0 if metrics.get('face', 0) > 0 else 0)
    score += weights['brightness'] * metrics.get('brightness', 0)
    score += weights['contrast'] * metrics.get('contrast', 0)
    score += weights['sharpness'] * metrics.get('sharpness', 0)
    score += weights['colorfulness'] * metrics.get('colorfulness', 0)
    score += weights['rule_of_thirds'] * metrics.get('rule_of_thirds', 0)
    score += weights['clutter'] * metrics.get('clutter', 0)
    score += weights['text_density'] * metrics.get('text_density', 0)
    score += weights['readability'] * metrics.get('readability', 0)
    score = max(0.0, min(1.0, score))
    return int(score * 100)


# ----------------------- Runner / CLI & Tests -----------------------

def analyze_image(img_pil):
    brightness, contrast = compute_brightness_contrast(img_pil)
    sharpness = compute_sharpness(img_pil)
    colorfulness = colorfulness_score(img_pil)
    colors = dominant_colors(img_pil, k=5)
    faces = face_detection_cv(img_pil)
    ocr_str = ocr_text(img_pil)
    text_den = text_density(img_pil, ocr_str)
    rule3 = rule_of_thirds_score(img_pil)
    clutter = composition_clutter_score(img_pil)
    readability = readability_estimate(img_pil, ocr_str)
    metrics = {
        'brightness': brightness,
        'contrast': contrast,
        'sharpness': sharpness,
        'colorfulness': colorfulness,
        'rule_of_thirds': rule3,
        'clutter': clutter,
        'text_density': text_den,
        'readability': readability,
        'face': len(faces),
    }
    er = engagement_prediction(metrics)
    results = {
        'metrics': metrics,
        'engagement_score': er,
        'faces': faces,
        'ocr': ocr_str,
        'dominant_colors': colors,
    }
    return results


def save_results_json(results, filename='image_analysis_cli.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, default=str, ensure_ascii=False, indent=2)
    return filename


def make_sample_image(width=800, height=600):
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(arr.shape[0]):
        arr[i, :, 0] = np.linspace(50, 255, arr.shape[1], dtype=np.uint8)
        arr[i, :, 1] = 120
        arr[i, :, 2] = np.linspace(255, 50, arr.shape[1], dtype=np.uint8)
    return Image.fromarray(arr)


def cli_main():
    parser = argparse.ArgumentParser(description='Image Analyzer CLI fallback (no streamlit).')
    parser.add_argument('--image', '-i', help='Path to image file to analyze')
    parser.add_argument('--sample', '-s', action='store_true', help='Analyze a generated sample image')
    parser.add_argument('--out', '-o', default='image_analysis_cli.json', help='Output JSON filename')
    args = parser.parse_args()

    if not args.image and not args.sample:
        print('No image provided. Running built-in tests / sample analysis...')
        # run a basic test
        img = make_sample_image()
        results = analyze_image(img)
        out = save_results_json(results, args.out)
        print(f'Sample analysis saved to {out}')
        print('Summary:')
        print(f"Engagement score: {results['engagement_score']}/100")
        print('Metrics:')
        for k, v in results['metrics'].items():
            print(f' - {k}: {v}')
        # save annotated image
        annotated = pil_to_cv2(img)
        for (x, y, ww, hh) in results['faces']:
            cv2.rectangle(annotated, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
        h_cv, w_cv = annotated.shape[:2]
        for gx in [w_cv // 3, 2 * w_cv // 3]:
            cv2.line(annotated, (gx, 0), (gx, h_cv), (255, 255, 255), 1)
        for gy in [h_cv // 3, 2 * h_cv // 3]:
            cv2.line(annotated, (0, gy), (w_cv, gy), (255, 255, 255), 1)
        out_img = 'annotated_sample.png'
        Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)).save(out_img)
        print(f'Annotated image saved to {out_img}')
        return

    if args.sample:
        img = make_sample_image()
    else:
        if not os.path.exists(args.image):
            print(f'Image path does not exist: {args.image}', file=sys.stderr)
            sys.exit(2)
        img = Image.open(args.image).convert('RGB')

    results = analyze_image(img)
    out = save_results_json(results, args.out)
    print(f'Analysis saved to {out}')
    print(f"Engagement score: {results['engagement_score']}/100")


# ----------------------- Streamlit App -----------------------

if STREAMLIT_AVAILABLE:
    st.set_page_config(layout='wide', page_title='Image Analyzer for Social Campaigns')
    st.title('Image Analyzer — Social Media Campaigns')
    st.markdown('Unggah gambar untuk mendapatkan analisis visual + rekomendasi untuk Instagram, Facebook, dan Twitter/X.')

    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded = st.file_uploader('Upload image', type=['png', 'jpg', 'jpeg', 'webp'])
        st.write('Atau coba contoh:')
        sample = st.button('Gunakan sample image')

        if sample and not uploaded:
            img_pil = make_sample_image()
            buf = io.BytesIO()
            img_pil.save(buf, format='PNG')
            buf.seek(0)
            uploaded = buf

        if uploaded:
            img_pil = Image.open(uploaded).convert('RGB')
            st.image(img_pil, use_column_width=True, caption='Preview')
            w, h = img_pil.size
            st.write(f'Size: {w} x {h} px')

    with col2:
        if not ('uploaded' in locals() and uploaded):
            st.info('Upload gambar di panel kiri untuk memulai analisis.')
        else:
            with st.spinner('Menganalisis...'):
                results = analyze_image(img_pil)

            metrics = results['metrics']
            er = results['engagement_score']
            faces = results['faces']
            ocr_str = results['ocr']
            colors = results['dominant_colors']
            brightness = metrics['brightness']
            contrast = metrics['contrast']
            sharpness = metrics['sharpness']
            colorfulness = metrics['colorfulness']
            rule3 = metrics['rule_of_thirds']
            clutter = metrics['clutter']
            text_den = metrics['text_density']

            st.subheader('Metrics')
            st.metric('Predicted Engagement Score', f'{er} / 100')

            st.write('**Visual metrics (0..1)**')
            cols = st.columns(4)
            cols[0].metric('Brightness', f'{brightness:.2f}')
            cols[1].metric('Contrast', f'{contrast:.2f}')
            cols[2].metric('Sharpness', f'{sharpness:.2f}')
            cols[3].metric('Colorfulness', f'{colorfulness:.2f}')

            st.write('**Composition & Content**')
            ccols = st.columns(3)
            ccols[0].metric('Rule of Thirds Score', f'{rule3:.2f}')
            ccols[1].metric('Clutter (entropy)', f'{clutter:.2f}')
            ccols[2].metric('Text Density', f'{text_den:.2f}')

            st.write('Faces detected: ', len(faces))
            if len(faces) > 0:
                st.write(faces)

            if ocr_str:
                st.write('Detected text (OCR):')
                st.text_area('OCR output', ocr_str, height=120)
            else:
                st.info('No visible text detected (OCR)')

            st.write('Dominant colors:')
            color_cols = st.columns(len(colors))
            for i, (c, cnt) in enumerate(colors):
                try:
                    r, g, b = c
                except Exception:
                    r, g, b = (0, 0, 0)
                color_cols[i].markdown(f'<div style="width:100%;height:60px;background:rgb({r},{g},{b});"></div>', unsafe_allow_html=True)
                color_cols[i].caption(f'{c} - {cnt} px')

            st.write('Platform recommendations:')
            recs = platform_recommendations(*img_pil.size)
            for k, v in recs.items():
                st.write(f'- **{k}** — ideal size {v["size"]}, crop ratio approx {v["recommended_crop"]}')

            annotated = pil_to_cv2(img_pil)
            for (x, y, ww, hh) in faces:
                cv2.rectangle(annotated, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
            h_cv, w_cv = annotated.shape[:2]
            for gx in [w_cv // 3, 2 * w_cv // 3]:
                cv2.line(annotated, (gx, 0), (gx, h_cv), (255, 255, 255), 1)
            for gy in [h_cv // 3, 2 * h_cv // 3]:
                cv2.line(annotated, (0, gy), (w_cv, gy), (255, 255, 255), 1)
            st.image(cv2_to_pil(annotated), caption='Annotated (faces + thirds)')

            st.download_button('Download JSON report', json.dumps(results, default=str), file_name='image_analysis.json')

            st.subheader('Simple recommendations')
            rec_list = []
            if brightness < 0.35:
                rec_list.append('Gambar agak gelap — pertimbangkan meningkatkan exposure atau brightness.')
            if contrast < 0.15:
                rec_list.append('Kontras rendah — tambahkan kontras agar lebih menonjol di feed.')
            if sharpness < 0.3:
                rec_list.append('Kurang tajam — gunakan foto yang lebih fokus atau tingkatkan sharpness.')
            if colorfulness < 0.25:
                rec_list.append('Warna kurang hidup — pertimbangkan saturasi yang sedikit lebih tinggi sesuai brand.')
            if text_den > 0.6:
                rec_list.append('Terlalu banyak teks — kurangi teks di gambar atau pindahkan ke caption.')
            if len(faces) == 0:
                rec_list.append('Tidak ada wajah — jika ingin engagement lebih tinggi, sertakan manusia/ekspresi.')
            if rule3 < 0.3:
                rec_list.append('Komposisi kurang ideal — pertimbangkan rule of thirds.')

            if rec_list:
                for r in rec_list:
                    st.write('- ', r)
            else:
                st.write('Gambar nampak baik untuk posting — coba A/B test untuk konfirmasi.')

    st.markdown('---')
    st.caption('Built with ❤️ — modify weights and heuristics to fit your brand or A/B test results for better prediction.')


if __name__ == '__main__':
    if STREAMLIT_AVAILABLE:
        # If streamlit is installed, the user should run via `streamlit run streamlit_image_analyzer.py`
        print('Streamlit is available. To run the app, execute:')
        print('  streamlit run streamlit_image_analyzer.py')
    else:
        # CLI fallback
        cli_main()
