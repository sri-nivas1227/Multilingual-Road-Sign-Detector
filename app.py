from flask import Flask, request, render_template
import easyocr
import numpy as np
import cv2
import os
import requests

app = Flask(__name__)

# 1) EasyOCR readers
latin_langs  = ['en','fr','de','es','pt','it','nl']
reader_latin = easyocr.Reader(latin_langs, gpu=False)
reader_ch    = easyocr.Reader(['ch_sim','en'], gpu=False)
reader_ja    = easyocr.Reader(['ja','en'],     gpu=False)
reader_ko    = easyocr.Reader(['ko','en'],     gpu=False)
reader_ar    = easyocr.Reader(['ar','en'],     gpu=False)
reader_hi    = easyocr.Reader(['hi','en'],     gpu=False)

# 2) DeepL Free-API config
DEEPL_KEY = "ba1608b8-dfc2-48e1-964c-4240298e8cc1:fx"
DEEPL_URL = "https://api-free.deepl.com/v2/translate"

def deepl_translate(text: str) -> str:
    try:
        resp = requests.post(
            DEEPL_URL,
            headers={"Authorization": f"DeepL-Auth-Key {DEEPL_KEY}"},
            data=[("text", text), ("target_lang", "EN")],
            timeout=5
        )
        resp.raise_for_status()
        return resp.json()["translations"][0]["text"]
    except Exception as e:
        print(f"[DeepL HTTP error] {e}")
        return text

# 3) Accent-restoration map for common French OCR errors
ACCENT_MAP = {
    "ARR?T": "ARRÊT",
    "ARRET": "ARRÊT",
    "LIVRAISONS": "LIVRAISONS",
    # add more as needed…
}

def restore_accents(txt: str) -> str:
    for bad, good in ACCENT_MAP.items():
        txt = txt.replace(bad, good)
    return txt

def iou(boxA, boxB):
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    areaA = (xa2-xa1)*(ya2-ya1)
    areaB = (xb2-xb1)*(yb2-yb1)
    union = areaA + areaB - inter
    return inter/union if union>0 else 0

@app.route('/', methods=['GET','POST'])
def upload_image():
    extracted_text, translated_text = [], []
    output_image = None

    if request.method == 'POST':
        file = request.files.get('image')
        if not file:
            return render_template(
                'index.html',
                extracted_text=[], translated_text=[], output_image=None
            )

        # Load image
        npimg = np.frombuffer(file.read(), np.uint8)
        img   = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # OCR passes
        raw_results = []
        for rdr in (reader_latin, reader_ch, reader_ja,
                    reader_ko, reader_ar, reader_hi):
            raw_results += rdr.readtext(img)

        # Sort into reading order
        heights = [abs(b[2][1] - b[0][1]) for b,_,_ in raw_results]
        avg_h   = float(np.mean(heights)) if heights else 1.0
        def sort_key(item):
            (x0,y0),_,(x2,y2),_ = item[0]
            row = int(((y0+y2)/2) // (avg_h * 0.8))
            return (row, x0)
        raw_results.sort(key=sort_key)

        # Spatial dedupe + accent restoration + translation
        seen_boxes = []
        for bbox, text, conf in raw_results:
            if conf < 0.4:
                continue

            (x0,y0),_,(x2,y2),_ = bbox
            box = (int(x0), int(y0), int(x2), int(y2))

            if any(iou(box, sb) > 0.5 for sb in seen_boxes):
                continue
            seen_boxes.append(box)

            # normalize & restore accents
            txt = text.strip().upper()
            txt = restore_accents(txt)

            extracted_text.append(txt)
            translated_text.append(deepl_translate(txt))

            # draw on image
            tl, br = (box[0], box[1]), (box[2], box[3])
            cv2.rectangle(img, tl, br, (0,255,0), 2)
            cv2.putText(
                img, txt,
                (tl[0], tl[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255,0,0), 2
            )

        # Save annotated image
        os.makedirs('static', exist_ok=True)
        cv2.imwrite('static/output.jpg', img)
        output_image = 'output.jpg'

    return render_template(
        'index.html',
        extracted_text=extracted_text,
        translated_text=translated_text,
        output_image=output_image
    )

if __name__ == '__main__':
    app.run(debug=True)
