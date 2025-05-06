from flask import Flask, request, render_template
import easyocr
import numpy as np
import cv2
import os
from deep_translator import GoogleTranslator

app = Flask(__name__)

# Initialize EasyOCR readers for English+Hindi and English+Chinese
reader_hi = easyocr.Reader(['en', 'hi'], gpu=False)
reader_ch = easyocr.Reader(['en', 'ch_sim'], gpu=False)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    extracted_text = []
    translated_text = []
    output_image = None

    if request.method == 'POST':
        file = request.files.get('image')
        if not file:
            return render_template('index.html',
                                   extracted_text=['No file selected'],
                                   translated_text=[],
                                   output_image=None)

        # Load image into OpenCV
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Run OCR with both readers
        results_hi = reader_hi.readtext(img)
        results_ch = reader_ch.readtext(img)
        results = results_hi + results_ch

        # --- STEP 1: compute each box height and vertical centre ---
        heights = []
        for bbox, text, conf in results:
            (x0, y0), _, (x2, y2), _ = bbox
            heights.append(abs(y2 - y0))
        avg_h = np.mean(heights) if heights else 1

        # --- STEP 2: define a sort key that clusters into rows, then x-order ---
        def sort_key(item):
            bbox, text, conf = item
            (x0, y0), _, (x2, y2), _ = bbox
            cy = (y0 + y2) / 2
            # cluster into line bins
            line_idx = int(cy // (avg_h * 0.8))
            return (line_idx, x0)

        # sort top-to-bottom by line, then left-to-right
        results = sorted(results, key=sort_key)

        # --- STEP 3: filter, translate, draw boxes & accumulate text ---
        for bbox, text, confidence in results:
            if confidence < 0.4:
                continue

            extracted_text.append(text)

            # detect Hindi or Chinese glyphs
            if any('\u0900' <= c <= '\u097F' for c in text):
                lang = 'hi'
            elif any('\u4e00' <= c <= '\u9fff' for c in text):
                lang = 'zh'
            else:
                lang = 'en'

            if lang != 'en':
                translated = GoogleTranslator(source=lang, target='en').translate(text)
                translated_text.append(translated)
            else:
                translated_text.append(text)

            # draw bounding box + original text
            (x0, y0), _, (x2, y2), _ = bbox
            tl = (int(x0), int(y0))
            br = (int(x2), int(y2))
            cv2.rectangle(img, tl, br, (0, 255, 0), 2)
            cv2.putText(img, text, (tl[0], tl[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # save annotated image
        os.makedirs('static', exist_ok=True)
        output_path = 'static/output.jpg'
        cv2.imwrite(output_path, img)
        output_image = 'output.jpg'

    return render_template('index.html',
                           extracted_text=extracted_text,
                           translated_text=translated_text,
                           output_image=output_image)

if __name__ == '__main__':
    app.run(debug=True)
