from flask import Flask, request, render_template, url_for
import easyocr
import numpy as np
import cv2
import os
from deep_translator import GoogleTranslator
app = Flask(__name__)
# Initialize EasyOCR Reader (multilingual support)
reader_hi = easyocr.Reader(['en', 'hi'])
reader_ch = easyocr.Reader(['en', 'ch_sim'])
# translator = Translator()
@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    extracted_text = []
    translated_text = []
    output_image = None

    if request.method == 'POST':
        file = request.files['image']
        if not file:
            return render_template('index.html', extracted_text=['No file selected'])

        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # OCR with both models
        results_hi = reader_hi.readtext(img)
        results_ch = reader_ch.readtext(img)

        # Merge results
        results = results_hi + results_ch

        for (bbox, text, confidence) in results:
            if confidence > 0.4:
                extracted_text.append(text)

                if any('\u0900' <= c <= '\u097F' for c in text):  # Hindi
                    lang_detected = 'hi'
                elif any('\u4e00' <= c <= '\u9fff' for c in text):  # Chinese
                    lang_detected = 'zh'
                else:
                    lang_detected = 'en'

                if lang_detected != 'en':
                    translated = GoogleTranslator(source=lang_detected, target='en').translate(text)
                    translated_text.append(translated)
                else:
                    translated_text.append(text)

                # Draw bounding box
                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))
                img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
                img = cv2.putText(img, text, (top_left[0], top_left[1] - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        output_path = 'static/output.jpg'
        os.makedirs('static', exist_ok=True)
        cv2.imwrite(output_path, img)
    print(translated_text)
    return render_template('index.html', extracted_text=extracted_text, translated_text=translated_text, output_image='output.jpg')
if __name__ == '__main__':
    app.run(debug=True)