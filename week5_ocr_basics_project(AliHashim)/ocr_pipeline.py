
from pathlib import Path
import cv2
import pytesseract
from PIL import Image
from receipt_parser import parse_receipt

try:
    import easyocr
    _easyocr_error = ''
    _reader = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    _reader = None
    _easyocr_error = str(e)

def preprocess_image(image_path):
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return img, gray, denoised, adaptive

def compare_ocr_methods(image_path):
    img, gray, denoised, adaptive = preprocess_image(image_path)
    tess_original = pytesseract.image_to_string(Image.open(image_path))
    tess_preprocessed = pytesseract.image_to_string(Image.fromarray(adaptive))
    if _reader is None:
        easy_text = f'[EasyOCR unavailable in this environment: {_easyocr_error}]'
    else:
        easy_result = _reader.readtext(str(image_path))
        easy_text = ' '.join([r[1] for r in easy_result])
    return {
        'tesseract_original': tess_original,
        'tesseract_preprocessed': tess_preprocessed,
        'easyocr': easy_text,
        'parsed_preprocessed': parse_receipt(tess_preprocessed),
    }

if __name__ == '__main__':
    sample_dir = Path('receipts')
    for path in sorted(sample_dir.glob('*.jpg')):
        print(f'--- {path.name} ---')
        print(compare_ocr_methods(path)['parsed_preprocessed'])
