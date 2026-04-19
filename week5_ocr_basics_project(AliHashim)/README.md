# Week 5 OCR Basics Submission

This folder contains a completed starter submission for the Week 5 lab on OCR and document intelligence.

## Included files
- `week5_ocr_basics.ipynb` - completed notebook with results and observations
- `ocr_pipeline.py` - reusable preprocessing and OCR comparison functions
- `receipt_parser.py` - regex-based parser for merchant, date, and total
- `receipts/` - 5 generated receipt images plus `ground_truth.json`
- `outputs/ocr_results.csv` - OCR comparison table
- `outputs/sample_ocr_outputs.json` - raw OCR text samples
- `outputs/summary.json` - aggregate results

## Deliverables covered
- Tesseract OCR extraction
- Confidence scores
- EasyOCR code path included
- Grayscale + Gaussian blur + thresholding
- Before/after OCR comparison
- Receipt parser
- 5 receipts processed with results

## Aggregate summary
- Receipts processed: 5
- Avg. Tesseract confidence (original): 94.65
- Avg. Tesseract confidence (preprocessed): 85.34
- Avg. character improvement after preprocessing: -64.4
- Totals extracted: 1 / 5
- Dates extracted: 4 / 5

## Important note
EasyOCR could not be executed in this container because of a Torch/Torchvision compatibility issue. The notebook and pipeline still include EasyOCR code so you can run it in Kaggle/Colab, which the assignment explicitly allows.
