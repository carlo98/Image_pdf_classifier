# Image_pdf_classifier
Use case: Classify 4 different types of document scanned and saved as pdf. Each file can be oriented at 0, 90°, 180° or 270° degrees.

This solution extracts an image from the pdf file and, through a CNN (model_1.h5), it rotates the newly created image; then it extracts the text using pytesseract and searches for some of the most important words of each file.

Words used to descriminate each file can be set in Classifier/docClassifier.py.
