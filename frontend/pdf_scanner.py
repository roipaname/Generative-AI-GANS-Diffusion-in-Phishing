"""
PDF Document Scanner Module
Extracts text, URLs, and images from PDF files
"""

import PyPDF2
import re
from typing import List
import pdfplumber
from PIL import Image

class PDFScanner:
    def __init__(self):
        """Initialize URL regex pattern"""
        self.url_pattern = re.compile(
            r'https?://(?:[a-zA-Z0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )

    def extract_text_pdfplumber(self, pdf_file) -> str:
        """Extract text using pdfplumber"""
        try:
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
        except Exception as e:
            print(f"❌ pdfplumber extraction error: {e}")
            return ""

    def extract_text_pypdf2(self, pdf_file) -> str:
        """Fallback text extraction using PyPDF2"""
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text.strip()
        except Exception as e:
            print(f"❌ PyPDF2 extraction error: {e}")
            return ""

    def extract_urls(self, text: str) -> List[str]:
        """Extract unique URLs from text"""
        urls = self.url_pattern.findall(text)
        return list(set(urls))

    def extract_images(self, pdf_file) -> List[Image.Image]:
        """Extract images from PDF pages"""
        images = []
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    for img in page.images:
                        try:
                            cropped = page.within_bbox(
                                (img['x0'], img['top'], img['x1'], img['bottom'])
                            ).to_image(resolution=150)
                            images.append(cropped)
                        except Exception:
                            continue
        except Exception as e:
            print(f"❌ Image extraction error: {e}")
        return images

    def scan_pdf(self, pdf_file) -> dict:
        """Extract text, URLs, and images from a PDF"""
        pdf_file.seek(0)
        text = self.extract_text_pdfplumber(pdf_file)
        if not text:
            pdf_file.seek(0)
            text = self.extract_text_pypdf2(pdf_file)

        urls = self.extract_urls(text)
        pdf_file.seek(0)
        images = self.extract_images(pdf_file)

        return {
            'text': text,
            'urls': urls,
            'images': images,
            'url_count': len(urls),
            'image_count': len(images)
        }


# === Example Usage ===
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        with open(pdf_path, "rb") as f:
            scanner = PDFScanner()
            result = scanner.scan_pdf(f)

        print(f"Text length: {len(result['text'])} characters")
        print(f"URLs found ({result['url_count']}): {result['urls']}")
        print(f"Images extracted: {result['image_count']}")
    else:
        print("Usage: python pdf_scanner.py <path_to_pdf>")
