import fitz
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
import io
import os

class PDFNearestImageExtractor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.text_model = SentenceTransformer(model_name, device=self.device)

    def extract_text_blocks(self, page):
        blocks = page.get_text("dict")['blocks']
        result = []
        for block in blocks:
            if "lines" in block:
                for line in block['lines']:
                    text = " ".join([span['text'] for span in line['spans']]).strip()
                    if text:
                        # Get the bounding box for the line
                        bbox = [span['bbox'] for span in line['spans']]
                        # Merge all span bboxes into one for the line
                        if bbox:
                            x0 = min(b[0] for b in bbox)
                            y0 = min(b[1] for b in bbox)
                            x1 = max(b[2] for b in bbox)
                            y1 = max(b[3] for b in bbox)
                            line_bbox = (x0, y0, x1, y1)
                            result.append({"text": text, "bbox": line_bbox})
        return result

    def extract_images_with_bbox(self, doc, page):
        images = []
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            img_rects = page.get_image_rects(xref)
            img_bbox = img_rects[0] if img_rects else None
            pix = fitz.Pixmap(doc, xref)
            # Handle alpha channel/masks
            if img_info[1] > 0:
                mask = fitz.Pixmap(doc, img_info[1])
                pix = fitz.Pixmap(pix, mask)
            # Convert to supported color space if necessary (always use RGB)
            if pix.n > 4 or pix.colorspace is not fitz.csRGB:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_bytes = pix.tobytes("png")
            img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            images.append({'image': img_pil, 'bbox': img_bbox, 'xref': xref})
            pix = None
        return images

    def find_best_text_match(self, text_blocks, query):
        def normalize(text):
            return " ".join(text.lower().split())
        block_texts = [normalize(tb['text']) for tb in text_blocks]
        if not block_texts:
            return None, None
        block_embeds = self.text_model.encode(block_texts)
        query_embed = self.text_model.encode([normalize(query)])
        sims = np.dot(block_embeds, query_embed.T).squeeze()
        best_idx = int(np.argmax(sims))
        return text_blocks[best_idx], sims[best_idx]

    def get_distance(self, bbox1, bbox2):
        if not bbox1 or not bbox2:
            return float('inf')
        x1 = (bbox1[0] + bbox1[2]) / 2
        y1 = (bbox1[1] + bbox1[3]) / 2
        x2 = (bbox2[0] + bbox2[2]) / 2
        y2 = (bbox2[1] + bbox2[3]) / 2
        return np.hypot(x2 - x1, y2 - y1)

    def find_nearest_images(self, images, ref_bbox, max_distance=100):
        if not images:
            return []
        distances = [self.get_distance(img['bbox'], ref_bbox) for img in images]
        # Get all images within max_distance
        result = [(img, dist) for img, dist in zip(images, distances) if dist < max_distance]
        # Sort by distance
        result.sort(key=lambda x: x[1])
        return result

    def find_nearest_image(self, images, ref_bbox):
        if not images:
            return None, float('inf')
        distances = [self.get_distance(img['bbox'], ref_bbox) for img in images]
        min_idx = int(np.argmin(distances))
        return images[min_idx] if distances[min_idx] != float('inf') else None, distances[min_idx]

    def crop_page_to_bbox(self, page, bbox, expand=50, dpi=200):
        # Render full page to image, then crop around the bbox with padding
        rect = fitz.Rect(bbox)
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        full_img = Image.open(io.BytesIO(pix.tobytes("png")))
        # Convert PDF points to pixels
        scale = dpi / 72.0
        x0, y0, x1, y1 = [int(v * scale) for v in rect]
        # Add padding but keep within image
        x0 = max(x0 - expand, 0)
        y0 = max(y0 - expand, 0)
        x1 = min(x1 + expand, full_img.size[0])
        y1 = min(y1 + expand, full_img.size[1])
        return full_img.crop((x0, y0, x1, y1))

    def find_image_for_query(self, pdf_path, query, output_image_path="nearest_image.png"):
        doc = fitz.open(pdf_path)
        # Step 1: Find best-matching text block in all pages
        best_match = None
        best_score = -np.inf
        best_page_no = None
        best_bbox = None
        for pno, page in enumerate(doc):
            text_blocks = self.extract_text_blocks(page)
            match, score = self.find_best_text_match(text_blocks, query)
            if match and score > best_score:
                best_match, best_score = match, score
                best_page_no, best_bbox = pno, match['bbox']
        if not best_match:
            print("No relevant text found for query.")
            return None

        print(f"\nBest text match (Page {best_page_no+1}): {best_match['text'][:80]}...  (score: {best_score:.3f})")

        # Step 2: On the same page, find all valid images with bbox, and compute spatial distance
        images = self.extract_images_with_bbox(doc, doc[best_page_no])
        if images:
            print(f"Detected {len(images)} images on page {best_page_no+1}")
            for idx, img in enumerate(images):
                print(f"  Image {idx}: bbox={img['bbox']}, xref={img['xref']}")
            nearest_images = self.find_nearest_images(images, best_bbox, max_distance=150)
            if nearest_images:
                # Save the closest one, or all if you want
                img_obj, dist = nearest_images[0]
                print(f"Found image near text (distance: {dist:.1f}) [xref: {img_obj['xref']}]")
                img_obj['image'].save(output_image_path)
                print(f"Saved image: {output_image_path}")
                return img_obj['image']
            else:
                # Images exist, but none are close. Return the closest anyway.
                # This prevents fallback to cropping text.
                img_obj, dist = min(
                    ((img, self.get_distance(img['bbox'], best_bbox)) for img in images),
                    key=lambda x: x[1]
                )
                print(f"No image within threshold, but returning closest image (distance: {dist:.1f}) [xref: {img_obj['xref']}]")
                img_obj['image'].save(output_image_path)
                print(f"Saved image: {output_image_path}")
                return img_obj['image']
        else:
            # No images detected at all, fallback to cropping
            print("No images detected on the page. Cropping page as fallback...")
            page = doc[best_page_no]
            cropped = self.crop_page_to_bbox(page, best_bbox, expand=50, dpi=200)
            cropped.save(output_image_path)
            print(f"Saved cropped text region as image: {output_image_path}")
            return cropped


if __name__ == "__main__":
    pdf_path = r"C:\Users\Netcom\Desktop\img ext\1753960689184-[Luxrobo] ì_¸ë__ êµ_ì_¡ì_© Lv2 - Jan 26th, 2024.pdf"
    query = "Battery"
    extractor = PDFNearestImageExtractor()  
    img = extractor.find_image_for_query(pdf_path, query)   
    if img is not None:
        print("Image extraction complete.")
    else:
        print("No image found for your query.")
