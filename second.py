import fitz
import torch
from PIL import Image, ImageDraw, ImageFont
from sentence_transformers import SentenceTransformer
import numpy as np
import io
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import clip

@dataclass
class SpatialRelationship:
    """Represents spatial relationship between text and image"""
    distance: float
    relative_x: str  # 'left', 'right', 'overlapping'
    relative_y: str  # 'above', 'below', 'overlapping'
    vertical_overlap: float  # 0-1, how much they overlap vertically
    horizontal_overlap: float  # 0-1, how much they overlap horizontally
    is_in_same_section: bool  # whether image is in same vertical section as text

class EnhancedPDFImageExtractor:
    def __init__(self, model_name='all-MiniLM-L6-v2', use_clip=False, debug_mode=False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.text_model = SentenceTransformer(model_name, device=self.device)
        self.debug_mode = debug_mode
        self.use_clip = use_clip
        
        # Initialize CLIP if requested
        if use_clip:
            try:
                import clip
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
                print("CLIP model loaded successfully")
            except ImportError:
                print("Warning: CLIP not available. Install with: pip install openai-clip-torch")
                self.use_clip = False
        
        # Configuration parameters
        self.min_image_size = 50  # Minimum width/height in pixels
        self.position_weight = 0.4  # Weight for positional scoring
        self.size_weight = 0.3  # Weight for image size scoring
        self.distance_weight = 0.3  # Weight for distance scoring

    def extract_text_blocks(self, page):
        """Enhanced text block extraction with better bbox handling"""
        blocks = page.get_text("dict")['blocks']
        result = []
        
        for block in blocks:
            if "lines" in block:
                for line in block['lines']:
                    spans_with_text = [span for span in line['spans'] if span['text'].strip()]
                    if not spans_with_text:
                        continue
                        
                    text = " ".join([span['text'] for span in spans_with_text]).strip()
                    if text:
                        # Calculate accurate line bbox
                        all_bboxes = [span['bbox'] for span in spans_with_text]
                        x0 = min(b[0] for b in all_bboxes)
                        y0 = min(b[1] for b in all_bboxes)
                        x1 = max(b[2] for b in all_bboxes)
                        y1 = max(b[3] for b in all_bboxes)
                        
                        # Get font information for adaptive padding later
                        avg_font_size = np.mean([span.get('size', 12) for span in spans_with_text])
                        
                        result.append({
                            "text": text,
                            "bbox": (x0, y0, x1, y1),
                            "font_size": avg_font_size,
                            "height": y1 - y0
                        })
        return result

    def extract_images_with_enhanced_filtering(self, doc, page):
        """Extract images with size filtering and enhanced metadata"""
        images = []
        page_rect = page.rect
        
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            img_rects = page.get_image_rects(xref)
            
            if not img_rects:
                continue
            
            # Use the largest rectangle if multiple exist
            largest_rect = max(img_rects, key=lambda r: (r.width * r.height))
            img_bbox = tuple(largest_rect)
            
            # Filter out small images (icons, logos, decorative elements)
            if largest_rect.width < self.min_image_size or largest_rect.height < self.min_image_size:
                if self.debug_mode:
                    print(f"Filtered out small image: {largest_rect.width}x{largest_rect.height}")
                continue
            
            try:
                pix = fitz.Pixmap(doc, xref)
                
                # Handle alpha channel/masks
                if img_info[1] > 0:
                    try:
                        mask = fitz.Pixmap(doc, img_info[1])
                        pix = fitz.Pixmap(pix, mask)
                        mask = None
                    except:
                        pass
                
                # Convert to RGB
                if pix.n > 4 or pix.colorspace != fitz.csRGB:
                    temp_pix = fitz.Pixmap(fitz.csRGB, pix)
                    pix = None
                    pix = temp_pix
                
                img_bytes = pix.tobytes("png")
                img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                
                images.append({
                    'image': img_pil,
                    'bbox': img_bbox,
                    'xref': xref,
                    'width': largest_rect.width,
                    'height': largest_rect.height,
                    'area': largest_rect.width * largest_rect.height,
                    'aspect_ratio': largest_rect.width / largest_rect.height
                })
                
                pix = None
                
            except Exception as e:
                if self.debug_mode:
                    print(f"Warning: Could not extract image {xref}: {e}")
                continue
        
        return images

    def analyze_spatial_relationship(self, text_bbox, img_bbox) -> SpatialRelationship:
        """Analyze detailed spatial relationship between text and image"""
        if not text_bbox or not img_bbox:
            return SpatialRelationship(
                distance=float('inf'),
                relative_x='unknown',
                relative_y='unknown',
                vertical_overlap=0,
                horizontal_overlap=0,
                is_in_same_section=False
            )
        
        # Unpack bounding boxes
        tx0, ty0, tx1, ty1 = text_bbox
        ix0, iy0, ix1, iy1 = img_bbox
        
        # Calculate centers
        text_center_x = (tx0 + tx1) / 2
        text_center_y = (ty0 + ty1) / 2
        img_center_x = (ix0 + ix1) / 2
        img_center_y = (iy0 + iy1) / 2
        
        # Basic distance
        distance = np.hypot(img_center_x - text_center_x, img_center_y - text_center_y)
        
        # Relative positions
        if ix1 < tx0:
            relative_x = 'left'
        elif ix0 > tx1:
            relative_x = 'right'
        else:
            relative_x = 'overlapping'
        
        if iy1 < ty0:
            relative_y = 'above'
        elif iy0 > ty1:
            relative_y = 'below'
        else:
            relative_y = 'overlapping'
        
        # Calculate overlap ratios
        # Vertical overlap
        v_overlap_start = max(ty0, iy0)
        v_overlap_end = min(ty1, iy1)
        vertical_overlap = max(0, v_overlap_end - v_overlap_start) / (ty1 - ty0)
        
        # Horizontal overlap
        h_overlap_start = max(tx0, ix0)
        h_overlap_end = min(tx1, ix1)
        horizontal_overlap = max(0, h_overlap_end - h_overlap_start) / (tx1 - tx0)
        
        # Check if in same vertical section (extended text height)
        text_height = ty1 - ty0
        extended_text_range = (ty0 - text_height, ty1 + text_height * 2)  # Allow below text
        is_in_same_section = not (iy1 < extended_text_range[0] or iy0 > extended_text_range[1])
        
        return SpatialRelationship(
            distance=distance,
            relative_x=relative_x,
            relative_y=relative_y,
            vertical_overlap=vertical_overlap,
            horizontal_overlap=horizontal_overlap,
            is_in_same_section=is_in_same_section
        )

    def score_image_relevance(self, img_data, text_bbox, spatial_rel: SpatialRelationship) -> float:
        """Score image relevance based on position, size, and spatial relationship"""
        base_score = 1000  # Start with high base score
        
        # Position scoring - prioritize images below or beside text
        position_score = 0
        if spatial_rel.relative_y == 'below':
            position_score = 100  # Strong preference for images below text
        elif spatial_rel.relative_y == 'overlapping' and spatial_rel.vertical_overlap > 0.3:
            position_score = 80  # Good if significant vertical overlap
        elif spatial_rel.relative_x in ['left', 'right'] and spatial_rel.vertical_overlap > 0.5:
            position_score = 70  # Side-by-side with good alignment
        elif spatial_rel.relative_y == 'above':
            position_score = -50  # Penalize images above text (unless very close)
            if spatial_rel.distance < 50:  # Very close images above might still be relevant
                position_score = 20
        
        # Same section bonus
        if spatial_rel.is_in_same_section:
            position_score += 30
        
        # Size scoring - prefer larger images
        area_score = min(img_data['area'] / 10000, 100)  # Cap at 100 points
        
        # Aspect ratio bonus - avoid very thin or very wide images
        aspect_ratio = img_data['aspect_ratio']
        aspect_score = 20 if 0.2 <= aspect_ratio <= 5.0 else -10
        
        # Distance penalty (closer is better, but position matters more)
        distance_penalty = min(spatial_rel.distance / 10, 100)
        
        # Vertical overlap bonus
        overlap_bonus = spatial_rel.vertical_overlap * 40
        
        total_score = (base_score + 
                      position_score * self.position_weight * 100 +
                      area_score * self.size_weight +
                      overlap_bonus -
                      distance_penalty * self.distance_weight)
        
        if self.debug_mode:
            print(f"  Image scoring breakdown:")
            print(f"    Position: {spatial_rel.relative_y}/{spatial_rel.relative_x} = {position_score}")
            print(f"    Area: {img_data['area']:.0f} = {area_score:.1f}")
            print(f"    Distance penalty: {spatial_rel.distance:.1f} = -{distance_penalty:.1f}")
            print(f"    Overlap bonus: {spatial_rel.vertical_overlap:.2f} = {overlap_bonus:.1f}")
            print(f"    Total score: {total_score:.1f}")
        
        return total_score

    def clip_score_image(self, image, query_text):
        """Score image using CLIP model for semantic similarity"""
        if not self.use_clip:
            return 0
        
        try:
            # Preprocess image
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenize text
            text_input = clip.tokenize([query_text]).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_input)
                
                # Calculate similarity
                similarity = torch.cosine_similarity(image_features, text_features).item()
                
            return similarity * 100  # Scale to 0-100
            
        except Exception as e:
            if self.debug_mode:
                print(f"CLIP scoring failed: {e}")
            return 0

    def find_best_text_match(self, text_blocks, query):
        """Find best matching text block using semantic similarity"""
        def normalize(text):
            return " ".join(text.lower().split())
        
        if not text_blocks:
            return None, -1
        
        block_texts = [normalize(tb['text']) for tb in text_blocks]
        block_embeds = self.text_model.encode(block_texts)
        query_embed = self.text_model.encode([normalize(query)])
        sims = np.dot(block_embeds, query_embed.T).squeeze()
        sims = np.atleast_1d(sims)  # Ensure sims is always at least 1D
        best_idx = int(np.argmax(sims))
        return text_blocks[best_idx], sims[best_idx]

    def create_debug_visualization(self, page, text_bbox, images, selected_image_idx=None, 
                                 output_path="debug_visualization.png", dpi=150):
        """Create debug visualization showing bounding boxes"""
        if not self.debug_mode:
            return None
        
        # Render page to image
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        
        # Create drawing context
        draw = ImageDraw.Draw(img)
        scale = dpi / 72.0
        
        # Draw text bbox in red
        if text_bbox:
            scaled_text_bbox = [int(coord * scale) for coord in text_bbox]
            draw.rectangle(scaled_text_bbox, outline='red', width=3)
            draw.text((scaled_text_bbox[0], scaled_text_bbox[1] - 20), 
                     "MATCHED TEXT", fill='red')
        
        # Draw image bboxes
        for idx, img_data in enumerate(images):
            if img_data['bbox']:
                scaled_img_bbox = [int(coord * scale) for coord in img_data['bbox']]
                
                if idx == selected_image_idx:
                    # Selected image in green
                    draw.rectangle(scaled_img_bbox, outline='green', width=4)
                    draw.text((scaled_img_bbox[0], scaled_img_bbox[1] - 20), 
                             f"SELECTED IMG {idx}", fill='green')
                else:
                    # Other images in blue
                    draw.rectangle(scaled_img_bbox, outline='blue', width=2)
                    draw.text((scaled_img_bbox[0], scaled_img_bbox[1] - 15), 
                             f"IMG {idx}", fill='blue')
        
        img.save(output_path)
        print(f"Debug visualization saved: {output_path}")
        pix = None
        return img

    def adaptive_crop_page(self, page, bbox, base_expand=50, dpi=300):
        """Crop page with adaptive padding based on text characteristics"""
        if not bbox:
            return None
        
        text_height = bbox[3] - bbox[1]
        text_width = bbox[2] - bbox[0]
        
        # Adaptive padding based on text size
        vertical_padding = max(base_expand, text_height * 0.5)
        horizontal_padding = max(base_expand, text_width * 0.2)
        
        # Create expanded rectangle
        expanded_bbox = (
            max(bbox[0] - horizontal_padding, 0),
            max(bbox[1] - vertical_padding, 0),
            min(bbox[2] + horizontal_padding, page.rect.width),
            min(bbox[3] + vertical_padding * 2, page.rect.height)  # More padding below
        )
        
        rect = fitz.Rect(expanded_bbox)
        mat = fitz.Matrix(dpi/72, dpi/72)
        
        # Crop directly to avoid memory issues
        pix = page.get_pixmap(matrix=mat, clip=rect)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        if self.debug_mode:
            print(f"Adaptive crop: text_size={text_width:.1f}x{text_height:.1f}, "
                  f"padding={horizontal_padding:.1f}x{vertical_padding:.1f}")
        
        pix = None
        return img

    def find_image_for_query(self, pdf_path, query, output_image_path="nearest_image.png", 
                           debug_output_path="debug_visualization.png"):
        """Main method to find best image for query with enhanced logic"""
        doc = fitz.open(pdf_path)
        
        print(f"Searching for: '{query}'")
        print(f"Processing {len(doc)} pages...")
        
        # Step 1: Find best text match across all pages
        best_match = None
        best_score = -np.inf
        best_page_no = None
        best_page = None
        
        for page_no, page in enumerate(doc):
            text_blocks = self.extract_text_blocks(page)
            match, score = self.find_best_text_match(text_blocks, query)
            
            if match and score > best_score:
                best_match = match
                best_score = score
                best_page_no = page_no
                best_page = page
        
        if not best_match:
            print("No relevant text found for query.")
            doc.close()
            return None
        
        print(f"\nBest text match (Page {best_page_no + 1}): '{best_match['text'][:80]}...'")
        print(f"Text match score: {best_score:.3f}")
        
        # Step 2: Extract and analyze images on the same page
        images = self.extract_images_with_enhanced_filtering(doc, best_page)
        
        if not images:
            print("No suitable images found. Using adaptive cropping...")
            cropped_img = self.adaptive_crop_page(best_page, best_match['bbox'])
            if cropped_img:
                cropped_img.save(output_image_path)
                print(f"Saved cropped region: {output_image_path}")
            doc.close()
            return cropped_img
        
        print(f"Found {len(images)} suitable images on page {best_page_no + 1}")
        
        # Step 3: Score all images based on enhanced criteria
        image_scores = []
        spatial_relationships = []
        
        for idx, img_data in enumerate(images):
            spatial_rel = self.analyze_spatial_relationship(best_match['bbox'], img_data['bbox'])
            spatial_relationships.append(spatial_rel)
            
            # Base relevance score
            relevance_score = self.score_image_relevance(img_data, best_match['bbox'], spatial_rel)
            
            # Add CLIP score if available
            clip_score = 0
            if self.use_clip:
                clip_score = self.clip_score_image(img_data['image'], query)
                relevance_score += clip_score * 0.2  # 20% weight for CLIP
            
            image_scores.append(relevance_score)
            
            if self.debug_mode:
                print(f"\nImage {idx} (xref: {img_data['xref']}):")
                print(f"  Size: {img_data['width']:.0f}x{img_data['height']:.0f}")
                print(f"  Position: {spatial_rel.relative_y} and {spatial_rel.relative_x}")
                print(f"  Distance: {spatial_rel.distance:.1f}")
                print(f"  Vertical overlap: {spatial_rel.vertical_overlap:.2f}")
                if self.use_clip:
                    print(f"  CLIP score: {clip_score:.1f}")
                print(f"  Total score: {relevance_score:.1f}")
        
        # Step 4: Select best image
        if not image_scores:
            print("No images could be scored.")
            doc.close()
            return None
        
        best_img_idx = int(np.argmax(image_scores))
        best_img_data = images[best_img_idx]
        best_spatial_rel = spatial_relationships[best_img_idx]
        
        print(f"\nSelected image {best_img_idx} with score: {image_scores[best_img_idx]:.1f}")
        print(f"Image position: {best_spatial_rel.relative_y} and {best_spatial_rel.relative_x} of text")
        print(f"Distance from text: {best_spatial_rel.distance:.1f}")
        
        # Step 5: Create debug visualization if enabled
        if self.debug_mode:
            self.create_debug_visualization(
                best_page, best_match['bbox'], images, best_img_idx, debug_output_path
            )
        
        # Step 6: Save the selected image
        best_img_data['image'].save(output_image_path)
        print(f"Saved image: {output_image_path}")
        
        doc.close()
        return best_img_data['image']


if __name__ == "__main__":
    pdf_path = r"D:\image_extraction-\unit_67ad9a036de7f1459b25a40f.pdf"
    query = "phone"
    
    # ----- CLIP -----
    extractor = EnhancedPDFImageExtractor(
        debug_mode=True,  
        use_clip=False    
    )
    
    img = extractor.find_image_for_query(
        pdf_path, 
        query,
        output_image_path="extracted_image.png",
        debug_output_path="debug_page_visualization.png"
    )
    
    if img is not None:
        print(f"\nImage extraction complete! Image size: {img.size}")
    else:
        print("No suitable image found for your query.")