"""OFD Document Backend with Multiprocessing Support.

This backend uses the `multiprocessing` module to bypass the GIL and achieve
true parallelism for OFD parsing.
"""

from __future__ import annotations

import logging
import zipfile
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from docling_core.types.doc import DocItemLabel, DoclingDocument, DocumentOrigin

from .ofd_parser import (
    OFDParserError,
    OFDZipParser,
    OFDPage,
    OFDTextBlock,
    OFDFontResource,
)

try:
    from .ofd_glyph_decoder import OFDGlyphDecoder
except ImportError:
    OFDGlyphDecoder = None

_log = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from fontTools.ttLib import TTFont
    FONTTOOLS_AVAILABLE = True
except ImportError:
    FONTTOOLS_AVAILABLE = False

try:
    from rapidocr_onnxruntime import RapidOCR
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


@dataclass
class _TextItem:
    text: str
    x: float
    y: float
    width: float
    height: float
    font_size: float


@dataclass
class _PageResult:
    """Result of processing a single page."""
    page_index: int
    text_items: List[_TextItem]
    stats: Dict[str, int]
    image_count: int
    error: Optional[str] = None


# --- Worker Function (Must be top-level for pickling) ---

def _process_page_worker(
    ofd_path: str,
    page_data: bytes,  # Pass page XML content directly to avoid re-reading if possible, or just page index/path
    page_index: int,
    page_id: Optional[str],
    physical_box: Tuple[float, float, float, float],
    text_blocks_data: List[Dict[str, Any]], # Serialized text blocks
    image_objects_count: int,
    font_paths: Dict[str, str],
    is_bytes_io: bool = False,
    file_content: Optional[bytes] = None
) -> _PageResult:
    """Worker function to process a single page in a separate process.
    
    Args:
        ofd_path: Path to the OFD file (if file-based)
        page_data: Raw XML data of the page (not used currently, we reconstruct objects)
        page_index: Index of the page
        page_id: ID of the page
        physical_box: Page boundary
        text_blocks_data: List of dictionaries containing text block data
        image_objects_count: Number of images
        font_paths: Map of font IDs to internal paths
        is_bytes_io: Whether the source was BytesIO
        file_content: Raw file content if source was BytesIO
    """
    
    # Reconstruct minimal context needed for processing
    # We need to open the zip file to read fonts
    
    stats: Dict[str, int] = {
        "total_blocks": 0,
        "processed_blocks": 0,
        "raw_pua_blocks": 0,
        "remaining_pua_blocks": 0,
        "filtered_pua_chars": 0,
        "used_glyph_decoder": 0,
        "used_ocr": 0,
    }
    
    items: List[_TextItem] = []
    
    try:
        # Open ZipFile
        if is_bytes_io:
            if file_content is None:
                return _PageResult(page_index, [], {}, 0, "Missing file content for BytesIO")
            f = BytesIO(file_content)
            ofd_zip = zipfile.ZipFile(f, "r")
        else:
            ofd_zip = zipfile.ZipFile(ofd_path, "r")
            
        # Initialize Glyph Decoder
        glyph_decoder = None
        if FONTTOOLS_AVAILABLE and font_paths:
            try:
                # We need to reconstruct the font_paths dict expected by OFDGlyphDecoder
                # The worker receives {font_id: internal_path}
                # OFDGlyphDecoder expects {font_id: internal_path} (optional wrapper)
                glyph_decoder = OFDGlyphDecoder(ofd_zip, font_paths)
            except Exception as e:
                # Just log/ignore, continue without decoder
                pass
                
        # Initialize OCR (lazy load)
        ocr_engine = None
        
        def get_ocr_engine():
            nonlocal ocr_engine
            if OCR_AVAILABLE and ocr_engine is None:
                try:
                    ocr_engine = RapidOCR()
                except Exception:
                    pass
            return ocr_engine

        # Helper methods (duplicated from backend to be standalone)
        def _is_pua_char(char: str) -> bool:
            return 0xE000 <= ord(char) <= 0xF8FF

        def _has_pua_characters(text: str) -> bool:
            return any(_is_pua_char(ch) for ch in text)

        def _clean_text_with_pua(text: str) -> str:
            return "".join(c for c in text if not _is_pua_char(c)).strip()

        def _count_pua_chars(text: str) -> int:
            return sum(1 for c in text if _is_pua_char(c))
            
        def _estimate_text_width(text: str, font_size: float) -> float:
            length = max(len(text), 1)
            return font_size * 0.6 * length

        # Font loading helper
        font_cache = {}
        def _load_font_bytes(font_id: str) -> Optional[bytes]:
            if font_id in font_cache:
                return font_cache[font_id]
            
            path = font_paths.get(font_id)
            if not path or path not in ofd_zip.namelist():
                return None
                
            try:
                data = ofd_zip.read(path)
                font_cache[font_id] = data
                return data
            except Exception:
                return None

        # OCR Helper
        def _ocr_decode_text(text: str, font_id: str, font_size: float) -> Optional[str]:
            engine = get_ocr_engine()
            if not engine:
                return None
                
            font_bytes = _load_font_bytes(font_id) if font_id else None
            if not font_bytes:
                return None
                
            try:
                from PIL import Image, ImageDraw, ImageFont
                import numpy as np
            except ImportError:
                return None
                
            # Simplified OCR logic for worker
            # ... (Copying the core OCR logic from backend) ...
            
            def run_ocr_segment(segment_text: str) -> str:
                # Simplified version of the complex logic in backend
                # For brevity, implementing the core flow
                pixel_size = max(int(font_size * 5.5), 48)
                try:
                    font = ImageFont.truetype(BytesIO(font_bytes), pixel_size)
                except Exception:
                    return ""
                    
                try:
                    width, height = font.getsize(segment_text)
                except AttributeError: # Pillow >= 10
                    left, top, right, bottom = font.getbbox(segment_text)
                    width = right - left
                    height = bottom - top
                    
                if width <= 0 or height <= 0:
                    return ""
                    
                margin = 32
                img = Image.new("L", (width + margin, height + margin), color=255)
                draw = ImageDraw.Draw(img)
                draw.text((margin//2, margin//2), segment_text, fill=0, font=font)
                
                try:
                    result, _ = engine(np.array(img))
                    if result:
                        # Join results
                        return "".join([line[1] for line in result])
                except Exception:
                    pass
                return ""

            # Simple PUA replacement strategy
            pua_indices = [i for i, c in enumerate(text) if _is_pua_char(c)]
            if not pua_indices:
                return None
                
            # Try to OCR the whole text first
            full_ocr = run_ocr_segment(text)
            if full_ocr and len(full_ocr) == len(text):
                return full_ocr
                
            # If length mismatch, might need more complex alignment or segment processing
            # For this worker implementation, we'll stick to a simple attempt
            return full_ocr if full_ocr else None

        # Process Blocks
        for block_data in text_blocks_data:
            stats["total_blocks"] += 1
            
            # Unpack block data
            raw_text = block_data.get("text", "")
            font_id = block_data.get("font_id")
            font_size = block_data.get("font_size", 0.0) or 0.0
            glyphs = block_data.get("glyphs")
            boundary = block_data.get("boundary", (0,0,0,0))
            
            decoded_text = raw_text
            used_decoder = False
            used_ocr = False
            
            # 1. Glyph Decoding
            if glyph_decoder and glyphs and font_id:
                try:
                    res = glyph_decoder.decode_glyphs(font_id, glyphs, raw_text)
                    if res:
                        decoded_text = res
                        used_decoder = True
                except Exception:
                    pass
            
            # 2. OCR Fallback
            if _has_pua_characters(decoded_text):
                ocr_res = _ocr_decode_text(decoded_text, font_id, font_size)
                if ocr_res:
                    decoded_text = ocr_res
                    used_ocr = True
            
            # 3. Cleanup
            final_text = decoded_text
            remaining_pua = _count_pua_chars(final_text)
            filtered_chars = 0
            
            if remaining_pua > 0:
                cleaned = _clean_text_with_pua(final_text)
                if cleaned:
                    filtered_chars = remaining_pua - _count_pua_chars(cleaned)
                    final_text = cleaned
            
            # Update stats
            if _count_pua_chars(raw_text) > 0:
                stats["raw_pua_blocks"] += 1
            if _count_pua_chars(final_text) > 0:
                stats["remaining_pua_blocks"] += 1
            stats["filtered_pua_chars"] += filtered_chars
            if used_decoder:
                stats["used_glyph_decoder"] += 1
            if used_ocr:
                stats["used_ocr"] += 1
                
            if not final_text.strip():
                continue
                
            stats["processed_blocks"] += 1
            
            x, y, w, h = boundary
            if w <= 0:
                w = _estimate_text_width(final_text, font_size)
            if h <= 0:
                h = max(font_size * 1.2, 2.0) if font_size else 10.0
                
            items.append(_TextItem(
                text=final_text.strip(),
                x=x,
                y=y,
                width=w,
                height=h,
                font_size=font_size or h
            ))
            
        ofd_zip.close()
        return _PageResult(page_index, items, stats, image_objects_count)
        
    except Exception as e:
        import traceback
        return _PageResult(page_index, [], {}, 0, f"Worker error: {str(e)}\n{traceback.format_exc()}")


class OFDDocumentBackendMultiprocess:
    """Backend for parsing OFD documents using Multiprocessing."""

    def __init__(
        self, 
        path_or_stream: Union[BytesIO, Path],
        max_workers: Optional[int] = None
    ):
        self.path_or_stream = path_or_stream
        self.max_workers = max_workers or os.cpu_count()
        self.valid = self._probe_validity()

    def _probe_validity(self) -> bool:
        try:
            if isinstance(self.path_or_stream, BytesIO):
                self.path_or_stream.seek(0)
                try:
                    with zipfile.ZipFile(self.path_or_stream, "r") as z:
                        return "OFD.xml" in z.namelist()
                finally:
                    self.path_or_stream.seek(0)
            else:
                with zipfile.ZipFile(self.path_or_stream, "r") as z:
                    return "OFD.xml" in z.namelist()
        except Exception:
            return False

    def is_valid(self) -> bool:
        return self.valid

    def convert(self) -> DoclingDocument:
        _log.info(f"Converting OFD using Multiprocessing (workers={self.max_workers})...")
        
        # Prepare document info
        filename = "file"
        file_content = None
        is_bytes_io = False
        ofd_path_str = ""
        
        if isinstance(self.path_or_stream, Path):
            filename = self.path_or_stream.name
            ofd_path_str = str(self.path_or_stream.absolute())
            with open(self.path_or_stream, "rb") as f:
                doc_hash = hashlib.sha256(f.read()).hexdigest()
        else:
            self.path_or_stream.seek(0)
            file_content = self.path_or_stream.read()
            doc_hash = hashlib.sha256(file_content).hexdigest()
            self.path_or_stream.seek(0)
            is_bytes_io = True

        origin = DocumentOrigin(
            filename=filename,
            mimetype="application/zip",
            binary_hash=doc_hash
        )
        doc = DoclingDocument(name=Path(filename).stem, origin=origin)

        # Parse Structure (Main Process)
        try:
            if is_bytes_io:
                z = zipfile.ZipFile(BytesIO(file_content), "r")
            else:
                z = zipfile.ZipFile(ofd_path_str, "r")
                
            with z as ofd_zip:
                parser = OFDZipParser(ofd_zip)
                parsed_doc = parser.parse()
                
                # Extract Font Paths for workers
                font_paths = {
                    font_id: font.font_file
                    for font_id, font in parsed_doc.fonts.items()
                    if font.font_file
                }
                
                # Prepare Tasks
                tasks = []
                for page in parsed_doc.pages:
                    # Serialize text blocks to simple dicts to avoid pickling complex objects if needed
                    # (dataclasses are picklable, but let's be safe and explicit)
                    blocks_data = []
                    for block in page.text_blocks:
                        blocks_data.append({
                            "text": block.raw_text,
                            "font_id": block.font_id,
                            "font_size": block.font_size,
                            "glyphs": block.glyphs,
                            "boundary": block.boundary
                        })
                    
                    tasks.append({
                        "ofd_path": ofd_path_str,
                        "page_data": b"", # Not used currently
                        "page_index": page.index,
                        "page_id": page.page_id,
                        "physical_box": page.physical_box,
                        "text_blocks_data": blocks_data,
                        "image_objects_count": len(page.image_objects),
                        "font_paths": font_paths,
                        "is_bytes_io": is_bytes_io,
                        "file_content": file_content if is_bytes_io else None
                    })

            # Execute in Process Pool
            page_results = []
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(_process_page_worker, **task) for task in tasks]
                
                for future in as_completed(futures):
                    res = future.result()
                    if res.error:
                        _log.error(f"Page {res.page_index} error: {res.error}")
                    page_results.append(res)
            
            # Sort and Assemble
            page_results.sort(key=lambda r: r.page_index)
            
            total_stats = {}
            
            # Helper for line grouping (duplicated from backend)
            def _group_lines(items: List[_TextItem]) -> List[Dict]:
                if not items: return []
                sorted_items = sorted(items, key=lambda i: (round(i.y, 3), i.x))
                grouped = []
                current = []
                current_top = None
                current_height = None
                
                for item in sorted_items:
                    if current:
                        diff = abs(item.y - current_top)
                        thresh = max(current_height, item.height) * 0.6 + 0.8
                        if diff <= thresh:
                            current.append(item)
                            current_top = min(current_top, item.y)
                            current_height = max(current_height, item.height)
                        else:
                            grouped.append(current)
                            current = [item]
                            current_top = item.y
                            current_height = item.height
                    else:
                        current = [item]
                        current_top = item.y
                        current_height = item.height
                if current: grouped.append(current)
                
                lines = []
                for g in grouped:
                    g.sort(key=lambda i: i.x)
                    # Merge logic simplified
                    text_parts = []
                    last_end = None
                    for i in g:
                        if last_end is not None and (i.x - last_end) > max(i.font_size*0.6, 0.8):
                            text_parts.append(" ")
                        text_parts.append(i.text)
                        last_end = i.x + i.width
                    text = "".join(text_parts).strip()
                    if text:
                        lines.append({
                            "text": text,
                            "top": min(i.y for i in g),
                            "height": max(i.height for i in g)
                        })
                return lines

            def _build_paragraphs(lines: List[Dict]) -> List[str]:
                if not lines: return []
                paras = []
                cur_lines = []
                prev_top = None
                prev_height = None
                
                for line in lines:
                    if prev_top is None:
                        cur_lines = [line["text"]]
                    else:
                        gap = line["top"] - prev_top
                        thresh = max(prev_height, line["height"]) * 1.4
                        if gap > thresh:
                            paras.append(" ".join(cur_lines))
                            cur_lines = [line["text"]]
                        else:
                            cur_lines.append(line["text"])
                    prev_top = line["top"]
                    prev_height = line["height"]
                if cur_lines: paras.append(" ".join(cur_lines))
                return paras

            for res in page_results:
                # Merge stats
                for k, v in res.stats.items():
                    total_stats[k] = total_stats.get(k, 0) + v
                
                # Build structure
                lines = _group_lines(res.text_items)
                paras = _build_paragraphs(lines)
                
                for p in paras:
                    doc.add_text(label=DocItemLabel.PARAGRAPH, text=p)
                
                for _ in range(res.image_count):
                    doc.add_picture()
            
            _log.info(f"Converted {len(page_results)} pages. Stats: {total_stats}")
            
        except Exception as e:
            _log.error(f"Multiprocess conversion failed: {e}")
            import traceback
            traceback.print_exc()
            
        return doc
