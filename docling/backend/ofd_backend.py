"""OFD Document Backend for Docling.

This backend provides support for parsing OFD (Open Fixed-layout Document) files,
which is the Chinese national standard for electronic documents.
"""

from __future__ import annotations

import logging
import zipfile
import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from docling_core.types.doc import DocItemLabel, DoclingDocument, DocumentOrigin

try:
    from .ofd_parser import (
        OFDParserError,
        OFDZipParser,
        OFDPage,
        OFDTextBlock,
        OFDFontResource,
    )
except ImportError:
    from ofd_parser import (
        OFDParserError,
        OFDZipParser,
        OFDPage,
        OFDTextBlock,
        OFDFontResource,
    )

_log = logging.getLogger(__name__)

try:
    try:
        from .ofd_glyph_decoder import OFDGlyphDecoder
    except ImportError:
        from ofd_glyph_decoder import OFDGlyphDecoder
    FONTTOOLS_AVAILABLE = True
except ImportError:
    FONTTOOLS_AVAILABLE = False
    _log.info("fonttools not available, PUA character decoding disabled")

try:  # pragma: no cover - optional dependency
    from rapidocr_onnxruntime import RapidOCR
    OCR_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    RapidOCR = None  # type: ignore[assignment]
    OCR_AVAILABLE = False
    _log.info("RapidOCR not available, OCR fallback disabled")


@dataclass
class _TextItem:
    text: str
    x: float
    y: float
    width: float
    height: float
    font_size: float


class OFDDocumentBackend:
    """Backend for parsing OFD (Open Fixed-layout Document) format files."""

    def __init__(self, path_or_stream: Union[BytesIO, Path]):
        _log.debug("Starting OFDDocumentBackend...")
        self.path_or_stream = path_or_stream
        self.valid = self._probe_validity()
        self._ocr_engine = None
        self._active_zip: Optional[zipfile.ZipFile] = None
        self._font_resources: Dict[str, OFDFontResource] = {}
        self._font_byte_cache: Dict[str, bytes] = {}

    def _probe_validity(self) -> bool:
        try:
            if isinstance(self.path_or_stream, BytesIO):
                self.path_or_stream.seek(0)
                try:
                    with zipfile.ZipFile(self.path_or_stream, "r") as ofd_zip:
                        return "OFD.xml" in ofd_zip.namelist()
                finally:
                    self.path_or_stream.seek(0)
            elif isinstance(self.path_or_stream, Path):
                with zipfile.ZipFile(self.path_or_stream, "r") as ofd_zip:
                    return "OFD.xml" in ofd_zip.namelist()
        except Exception as exc:
            _log.error("Failed to initialize OFD backend: %s", exc)
        _log.warning("Invalid OFD file: missing OFD.xml")
        return False

    def is_valid(self) -> bool:
        """Check if the OFD file is valid."""
        return self.valid

    def unload(self):
        """Clean up resources."""
        if isinstance(self.path_or_stream, BytesIO):
            self.path_or_stream.close()
        self.path_or_stream = None

    def _open_zip(self) -> zipfile.ZipFile:
        if isinstance(self.path_or_stream, BytesIO):
            self.path_or_stream.seek(0)
            return zipfile.ZipFile(self.path_or_stream, "r")
        return zipfile.ZipFile(self.path_or_stream, "r")

    @staticmethod
    def _is_pua_char(char: str) -> bool:
        codepoint = ord(char)
        return 0xE000 <= codepoint <= 0xF8FF

    def _has_pua_characters(self, text: str) -> bool:
        return any(self._is_pua_char(ch) for ch in text)

    def _clean_text_with_pua(self, text: str) -> str:
        cleaned = "".join(c for c in text if not self._is_pua_char(c))
        return cleaned.strip()

    def _count_pua_chars(self, text: str) -> int:
        return sum(1 for c in text if self._is_pua_char(c))

    def _estimate_text_width(self, text: str, font_size: Optional[float]) -> float:
        if font_size is None or font_size <= 0:
            font_size = 6.0
        length = max(len(text), 1)
        return font_size * 0.6 * length

    def _ensure_ocr_engine(self):
        if not OCR_AVAILABLE:
            return None
        if self._ocr_engine is None:
            try:
                self._ocr_engine = RapidOCR()  # type: ignore[call-arg]
            except Exception as exc:  # pragma: no cover - optional dependency
                _log.debug("Failed to initialise RapidOCR: %s", exc)
                self._ocr_engine = None
        return self._ocr_engine

    def _load_font_bytes(self, font_id: str) -> Optional[bytes]:
        if font_id in self._font_byte_cache:
            return self._font_byte_cache[font_id]
        if not self._active_zip:
            return None
        font_res = self._font_resources.get(font_id)
        if not font_res or not font_res.font_file:
            return None
        font_path = font_res.font_file
        if font_path not in self._active_zip.namelist():
            return None
        try:
            data = self._active_zip.read(font_path)
        except KeyError:
            return None
        self._font_byte_cache[font_id] = data
        return data

    def _ocr_decode_text(self, block: OFDTextBlock) -> Optional[str]:
        if not self._has_pua_characters(block.raw_text or ""):
            return None
        engine = self._ensure_ocr_engine()
        if engine is None:
            return None
        font_bytes = self._load_font_bytes(block.font_id or "") if block.font_id else None
        if not font_bytes:
            return None
        try:
            from PIL import Image, ImageDraw, ImageFont  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            return None

        text = block.raw_text or ""
        if not text.strip():
            return None

        font_size = block.font_size or 12.0
        def run_ocr_segment(segment_text: str) -> str:
            recognized_segment: Optional[str] = None
            best_segment_score = 0.0
            best_segment_len = 0
            for scale in (4.5, 5.5, 6.5):
                pixel_size = max(int(font_size * scale), 48)
                try:
                    font = ImageFont.truetype(BytesIO(font_bytes), pixel_size)
                except Exception:  # pragma: no cover
                    continue

                try:
                    bbox = font.getbbox(segment_text)
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                except Exception:
                    width, height = font.getsize(segment_text)
                    bbox = (0, 0, width, height)

                if width <= 0 or height <= 0:
                    continue

                margin = max(pixel_size // 2, 32)
                img = Image.new("L", (width + margin, height + margin), color=255)
                draw = ImageDraw.Draw(img)
                offset_x = margin // 2 - bbox[0]
                offset_y = margin // 2 - bbox[1]
                draw.text((offset_x, offset_y), segment_text, fill=0, font=font)

                try:
                    result, _ = engine(np.array(img))
                except Exception as exc:  # pragma: no cover - OCR runtime errors
                    _log.debug("RapidOCR segment failed: %s", exc)
                    continue

                if not result:
                    continue

                segments = [item for item in result if len(item) > 1 and item[1]]
                if not segments:
                    continue

                candidate_builder: List[str] = []
                for item in segments:
                    segment_text = item[1].strip()
                    if not segment_text:
                        continue
                    if candidate_builder:
                        overlap = 0
                        last_text = candidate_builder[-1]
                        while overlap < len(segment_text) and last_text.endswith(segment_text[: overlap + 1]):
                            overlap += 1
                        candidate_builder[-1] = last_text + segment_text[overlap:]
                    else:
                        candidate_builder.append(segment_text)
                candidate = "".join(candidate_builder).strip()
                if not candidate:
                    continue

                try:
                    scores = [float(item[2]) for item in segments if len(item) > 2]
                    aggregate_score = sum(scores) / max(len(scores), 1)
                except Exception:
                    aggregate_score = 0.0

                candidate_len = len(candidate)
                if aggregate_score > best_segment_score or (
                    aggregate_score == best_segment_score and candidate_len > best_segment_len
                ):
                    recognized_segment = candidate
                    best_segment_score = aggregate_score
                    best_segment_len = candidate_len

            return recognized_segment or ""

        pua_sequences: List[List[int]] = []
        current_seq: List[int] = []
        for index, ch in enumerate(text):
            if self._is_pua_char(ch):
                current_seq.append(index)
            else:
                if current_seq:
                    pua_sequences.append(current_seq)
                    current_seq = []
        if current_seq:
            pua_sequences.append(current_seq)

        replacement: Dict[int, str] = {}
        full_candidate: Optional[str] = None
        for seq in pua_sequences:
            segment_text = "".join(text[i] for i in seq)
            recognized_segment = run_ocr_segment(segment_text)
            if not recognized_segment:
                continue
            segment_chars = list(recognized_segment)
            for offset, idx in enumerate(seq):
                if offset < len(segment_chars):
                    replacement[idx] = segment_chars[offset]

        full_candidate = run_ocr_segment(text)
        if full_candidate:
            alignment = self._align_ocr_result(text, full_candidate)
            replacement.update(alignment)

        if not replacement:
            return None

        merged: List[str] = []
        for idx, ch in enumerate(text):
            if self._is_pua_char(ch):
                merged.append(replacement.get(idx, ""))
            else:
                merged.append(ch)

        final_text = "".join(merged).strip()
        return final_text or None

    @staticmethod
    def _align_ocr_result(raw_text: str, recognized: str) -> Dict[int, str]:
        mapping: Dict[int, str] = {}
        j = 0
        n = len(recognized)
        for i, ch in enumerate(raw_text):
            while j < n and recognized[j].isspace():
                j += 1
            if j >= n:
                break
            if 0xE000 <= ord(ch) <= 0xF8FF:
                mapping[i] = recognized[j]
                j += 1
            else:
                pos = recognized.find(ch, j)
                if pos == -1:
                    return mapping
                j = pos + 1
        return mapping

    def _decode_text_block(
        self,
        block: OFDTextBlock,
        glyph_decoder: Optional[OFDGlyphDecoder],
    ) -> Tuple[str, bool, bool, int, int, int]:
        raw_text = block.raw_text or ""
        used_decoder = False
        used_ocr = False
        raw_pua = self._count_pua_chars(raw_text)

        decoded_text = raw_text
        if glyph_decoder and block.glyphs and block.font_id:
            try:
                glyph_result = glyph_decoder.decode_glyphs(block.font_id, block.glyphs, raw_text)
            except Exception as exc:  # pragma: no cover - defensive logging
                _log.debug("Glyph decoding failed for font %s: %s", block.font_id, exc)
                glyph_result = None
            if glyph_result:
                decoded_text = glyph_result
                used_decoder = True

        if self._has_pua_characters(decoded_text):
            ocr_text = self._ocr_decode_text(block)
            if ocr_text:
                decoded_text = ocr_text
                used_ocr = True

        final_text = decoded_text
        filtered_chars = 0
        remaining_after_decode = self._count_pua_chars(decoded_text)
        if remaining_after_decode:
            cleaned = self._clean_text_with_pua(decoded_text)
            if cleaned:
                filtered_chars = remaining_after_decode - self._count_pua_chars(cleaned)
                final_text = cleaned
            else:
                filtered_chars = remaining_after_decode

        remaining_pua = self._count_pua_chars(final_text)
        return final_text, used_decoder, used_ocr, raw_pua, remaining_pua, filtered_chars

    def _build_text_items(
        self,
        page: OFDPage,
        glyph_decoder: Optional[OFDGlyphDecoder],
    ) -> Tuple[List[_TextItem], Dict[str, int]]:
        items: List[_TextItem] = []
        stats: Dict[str, int] = {
            "total_blocks": 0,
            "processed_blocks": 0,
            "raw_pua_blocks": 0,
            "remaining_pua_blocks": 0,
            "filtered_pua_chars": 0,
            "used_glyph_decoder": 0,
            "used_ocr": 0,
        }

        for block in page.text_blocks:
            stats["total_blocks"] += 1
            text, used_decoder, used_ocr, raw_pua, remaining_pua, filtered_chars = self._decode_text_block(block, glyph_decoder)
            if not text.strip():
                continue

            stats["processed_blocks"] += 1
            if raw_pua:
                stats["raw_pua_blocks"] += 1
            if remaining_pua:
                stats["remaining_pua_blocks"] += 1
            if filtered_chars:
                stats["filtered_pua_chars"] += filtered_chars
            if used_decoder:
                stats["used_glyph_decoder"] += 1
            if used_ocr:
                stats["used_ocr"] += 1

            x, y, width, height = block.boundary
            font_size = block.font_size or 0.0
            if width <= 0:
                width = self._estimate_text_width(text, block.font_size)
            if height <= 0:
                if font_size:
                    height = max(font_size * 1.2, 2.0)
                else:
                    height = max(width / max(len(text), 1), 2.0)

            items.append(
                _TextItem(
                    text=text.strip(),
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    font_size=font_size or height,
                )
            )

        return items, stats

    def _merge_line_text(self, line_items: List[_TextItem]) -> str:
        pieces: List[str] = []
        prev_end: Optional[float] = None
        for item in line_items:
            width = item.width if item.width > 0 else self._estimate_text_width(item.text, item.font_size)
            if prev_end is not None:
                gap = item.x - prev_end
                threshold = max(item.font_size * 0.6, 0.8)
                if gap > threshold:
                    pieces.append(" ")
            pieces.append(item.text)
            prev_end = item.x + width
        return "".join(pieces)

    def _group_lines(self, items: List[_TextItem]) -> List[Dict[str, float]]:
        if not items:
            return []

        sorted_items = sorted(items, key=lambda i: (round(i.y, 3), i.x))
        grouped: List[List[_TextItem]] = []
        current: List[_TextItem] = []
        current_top: Optional[float] = None
        current_height: Optional[float] = None

        for item in sorted_items:
            top = item.y
            height = item.height
            if current:
                assert current_top is not None and current_height is not None
                vertical_gap = abs(top - current_top)
                threshold = max(current_height, height, item.font_size or height, 1.0) * 0.6 + 0.8
                if vertical_gap <= threshold:
                    current.append(item)
                    current_top = min(current_top, top)
                    current_height = max(current_height, height)
                else:
                    grouped.append(current)
                    current = [item]
                    current_top = top
                    current_height = height
            else:
                current = [item]
                current_top = top
                current_height = height

        if current:
            grouped.append(current)

        line_entries: List[Dict[str, float]] = []
        for line in grouped:
            line.sort(key=lambda i: i.x)
            text = self._merge_line_text(line).strip()
            if not text:
                continue
            top = min(i.y for i in line)
            height = max(i.height for i in line)
            line_entries.append({"text": text, "top": top, "height": height})
        return line_entries

    def _build_paragraphs(self, lines: List[Dict[str, float]]) -> List[str]:
        if not lines:
            return []

        paragraphs: List[str] = []
        current_lines: List[str] = []
        prev_top: Optional[float] = None
        prev_height: Optional[float] = None

        for entry in lines:
            text = entry["text"]
            top = entry["top"]
            height = entry["height"]
            if prev_top is None:
                current_lines = [text]
            else:
                gap = top - prev_top
                threshold = max(prev_height or height, height, 1.0) * 1.4
                if gap > threshold:
                    paragraphs.append(" ".join(current_lines).strip())
                    current_lines = [text]
                else:
                    current_lines.append(text)
            prev_top = top
            prev_height = height

        if current_lines:
            paragraphs.append(" ".join(current_lines).strip())

        return [p for p in paragraphs if p]

    def _log_pua_stats(self, stats: Dict[str, int]) -> None:
        processed = max(stats.get("processed_blocks", 0), 1)
        remaining = stats.get("remaining_pua_blocks", 0)
        if remaining:
            ratio = remaining / processed
            if ratio > 0.3:
                _log.warning(
                    "OFD document retains %d text blocks with Private Use Area (PUA) characters (%.1f%% of processed blocks). "
                    "Some text may be incomplete without font-specific decoding.",
                    remaining,
                    ratio * 100,
                )
            else:
                _log.debug(
                    "PUA characters remain in %d text blocks (%.1f%% of processed blocks).",
                    remaining,
                    ratio * 100,
                )

        filtered = stats.get("filtered_pua_chars", 0)
        if filtered:
            _log.debug("Filtered %d PUA characters during normalization.", filtered)

        if FONTTOOLS_AVAILABLE and stats.get("used_glyph_decoder", 0):
            _log.debug("Applied glyph decoder to %d text blocks.", stats["used_glyph_decoder"])
        if stats.get("used_ocr", 0):
            _log.debug("Applied OCR fallback to %d text blocks with heavy PUA usage.", stats["used_ocr"])

    def convert(self) -> DoclingDocument:
        """Convert OFD document to DoclingDocument format."""
        _log.debug("Converting OFD document...")

        filename = "file"
        if isinstance(self.path_or_stream, Path):
            filename = self.path_or_stream.name
        
        # Calculate hash for the document origin
        doc_hash = ""
        try:
            if isinstance(self.path_or_stream, BytesIO):
                self.path_or_stream.seek(0)
                doc_hash = hashlib.sha256(self.path_or_stream.read()).hexdigest()
                self.path_or_stream.seek(0)
            elif isinstance(self.path_or_stream, Path):
                with open(self.path_or_stream, "rb") as f:
                    doc_hash = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            pass

        origin = DocumentOrigin(
            filename=filename,
            mimetype="application/zip",
            binary_hash=doc_hash,
        )
        doc = DoclingDocument(name=Path(filename).stem, origin=origin)

        if not self.is_valid():
            _log.error("Cannot convert invalid OFD file")
            return doc

        try:
            with self._open_zip() as ofd_zip:
                parser = OFDZipParser(ofd_zip)
                parsed_doc = parser.parse()

                self._active_zip = ofd_zip
                self._font_resources = parsed_doc.fonts
                self._font_byte_cache.clear()

                title = parsed_doc.metadata.get("title")
                if title:
                    doc.add_title(text=title)

                glyph_decoder: Optional[OFDGlyphDecoder] = None
                if FONTTOOLS_AVAILABLE:
                    font_paths = {
                        font_id: font.font_file
                        for font_id, font in parsed_doc.fonts.items()
                        if font.font_file
                    }
                    if font_paths:
                        try:
                            glyph_decoder = OFDGlyphDecoder(ofd_zip, font_paths)
                        except Exception as exc:  # pragma: no cover - defensive logging
                            _log.debug("Failed to initialise glyph decoder: %s", exc)

                overall_stats: Dict[str, int] = {
                    "total_blocks": 0,
                    "processed_blocks": 0,
                    "raw_pua_blocks": 0,
                    "remaining_pua_blocks": 0,
                    "filtered_pua_chars": 0,
                    "used_glyph_decoder": 0,
                    "used_ocr": 0,
                }

                paragraph_count = 0

                for page in parsed_doc.pages:
                    text_items, page_stats = self._build_text_items(page, glyph_decoder)
                    for key, value in page_stats.items():
                        overall_stats[key] = overall_stats.get(key, 0) + value

                    line_entries = self._group_lines(text_items)
                    paragraphs = self._build_paragraphs(line_entries)
                    for paragraph in paragraphs:
                        doc.add_text(label=DocItemLabel.PARAGRAPH, text=paragraph)
                    paragraph_count += len(paragraphs)

                    for image_object in page.image_objects:
                        if image_object.resource_id and image_object.resource_id in parsed_doc.images:
                            doc.add_picture()

                self._log_pua_stats(overall_stats)
                _log.info(
                    "Successfully converted OFD document with %d pages and %d paragraphs",
                    len(parsed_doc.pages),
                    paragraph_count,
                )

        except OFDParserError as exc:
            _log.error("Failed to parse OFD document: %s", exc)
            raise RuntimeError(f"OFD parsing error: {exc}") from exc
        except Exception as exc:
            _log.error("Failed to convert OFD document: %s", exc)
            raise RuntimeError(f"OFD conversion error: {exc}") from exc
        finally:
            self._active_zip = None
            self._font_resources = {}
            self._font_byte_cache.clear()

        return doc
