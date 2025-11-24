"""OFD Glyph Decoder using font files."""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import PurePosixPath
from typing import Dict, List, Optional
import zipfile
import xml.etree.ElementTree as ET

try:
    from fontTools.ttLib import TTFont
    FONTTOOLS_AVAILABLE = True
except ImportError:
    TTFont = None
    FONTTOOLS_AVAILABLE = False

_log = logging.getLogger(__name__)


def _normalize(path: PurePosixPath) -> PurePosixPath:
    parts = []
    for part in path.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return PurePosixPath(*parts)


def _detect_namespaces(root: ET.Element) -> Dict[str, str]:
    if "}" in root.tag:
        ns = root.tag.split("}")[0][1:]
        return {"ofd": ns, "ct": ns}
    return {"ofd": "http://www.ofdspec.org/2016", "ct": "http://www.ofdspec.org/2016"}


class OFDGlyphDecoder:
    """Decoder for OFD PUA characters using font glyph mappings."""

    def __init__(self, ofd_zip: zipfile.ZipFile, font_paths: Optional[Dict[str, Optional[str]]] = None):
        self.ofd_zip = ofd_zip
        self.fonts: Dict[str, TTFont] = {}
        self.font_paths: Dict[str, str] = {}
        if font_paths:
            self.register_font_paths(font_paths)
        if not self.font_paths:
            self._discover_font_paths()

    def register_font_paths(self, font_paths: Dict[str, Optional[str]]) -> None:
        for font_id, font_path in font_paths.items():
            if font_id and font_path:
                normalized = str(_normalize(PurePosixPath(font_path)))
                self.font_paths[font_id] = normalized

    def _discover_font_paths(self) -> None:
        for name in self.ofd_zip.namelist():
            lower = name.lower()
            if not lower.endswith("res.xml"):
                continue
            try:
                root = ET.fromstring(self.ofd_zip.read(name))
            except Exception:  # pragma: no cover - defensive fallback
                continue
            namespaces = _detect_namespaces(root)
            base_loc = root.get("BaseLoc")
            res_dir = PurePosixPath(name).parent
            for font_elem in root.findall(".//ofd:Font", namespaces):
                font_id = font_elem.get("ID")
                if not font_id:
                    continue
                font_file_elem = font_elem.find("ofd:FontFile", namespaces)
                if font_file_elem is None or not font_file_elem.text:
                    continue
                target = PurePosixPath(font_file_elem.text.strip())
                if target.is_absolute():
                    resolved = _normalize(target)
                else:
                    base_path = res_dir
                    if base_loc:
                        base_candidate = PurePosixPath(base_loc)
                        if base_candidate.is_absolute():
                            base_path = _normalize(base_candidate)
                        else:
                            base_path = _normalize(res_dir / base_candidate)
                    resolved = _normalize(base_path / target)
                self.font_paths.setdefault(font_id, str(resolved))

    def _load_font(self, font_id: str) -> Optional[TTFont]:
        if font_id in self.fonts:
            return self.fonts[font_id]

        font_path = self.font_paths.get(font_id)
        if not font_path:
            return None

        if font_path not in self.ofd_zip.namelist():
            _log.debug("Font file not found in archive: %s", font_path)
            return None

        try:
            font_data = self.ofd_zip.read(font_path)
            font = TTFont(BytesIO(font_data))
            self.fonts[font_id] = font
            _log.debug("Loaded font %s from %s", font_id, font_path)
            return font
        except Exception as exc:  # pragma: no cover - font parsing edge cases
            _log.warning("Failed to load font %s: %s", font_id, exc)
            return None

    def decode_glyphs(self, font_id: str, glyphs_text: str, original_text: str) -> Optional[str]:
        font = self._load_font(font_id)
        if not font:
            return None

        try:
            glyph_ids = [int(gid) for gid in glyphs_text.split() if gid.strip()]
        except ValueError:
            return None

        if not glyph_ids:
            return None

        try:
            cmap = font.getBestCmap()
        except Exception as exc:  # pragma: no cover
            _log.debug("Failed to read cmap for font %s: %s", font_id, exc)
            return None

        if not cmap:
            _log.debug("Font %s does not expose a cmap", font_id)
            return None

        reverse_cmap = {glyph_name: codepoint for codepoint, glyph_name in cmap.items()}

        decoded_chars: List[str] = []
        for glyph_id in glyph_ids:
            try:
                glyph_name = font.getGlyphName(glyph_id)
            except Exception:  # pragma: no cover - glyph lookup failures
                glyph_name = None
            if glyph_name is None:
                continue
            codepoint = reverse_cmap.get(glyph_name)
            if codepoint is not None:
                decoded_chars.append(chr(codepoint))
            elif len(decoded_chars) < len(original_text):
                decoded_chars.append(original_text[len(decoded_chars)])

        if not decoded_chars:
            return None

        decoded_text = "".join(decoded_chars)
        _log.debug("Decoded %d glyphs for font %s", len(decoded_chars), font_id)
        return decoded_text
