"""Utilities for parsing OFD (Open Fixed-layout Document) archives.

This module adapts concepts from the `ofdrw` project to provide
high-fidelity OFD parsing for the Docling backend.  It focuses on
extracting structured information such as text objects, images,
fonts, and metadata from an OFD container (ZIP archive).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET


logger = logging.getLogger(__name__)


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_boundary(boundary: Optional[str]) -> Tuple[float, float, float, float]:
    if not boundary:
        return (0.0, 0.0, 0.0, 0.0)
    tokens = [token for token in boundary.replace(",", " ").split() if token]
    if len(tokens) != 4:
        return (0.0, 0.0, 0.0, 0.0)
    try:
        return tuple(float(token) for token in tokens)  # type: ignore[return-value]
    except ValueError:
        return (0.0, 0.0, 0.0, 0.0)


def _parse_delta(attr: Optional[str]) -> List[float]:
    if not attr:
        return []
    tokens = [token for token in attr.replace(",", " ").split() if token]
    result: List[float] = []
    repeat_mode = False
    repeat_count = 0
    for token in tokens:
        if token == "g":
            repeat_mode = True
            repeat_count = 0
            continue
        if repeat_mode and repeat_count == 0:
            try:
                repeat_count = int(float(token))
            except ValueError:
                repeat_mode = False
            continue
        if repeat_mode and repeat_count > 0:
            try:
                value = float(token)
            except ValueError:
                repeat_mode = False
                continue
            result.extend([value] * repeat_count)
            repeat_mode = False
            repeat_count = 0
            continue
        try:
            result.append(float(token))
        except ValueError:
            continue
    return result


def _normalize_posix(path: PurePosixPath) -> PurePosixPath:
    parts: List[str] = []
    for part in path.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return PurePosixPath(*parts)


def _resolve_path(
    document_dir: PurePosixPath,
    base_loc: Optional[str],
    target: Optional[str],
) -> Optional[str]:
    if not target:
        return None
    target = target.strip()
    if not target:
        return None

    target_path = PurePosixPath(target)
    if target_path.is_absolute():
        return str(_normalize_posix(target_path))

    base_path = document_dir
    if base_loc:
        base_candidate = PurePosixPath(base_loc)
        if base_candidate.is_absolute():
            base_path = _normalize_posix(base_candidate)
        else:
            base_path = _normalize_posix(document_dir / base_candidate)

    resolved = _normalize_posix(base_path / target_path)
    return str(resolved)


def _detect_namespaces(root: ET.Element) -> Dict[str, str]:
    if "}" in root.tag:
        ns_uri = root.tag.split("}")[0][1:]
        return {"ofd": ns_uri, "ct": ns_uri}
    return {"ofd": "http://www.ofdspec.org/2016", "ct": "http://www.ofdspec.org/2016"}


@dataclass
class OFDTextCode:
    text: str
    x: Optional[float]
    y: Optional[float]
    delta_x: List[float] = field(default_factory=list)
    delta_y: List[float] = field(default_factory=list)


@dataclass
class OFDTextBlock:
    object_id: Optional[str]
    boundary: Tuple[float, float, float, float]
    font_id: Optional[str]
    font_size: Optional[float]
    layer_id: Optional[str]
    ctm: Optional[str]
    text_codes: List[OFDTextCode] = field(default_factory=list)
    glyphs: Optional[str] = None

    @property
    def raw_text(self) -> str:
        return "".join(code.text for code in self.text_codes)


@dataclass
class OFDImageObject:
    object_id: Optional[str]
    resource_id: Optional[str]
    boundary: Tuple[float, float, float, float]
    layer_id: Optional[str]
    ctm: Optional[str]


@dataclass
class OFDPage:
    index: int
    page_id: Optional[str]
    physical_box: Tuple[float, float, float, float]
    text_blocks: List[OFDTextBlock] = field(default_factory=list)
    image_objects: List[OFDImageObject] = field(default_factory=list)


@dataclass
class OFDFontResource:
    font_id: str
    name: Optional[str]
    family_name: Optional[str]
    font_file: Optional[str]


@dataclass
class OFDImageResource:
    resource_id: str
    media_type: Optional[str]
    media_format: Optional[str]
    media_file: Optional[str]


@dataclass
class OFDParsedDocument:
    metadata: Dict[str, str]
    pages: List[OFDPage]
    fonts: Dict[str, OFDFontResource]
    images: Dict[str, OFDImageResource]
    document_dir: PurePosixPath


class OFDParserError(RuntimeError):
    """Raised when an unrecoverable OFD parsing error occurs."""


class OFDZipParser:
    """Parse OFD archives and expose high-level structures."""

    def __init__(self, ofd_zip):
        self.ofd_zip = ofd_zip
        self.namespaces: Dict[str, str] = {"ofd": "http://www.ofdspec.org/2016", "ct": "http://www.ofdspec.org/2016"}

    def parse(self) -> OFDParsedDocument:
        try:
            ofd_root = self._load_xml("OFD.xml")
        except KeyError as exc:  # pragma: no cover - invalid OFD container
            raise OFDParserError("Missing OFD.xml in archive") from exc

        self.namespaces = _detect_namespaces(ofd_root)
        metadata = self._parse_metadata(ofd_root)

        doc_body = ofd_root.find(".//ofd:DocBody", self.namespaces)
        if doc_body is None:
            raise OFDParserError("Unable to locate DocBody in OFD.xml")

        doc_root_elem = doc_body.find("ofd:DocRoot", self.namespaces)
        if doc_root_elem is None or not doc_root_elem.text:
            raise OFDParserError("DocRoot element missing in OFD.xml")

        doc_root_path = doc_root_elem.text.strip()
        document_dir = PurePosixPath(doc_root_path).parent

        document_root = self._load_xml(doc_root_path)
        pages, fonts, images = self._parse_document(document_root, document_dir, doc_root_path)

        return OFDParsedDocument(
            metadata=metadata,
            pages=pages,
            fonts=fonts,
            images=images,
            document_dir=document_dir,
        )

    def _load_xml(self, path: str) -> ET.Element:
        data = self.ofd_zip.read(path)
        return ET.fromstring(data)

    def _parse_metadata(self, ofd_root: ET.Element) -> Dict[str, str]:
        metadata: Dict[str, str] = {"version": ofd_root.get("Version", "1.0")}
        doc_info = ofd_root.find(".//ofd:DocInfo", self.namespaces)
        if doc_info is None:
            return metadata

        for child in doc_info:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            value = child.text.strip() if child.text else ""
            if value:
                metadata[tag.lower()] = value
        return metadata

    def _parse_document(
        self,
        document_root: ET.Element,
        document_dir: PurePosixPath,
        document_path: str,
    ) -> Tuple[List[OFDPage], Dict[str, OFDFontResource], Dict[str, OFDImageResource]]:
        fonts: Dict[str, OFDFontResource] = {}
        images: Dict[str, OFDImageResource] = {}

        common_data = document_root.find("ofd:CommonData", self.namespaces)
        if common_data is not None:
            res_paths = list(self._iter_resource_paths(common_data, document_dir))
            for res_path in res_paths:
                self._parse_resource_file(res_path, document_dir, fonts, images)

        pages: List[OFDPage] = []
        pages_elem = document_root.find("ofd:Pages", self.namespaces)
        if pages_elem is None:
            logger.warning("Document.xml contains no Pages element: %s", document_path)
            return pages, fonts, images

        for index, page_elem in enumerate(pages_elem.findall("ofd:Page", self.namespaces), start=1):
            base_loc = page_elem.get("BaseLoc")
            page_path = _resolve_path(document_dir, None, base_loc)
            if not page_path:
                continue
            try:
                page_root = self._load_xml(page_path)
            except KeyError:
                logger.warning("Page content not found in archive: %s", page_path)
                continue
            page = self._parse_page(page_root, page_elem, index)
            pages.append(page)

        return pages, fonts, images

    def _iter_resource_paths(
        self, common_data: ET.Element, document_dir: PurePosixPath
    ) -> Iterable[str]:
        for tag_name in ("PublicRes", "DocumentRes"):
            for res_elem in common_data.findall(f"ofd:{tag_name}", self.namespaces):
                if res_elem.text:
                    res_path = _resolve_path(document_dir, None, res_elem.text)
                    if res_path:
                        yield res_path

    def _parse_resource_file(
        self,
        res_path: str,
        document_dir: PurePosixPath,
        fonts: Dict[str, OFDFontResource],
        images: Dict[str, OFDImageResource],
    ) -> None:
        try:
            res_root = self._load_xml(res_path)
        except KeyError:
            logger.debug("Resource file not found in archive: %s", res_path)
            return

        base_loc = res_root.get("BaseLoc")

        for font_elem in res_root.findall("ofd:Fonts/ofd:Font", self.namespaces):
            font_id = font_elem.get("ID")
            if not font_id:
                continue
            font_file_elem = font_elem.find("ofd:FontFile", self.namespaces)
            font_file = None
            if font_file_elem is not None and font_file_elem.text:
                font_file = _resolve_path(document_dir, base_loc, font_file_elem.text)
            fonts[font_id] = OFDFontResource(
                font_id=font_id,
                name=font_elem.get("FontName"),
                family_name=font_elem.get("FamilyName"),
                font_file=font_file,
            )

        for mm_elem in res_root.findall("ofd:MultiMedias/ofd:MultiMedia", self.namespaces):
            media_id = mm_elem.get("ID")
            if not media_id:
                continue
            media_file_elem = mm_elem.find("ofd:MediaFile", self.namespaces)
            media_file = None
            if media_file_elem is not None and media_file_elem.text:
                media_file = _resolve_path(document_dir, base_loc, media_file_elem.text)
            images[media_id] = OFDImageResource(
                resource_id=media_id,
                media_type=mm_elem.get("Type"),
                media_format=mm_elem.get("Format"),
                media_file=media_file,
            )

    def _parse_page(self, page_root: ET.Element, page_elem: ET.Element, index: int) -> OFDPage:
        physical_box = (0.0, 0.0, 0.0, 0.0)
        physical_box_elem = page_root.find("ofd:Area/ofd:PhysicalBox", self.namespaces)
        if physical_box_elem is not None and physical_box_elem.text:
            physical_box = _parse_boundary(physical_box_elem.text)

        text_blocks: List[OFDTextBlock] = []
        image_objects: List[OFDImageObject] = []

        for layer_elem in page_root.findall("ofd:Content/ofd:Layer", self.namespaces):
            layer_id = layer_elem.get("ID")
            for text_elem in layer_elem.findall("ofd:TextObject", self.namespaces):
                block = self._parse_text_object(text_elem, layer_id)
                if block.text_codes:
                    text_blocks.append(block)
            for image_elem in layer_elem.findall("ofd:ImageObject", self.namespaces):
                image = self._parse_image_object(image_elem, layer_id)
                if image is not None:
                    image_objects.append(image)

        page_id = page_elem.get("ID")
        return OFDPage(
            index=index,
            page_id=page_id,
            physical_box=physical_box,
            text_blocks=text_blocks,
            image_objects=image_objects,
        )

    def _parse_text_object(self, text_elem: ET.Element, layer_id: Optional[str]) -> OFDTextBlock:
        object_id = text_elem.get("ID")
        boundary = _parse_boundary(text_elem.get("Boundary"))
        font_id = text_elem.get("Font")
        font_size = _to_float(text_elem.get("Size"))
        ctm = text_elem.get("CTM")

        text_codes: List[OFDTextCode] = []
        for text_code_elem in text_elem.findall("ofd:TextCode", self.namespaces):
            text_codes.append(
                OFDTextCode(
                    text=text_code_elem.text or "",
                    x=_to_float(text_code_elem.get("X")),
                    y=_to_float(text_code_elem.get("Y")),
                    delta_x=_parse_delta(text_code_elem.get("DeltaX")),
                    delta_y=_parse_delta(text_code_elem.get("DeltaY")),
                )
            )

        glyphs = None
        cg_transform = text_elem.find("ofd:CGTransform", self.namespaces)
        if cg_transform is not None:
            glyphs_elem = cg_transform.find("ofd:Glyphs", self.namespaces)
            if glyphs_elem is not None and glyphs_elem.text:
                glyphs = glyphs_elem.text.strip()

        return OFDTextBlock(
            object_id=object_id,
            boundary=boundary,
            font_id=font_id,
            font_size=font_size,
            layer_id=layer_id,
            ctm=ctm,
            text_codes=text_codes,
            glyphs=glyphs,
        )

    def _parse_image_object(
        self, image_elem: ET.Element, layer_id: Optional[str]
    ) -> Optional[OFDImageObject]:
        resource_id = image_elem.get("ResourceID")
        if not resource_id:
            return None
        boundary = _parse_boundary(image_elem.get("Boundary"))
        ctm = image_elem.get("CTM")
        object_id = image_elem.get("ID")
        return OFDImageObject(
            object_id=object_id,
            resource_id=resource_id,
            boundary=boundary,
            layer_id=layer_id,
            ctm=ctm,
        )
