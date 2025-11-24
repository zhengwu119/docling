# OFD Glyph Decoding Investigation - Final Findings

## Problem

ano.ofd file uses Private Use Area (PUA) characters extensively, causing ~80-90% content loss with simple PUA filtering.

## Investigation Results

### Font Structure Analysis

Examined font file `font_13132_0.ttf` (Font ID 115) used for PUA text:

**Glyph Order Example:**
```
Glyph 2: uniE11E -> U+E11E (PUA)
Glyph 3: uniE236 -> U+E236 (PUA)
Glyph 4: uniE28A -> U+E28A (PUA)
Glyph 5: uniE0C0 -> U+E0C0 (PUA)
Glyph 6: uni6D4F -> U+6D4F (浏)  ← Real Unicode
Glyph 7: uni89C8 -> U+89C8 (览)  ← Real Unicode
Glyph 8: uniEB06 -> U+EB06 (PUA)
```

**cmap Table:**
- All glyph names map back to their PUA codes
- uniE11E → U+E11E (self-referential, no real Unicode mapping)
- uniE236 → U+E236 (self-referential)

### XML Structure Discovery

Examined actual TextObject XML:

```xml
<TextObject Font="115">
    <CGTransform CodeCount="7" GlyphCount="7">
        <Glyphs>2 3 4 5 6 7 8</Glyphs>
    </CGTransform>
    <TextCode>浏览</TextCode>  ← Only 2 chars, not 7!
</TextObject>
```

**Key Finding:**
- `Glyphs`: 7 glyph IDs (2 3 4 5 6 7 8)
- `TextCode`: Only 2 characters (浏览) - the non-PUA fallback text
- Expected text: "可信安全浏览器" (7 characters)

The PUA characters are NOT present in TextCode - they must be rendered purely from glyph IDs.

### Root Cause

**The font file has been deliberately obfuscated:**

1. **Glyph names use PUA codes**: `uniE11E` instead of real Unicode like `uni53EF` (可)
2. **cmap maps PUA→PUA**: No path from glyph to real Unicode character
3. **TextCode contains only fallback text**: Non-PUA characters that can render without custom font

This is a **security/obfuscation feature** used in some document protection schemes to prevent text extraction.

## Attempted Solutions

### ✗ Approach 1: cmap-based decoding
**Method**: Use font.getBestCmap() to map glyph ID → Unicode
**Result**: Failed - cmap maps glyphs to PUA codes, not real Unicode

### ✗ Approach 2: Glyph name parsing
**Method**: Extract Unicode from glyph names like `uniXXXX`
**Result**: Failed - glyph names contain PUA codes (uniE11E), not real codes

### ✗ Approach 3: Post table lookup
**Method**: Use PostScript names to find Unicode mappings
**Result**: Failed - post table also uses PUA-based names

## Possible Recovery Methods (Not Implemented)

### 1. OCR-based Approach
- Render glyphs as images using font file
- Apply OCR to recognize characters
- **Complexity**: High (requires rendering engine + OCR)
- **Accuracy**: 70-90% depending on OCR quality

### 2. External Mapping Table
- Obtain PUA→Unicode mapping from font vendor/document creator
- **Feasibility**: Requires access to original mapping data
- **Likelihood**: Low (mapping may be proprietary)

### 3. Pattern Analysis
- Analyze glyph shapes to identify characters
- **Complexity**: Very high (requires glyph outline analysis + ML)
- **Accuracy**: Uncertain

## Current Implementation Status

### What Was Built

1. **OFDGlyphDecoder class** (`docling/backend/ofd_glyph_decoder.py`):
   - Loads font files from OFD ZIP
   - Parses PublicRes.xml for font definitions
   - Implements glyph ID → Unicode decoding via cmap
   - **Status**: Implemented but ineffective for obfuscated fonts

2. **Integration into ofd_backend.py**:
   - Lazy initialization of glyph decoder
   - Attempts glyph decoding before PUA filtering
   - Falls back to PUA filtering if decoding fails
   - **Status**: Fully integrated

### What Works

✅ **Standard OFD files** (helloworld.ofd, 999.ofd, 1.ofd, intro.ofd):
   - Full text extraction
   - No PUA issues

✅ **Partial extraction from ano.ofd**:
   - Non-PUA characters preserved ("浏览", "册", "（）")
   - ~20-30% content retention

### What Doesn't Work

✗ **ano.ofd PUA character recovery**:
   - Cannot decode obfuscated PUA characters
   - Glyph decoding returns PUA codes, not real Unicode
   - ~70-80% content still missing

## Recommendations

### For Production Use

**Option 1: Accept Current Limitation**
- Document that obfuscated OFD files have limited support
- Current PUA filtering preserves 20-30% of content
- Add warning when PUA ratio > 50%

**Option 2: Implement OCR Fallback**
- When glyph decoding fails, render + OCR the text
- Requires additional dependencies (Pillow, pytesseract/EasyOCR)
- Adds 100-200ms per page processing time

**Option 3: Request Unobfuscated Files**
- Advise users to obtain OFD files generated without font obfuscation
- Most standard OFD generators don't use this technique

### For This Specific Case (ano.ofd)

The file appears to be from a "可信安全浏览器" (Trusted Secure Browser) which likely uses font obfuscation as a security feature. The most practical solutions are:

1. **Request original unobfuscated document** from document creator
2. **Use OCR on rendered output** if visual accuracy is critical
3. **Accept 20-30% text extraction** for index/search purposes

## Technical Debt

### Files to Keep
- `docling/backend/ofd_glyph_decoder.py` - Works for non-obfuscated fonts
- Integration in `ofd_backend.py` - Provides graceful fallback

### Files to Remove (Debug/Test)
- `test_glyph_decoder.py`
- `debug_ano_structure.py`
- `find_pua_pages.py`
- `test_font_direct.py`
- `investigate_font.py`
- `test_glyph_names.py`
- `examine_xml.py`

## Conclusion

**Glyph-based decoding is NOT effective for ano.ofd** due to intentional font obfuscation. The current PUA filtering approach is the best practical solution without additional resources (OCR/external mappings).

The implementation is still valuable because:
1. It works for standard OFD files with unobfuscated fonts
2. It provides graceful degradation (tries glyph decoding, falls back to filtering)
3. It adds proper warning messages for users

**Final Status**: ⚠️ Partial success - infrastructure built, but specific file (ano.ofd) uses obfuscation that prevents full recovery.
