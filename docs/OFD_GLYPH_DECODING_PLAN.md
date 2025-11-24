# OFD字形解码完整实现方案

## 当前状况

ano.ofd转换结果信息丢失约80-90%：

| 原文 | 当前转换结果 | 信息保留率 |
|------|-------------|------------|
| 初始化扩展 | (空) | 0% |
| 设置侦听器 | 侦 | ~17% |
| 过滤事件 | 滤 | ~25% |
| 内容脚本 | 脚 | ~25% |
| 可信安全浏览器 | 浏览 | ~29% |

**根因**: ano.ofd使用PUA字符编码，需要通过字体文件的字形映射表才能解码。

## 完整解决方案

### 方案架构

```
OFD文件 (ZIP)
├── OFD.xml (文档结构)
├── Doc_0/
│   ├── Document.xml (页面列表)
│   ├── PublicRes.xml (字体定义)  ← 关键：字体路径
│   ├── Res/
│   │   ├── font_91_91.ttf       ← 字体文件
│   │   └── font_13132.ttf
│   └── Pages/
│       └── Page_1/
│           └── Content.xml      ← 包含Glyphs映射
│               <TextObject Font="91">
│                   <CGTransform>
│                       <Glyphs>2591 1553 4537</Glyphs>  ← 字形ID
│                   </CGTransform>
│                   <TextCode>\ue11e\ue236\ue28a</TextCode>  ← PUA字符
│               </TextObject>

解码流程:
1. 从PublicRes.xml读取字体ID→字体文件路径映射
2. 加载TTF字体文件 (fonttools)
3. 从CGTransform获取Glyphs ID列表
4. 在字体的cmap表中查找 glyph_id → unicode 映射
5. 重建正确的文本
```

### 实现步骤

#### 步骤1: 添加依赖

```bash
pip install fonttools
```

或在requirements.txt中添加：
```
fonttools>=4.38.0
```

#### 步骤2: 创建字形解码器类

**新文件**: `docling/backend/ofd_glyph_decoder.py`

```python
"""OFD Glyph Decoder using font files."""

import logging
from io import BytesIO
from typing import Dict, Optional
import zipfile
import xml.etree.ElementTree as ET

from fontTools.ttLib import TTFont

_log = logging.getLogger(__name__)


class OFDGlyphDecoder:
    """Decoder for OFD PUA characters using font glyph mappings."""

    def __init__(self, ofd_zip: zipfile.ZipFile, namespaces: dict):
        """
        Initialize glyph decoder.

        Args:
            ofd_zip: Opened OFD ZIP file
            namespaces: XML namespaces for parsing
        """
        self.ofd_zip = ofd_zip
        self.namespaces = namespaces
        self.fonts: Dict[str, TTFont] = {}  # font_id -> TTFont
        self.font_paths: Dict[str, str] = {}  # font_id -> file path
        self._load_font_definitions()

    def _load_font_definitions(self):
        """Load font definitions from PublicRes.xml."""
        try:
            # Try to find PublicRes.xml
            pub_res_files = [f for f in self.ofd_zip.namelist() if 'PublicRes.xml' in f]
            if not pub_res_files:
                _log.debug("No PublicRes.xml found")
                return

            pub_res_xml = self.ofd_zip.read(pub_res_files[0])
            root = ET.fromstring(pub_res_xml)

            # Parse font definitions
            for font_elem in root.findall('.//ofd:Font', self.namespaces):
                font_id = font_elem.get('ID')
                font_file_elem = font_elem.find('ofd:FontFile', self.namespaces)

                if font_id and font_file_elem is not None and font_file_elem.text:
                    # Construct full font path
                    base_loc = root.get('BaseLoc', 'Res')
                    font_path = f"Doc_0/{base_loc}/{font_file_elem.text}"
                    self.font_paths[font_id] = font_path
                    _log.debug(f"Found font {font_id}: {font_path}")

        except Exception as e:
            _log.warning(f"Failed to load font definitions: {e}")

    def _load_font(self, font_id: str) -> Optional[TTFont]:
        """
        Load a font file.

        Args:
            font_id: Font ID from TextObject

        Returns:
            TTFont object or None if failed
        """
        if font_id in self.fonts:
            return self.fonts[font_id]

        font_path = self.font_paths.get(font_id)
        if not font_path:
            return None

        try:
            if font_path not in self.ofd_zip.namelist():
                _log.debug(f"Font file not found: {font_path}")
                return None

            font_data = self.ofd_zip.read(font_path)
            font = TTFont(BytesIO(font_data))
            self.fonts[font_id] = font
            _log.debug(f"Loaded font {font_id} from {font_path}")
            return font

        except Exception as e:
            _log.warning(f"Failed to load font {font_id}: {e}")
            return None

    def decode_glyphs(self, font_id: str, glyphs_text: str, original_text: str) -> Optional[str]:
        """
        Decode text using glyph IDs.

        Args:
            font_id: Font ID
            glyphs_text: Space-separated glyph IDs (e.g., "2591 1553 4537")
            original_text: Original text with PUA characters

        Returns:
            Decoded text or None if decoding failed
        """
        font = self._load_font(font_id)
        if not font:
            return None

        try:
            # Parse glyph IDs
            glyph_ids = [int(gid) for gid in glyphs_text.split()]

            # Get cmap (character map)
            cmap = font.getBestCmap()
            if not cmap:
                _log.debug(f"Font {font_id} has no cmap")
                return None

            # Build reverse mapping: glyph_name -> unicode
            reverse_cmap = {}
            for unicode_val, glyph_name in cmap.items():
                reverse_cmap[glyph_name] = unicode_val

            # Decode each glyph
            decoded_chars = []
            for glyph_id in glyph_ids:
                # Get glyph name from ID
                glyph_name = font.getGlyphName(glyph_id)

                # Look up unicode value
                unicode_val = reverse_cmap.get(glyph_name)
                if unicode_val:
                    decoded_chars.append(chr(unicode_val))
                else:
                    # Fallback: use character from original text if available
                    if len(decoded_chars) < len(original_text):
                        decoded_chars.append(original_text[len(decoded_chars)])

            decoded_text = ''.join(decoded_chars)
            _log.debug(f"Decoded {len(glyph_ids)} glyphs: {original_text!r} -> {decoded_text!r}")
            return decoded_text

        except Exception as e:
            _log.debug(f"Failed to decode glyphs: {e}")
            return None
```

#### 步骤3: 修改OFD Backend

**文件**: `docling/backend/ofd_backend.py`

添加导入：
```python
try:
    from docling.backend.ofd_glyph_decoder import OFDGlyphDecoder
    FONTTOOLS_AVAILABLE = True
except ImportError:
    FONTTOOLS_AVAILABLE = False
    _log.info("fonttools not available, PUA character decoding disabled")
```

在`__init__`中初始化解码器：
```python
def __init__(self, in_doc: "InputDocument", path_or_stream: Union[BytesIO, Path]):
    super().__init__(in_doc, path_or_stream)
    # ... 现有代码 ...

    self.glyph_decoder = None  # 延迟初始化
```

修改`_parse_page_content`方法：
```python
def _parse_page_content(self, ofd_zip, content_path):
    # ... 现有代码 ...

    # 初始化字形解码器（第一次使用时）
    if self.glyph_decoder is None and FONTTOOLS_AVAILABLE:
        self.glyph_decoder = OFDGlyphDecoder(ofd_zip, detected_ns)

    # Parse text objects
    for text_obj in root.findall('.//ofd:TextObject', detected_ns):
        text_code = text_obj.find('.//ofd:TextCode', detected_ns)
        if text_code is not None and text_code.text:
            original_text = text_code.text
            decoded_text = original_text

            # 尝试使用字形解码
            if self.glyph_decoder and self._has_pua_characters(original_text):
                cg_transform = text_obj.find('.//ofd:CGTransform', detected_ns)
                if cg_transform is not None:
                    glyphs_elem = cg_transform.find('ofd:Glyphs', detected_ns)
                    font_id = text_obj.get('Font')

                    if glyphs_elem is not None and font_id:
                        result = self.glyph_decoder.decode_glyphs(
                            font_id,
                            glyphs_elem.text,
                            original_text
                        )
                        if result:
                            decoded_text = result
                            _log.debug(f"Glyph decoding successful: {original_text!r} -> {decoded_text!r}")

            # 如果字形解码失败，回退到PUA过滤
            if decoded_text == original_text:
                decoded_text = self._clean_text_with_pua(original_text)

            # 只添加有意义的文本
            if decoded_text.strip():
                content['text_objects'].append({
                    'text': decoded_text,
                    ...
                })
```

### 预期效果

#### 修复前（PUA过滤）
```
原文: 初始化扩展 (\ue1d8\ueda6\ue2be\ue462\ue586)
结果: (空)
保留: 0%

原文: 设置侦听器 (\ue216\ue344\ue0da\ue65a\ue34a\ue156)
结果: 侦
保留: ~17%
```

#### 修复后（字形解码）
```
原文: 初始化扩展 (\ue1d8\ueda6\ue2be\ue462\ue586)
Glyphs: 20 21 22 23 24
结果: 初始化扩展
保留: 100% ✓

原文: 设置侦听器 (\ue216\ue344\ue0da\ue65a\ue34a\ue156)
Glyphs: 100 101 102 103 104 105
结果: 设置侦听器
保留: 100% ✓
```

### 实现复杂度

| 任务 | 代码量 | 复杂度 |
|------|--------|--------|
| OFDGlyphDecoder类 | ~150行 | 中 |
| 字体文件加载 | ~30行 | 低 |
| 字形ID映射 | ~40行 | 中 |
| Backend集成 | ~50行 | 低 |
| 错误处理 | ~30行 | 低 |
| **总计** | **~300行** | **中等** |

### 依赖影响

- **依赖包**: fonttools (~500KB)
- **依赖项**: 无额外依赖
- **兼容性**: Python 3.7+
- **性能影响**: 首次加载字体需要50-100ms，之后缓存

### 降级策略

如果fonttools不可用，自动回退到PUA过滤方案：

```python
try:
    from docling.backend.ofd_glyph_decoder import OFDGlyphDecoder
    FONTTOOLS_AVAILABLE = True
except ImportError:
    FONTTOOLS_AVAILABLE = False
    # 使用现有的PUA过滤方案
```

### 测试计划

1. **单元测试**:
   ```python
   def test_glyph_decoder():
       # 测试字形解码
       decoder = OFDGlyphDecoder(ofd_zip, ns)
       result = decoder.decode_glyphs('91', '2591 1553', '\ue11e\ue236')
       assert result == '可信'
   ```

2. **集成测试**:
   ```bash
   # 测试ano.ofd完整转换
   curl -X POST http://localhost:8080/api/upload \
     -F "file=@ano.ofd" \
     -F "formats=markdown"
   # 预期: 完整目录，包含"初始化扩展"、"设置侦听器"等
   ```

### 实施步骤

#### 阶段1: 准备（5分钟）
1. 安装fonttools: `pip install fonttools`
2. 验证导入: `python -c "from fontTools.ttLib import TTFont"`

#### 阶段2: 实现（30分钟）
1. 创建`ofd_glyph_decoder.py` (~20分钟)
2. 修改`ofd_backend.py`集成解码器 (~10分钟)

#### 阶段3: 测试（15分钟）
1. 单元测试字形解码 (~5分钟)
2. 完整测试ano.ofd转换 (~5分钟)
3. 回归测试其他OFD文件 (~5分钟)

#### 总时间: ~50分钟

### 风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| fonttools安装失败 | 低 | 中 | 提供降级方案（PUA过滤） |
| 字体文件损坏 | 低 | 低 | Try-catch，回退到原文 |
| Glyphs映射错误 | 中 | 中 | 验证映射，记录警告 |
| 性能下降 | 低 | 低 | 字体缓存，延迟加载 |

### 替代方案

如果不想添加fonttools依赖，可以考虑：

#### 方案A: 预解码映射表（不推荐）
- 手动提取ano.ofd的字形映射
- 硬编码为Python字典
- **缺点**: 只适用于特定字体，通用性差

#### 方案B: 外部服务（不推荐）
- 调用外部OFD解析API
- **缺点**: 需要网络，隐私问题

#### 方案C: 接受当前限制（当前状态）
- 继续使用PUA过滤方案
- 在文档中说明限制
- **缺点**: ano.ofd转换效果差

## 建议

**推荐实施完整字形解码方案**，理由：

1. ✅ **效果显著**: 信息保留率从20% → 95%+
2. ✅ **通用性强**: 支持所有使用自定义字体的OFD
3. ✅ **实现合理**: ~300行代码，复杂度可控
4. ✅ **依赖轻量**: fonttools成熟稳定，只有500KB
5. ✅ **有降级方案**: fonttools不可用时自动回退

## 实施决策

请选择：

**选项1: 立即实施完整方案**
- 我将创建`ofd_glyph_decoder.py`并集成到backend
- 需要安装fonttools
- 预计50分钟完成

**选项2: 接受当前限制**
- ano.ofd保持部分支持（20%信息）
- 其他4个OFD文件完全正常
- 在文档中说明PUA限制

您希望我实施哪个选项？
