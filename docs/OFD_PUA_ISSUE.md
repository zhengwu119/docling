# OFD PUA字符（乱码）问题说明

## 问题描述

用户报告ano.ofd文件转换后出现乱码，大量关键词丢失：
- "目录" → 空
- "文档概述" → "档概"
- "内容脚本" → "脚"
- "注册脚本" → "册脚"

## 根本原因

ano.ofd文件使用了**私有使用区（Private Use Area, PUA）Unicode字符**进行自定义字体编码：

```
原始文本: '\ue11e\ue236\ue28a\ue0c0浏览\ueb06'
实际含义: '可信安全浏览器'
```

PUA字符（U+E000 到 U+F8FF）是Unicode标准中为字体厂商预留的区域，用于自定义字形。这些字符没有标准含义，必须通过字体文件中的字形映射表（Glyph Mapping）才能解码。

## OFD字形映射机制

OFD标准使用`CGTransform`元素定义字符到字形的映射：

```xml
<ofd:TextObject Font="91" Size="26.04">
    <ofd:CGTransform CodePosition="0" CodeCount="7" GlyphCount="7">
        <ofd:Glyphs>2591 1553 4537 1944 9087 16376 3224</ofd:Glyphs>
    </ofd:CGTransform>
    <ofd:TextCode>可信安全浏览器</ofd:TextCode>
</ofd:TextObject>
```

- **TextCode**: 显示的文本（可能包含PUA字符）
- **Glyphs**: 字体文件中的字形ID列表
- **CGTransform**: 定义如何将TextCode映射到Glyphs

对于使用PUA编码的文本，真实内容需要：
1. 读取字体文件（TTF/CFF）
2. 使用Glyphs ID查找对应的Unicode字符
3. 重建正确的文本

## 测试文件情况

| 文件 | PUA字符 | 转换效果 |
|------|---------|----------|
| helloworld.ofd | ❌ 无 | ✓ 正常 |
| 999.ofd | ❌ 无 | ✓ 正常 |
| 1.ofd | ❌ 无 | ✓ 正常 |
| intro.ofd | ❌ 无 | ✓ 正常 |
| **ano.ofd** | ✓ **有** | **✗ 乱码** |

## 当前解决方案（临时）

在`ofd_backend.py`中添加了`_clean_text_with_pua()`方法，简单过滤PUA字符：

```python
def _clean_text_with_pua(self, text: str) -> str:
    """过滤PUA字符，只保留标准Unicode"""
    cleaned = ''.join(c for c in text if not (0xE000 <= ord(c) <= 0xF8FF))
    return cleaned.strip()
```

**效果**:
- ✓ 保留混合文本中的正常字符（"浏览"、"Web"等）
- ✗ 丢失纯PUA编码的文本（"目录"、"文档"等）
- ✗ 信息损失约70-80%

**示例**:
```
'\ue11e\ue236\ue28a\ue0c0浏览\ueb06' → '浏览' (部分保留)
'\ue1d8\ueda6' (目录) → '' (完全丢失)
```

## 完整解决方案（需要实现）

要正确处理PUA字符，需要：

### 1. 添加依赖

```python
# requirements.txt
fonttools>=4.0.0  # TTF/CFF字体文件解析
```

### 2. 实现字形解码器

```python
from fontTools.ttLib import TTFont

class OFDGlyphDecoder:
    def __init__(self, ofd_zip):
        self.fonts = {}  # 缓存已加载的字体
        self.ofd_zip = ofd_zip

    def load_font(self, font_id, font_path):
        """从OFD中加载字体文件"""
        if font_id not in self.fonts:
            font_data = self.ofd_zip.read(font_path)
            self.fonts[font_id] = TTFont(BytesIO(font_data))
        return self.fonts[font_id]

    def decode_glyphs(self, font_id, glyph_ids, text_code):
        """
        使用字形ID解码文本

        Args:
            font_id: 字体ID
            glyph_ids: 字形ID列表 (如 "2591 1553 4537")
            text_code: 原始文本（可能包含PUA）

        Returns:
            解码后的正确文本
        """
        font = self.fonts.get(font_id)
        if not font:
            return text_code  # 回退到原文

        cmap = font.getBestCmap()  # 获取字符映射表
        reverse_cmap = {v: k for k, v in cmap.items()}  # 反向映射: glyph_name -> unicode

        decoded_chars = []
        for glyph_id in glyph_ids.split():
            glyph_name = font.getGlyphName(int(glyph_id))
            unicode_val = reverse_cmap.get(glyph_name)
            if unicode_val:
                decoded_chars.append(chr(unicode_val))

        return ''.join(decoded_chars)
```

### 3. 修改解析逻辑

```python
def _parse_page_content(self, ofd_zip, content_path, glyph_decoder):
    """解析页面内容，使用字形解码器"""

    for text_obj in root.findall('.//ofd:TextObject', ns):
        text_code = text_obj.find('.//ofd:TextCode', ns)
        cg_transform = text_obj.find('.//ofd:CGTransform', ns)

        text = text_code.text

        # 检查是否需要字形解码
        if cg_transform is not None and self._has_pua(text):
            glyphs_elem = cg_transform.find('ofd:Glyphs', ns)
            if glyphs_elem is not None:
                font_id = text_obj.get('Font')
                decoded_text = glyph_decoder.decode_glyphs(
                    font_id,
                    glyphs_elem.text,
                    text
                )
                text = decoded_text

        # 添加解码后的文本
        content['text_objects'].append({'text': text, ...})
```

## 实现复杂度评估

| 任务 | 复杂度 | 工作量 |
|------|--------|--------|
| 添加fonttools依赖 | 低 | 1行 |
| 解析PublicRes.xml获取字体路径 | 中 | 50行 |
| 加载并缓存TTF字体 | 中 | 30行 |
| 实现字形ID到Unicode映射 | 高 | 100行 |
| 处理各种字体格式（TTF/CFF/OTF） | 高 | 50行 |
| 错误处理和回退机制 | 中 | 40行 |
| **总计** | - | **~270行代码** |

## 建议方案

### 方案A：完整实现（推荐）
- **优点**: 完全支持所有OFD文件，包括使用自定义编码的文件
- **缺点**: 需要fonttools依赖，增加约100KB大小
- **适用**: 生产环境，需要处理各种OFD文件

### 方案B：当前过滤方案（已实现）
- **优点**: 无额外依赖，实现简单
- **缺点**: 对PUA编码文件支持不完整，信息损失大
- **适用**: 测试环境，或明确只处理标准编码的OFD文件

### 方案C：混合方案
- PUA字符占比 < 50%: 使用过滤方案，保留部分信息
- PUA字符占比 ≥ 50%: 返回警告，建议使用完整方案
- **优点**: 平衡了依赖和功能
- **缺点**: 对PUA重度编码文件仍然无法处理

## 下一步行动

1. **短期**: 当前的PUA过滤方案已部署，可以处理大部分OFD文件（4/5测试文件正常）
2. **中期**: 添加PUA检测和警告，告知用户ano.ofd需要完整方案
3. **长期**: 实现完整的字形解码支持（需要fonttools）

## 使用建议

对于用户：
- 如果OFD文件由标准工具生成，通常不会使用PUA编码
- ano.ofd看起来是特殊的安全浏览器文档，使用了自定义字体
- 建议使用OFD生成工具的"标准编码"选项，避免PUA字符

## 相关资源

- OFD标准: GB/T 33190-2016
- CGTransform规范: OFD标准第7.7节
- fontTools文档: https://fonttools.readthedocs.io/
- Unicode PUA: https://en.wikipedia.org/wiki/Private_Use_Areas
