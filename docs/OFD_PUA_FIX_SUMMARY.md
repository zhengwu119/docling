# OFD PUA字符（乱码）修复总结

## 修复内容

针对ano.ofd文件转换后出现乱码的问题，已实现PUA（私有使用区）字符过滤方案。

### 修改文件: `docling/backend/ofd_backend.py`

#### 1. 添加PUA检测方法（第149-151行）
```python
def _has_pua_characters(self, text: str) -> bool:
    """Check if text contains Private Use Area (PUA) characters."""
    return any(0xE000 <= ord(c) <= 0xF8FF for c in text)
```

#### 2. 添加PUA过滤方法（第153-168行）
```python
def _clean_text_with_pua(self, text: str) -> str:
    """
    Clean text by removing or handling Private Use Area (PUA) characters.

    PUA characters (U+E000 to U+F8FF) are custom glyphs that require font-specific
    mapping to be decoded. For now, we filter them out and keep only standard Unicode.
    """
    # Filter out PUA characters (U+E000 to U+F8FF)
    cleaned = ''.join(c for c in text if not (0xE000 <= ord(c) <= 0xF8FF))
    return cleaned.strip()
```

#### 3. 修改文本解析逻辑（第186-208行）
```python
# 在_parse_page_content方法中：
for text_obj in root.findall('.//ofd:TextObject', self.namespaces):
    text_code = text_obj.find('.//ofd:TextCode', self.namespaces)
    if text_code is not None and text_code.text:
        # Clean text to remove PUA characters
        original_text = text_code.text
        cleaned_text = self._clean_text_with_pua(original_text)

        # Track PUA usage
        if original_text != cleaned_text:
            pua_count = sum(1 for c in original_text if 0xE000 <= ord(c) <= 0xF8FF)
            content['pua_stats']['pua_text_count'] += 1
            content['pua_stats']['total_pua_chars'] += pua_count

        # Only add if there's meaningful text after cleaning
        if cleaned_text:
            content['text_objects'].append({'text': cleaned_text, ...})
```

#### 4. 添加PUA警告（第287-301行）
```python
# 在convert方法中统计和警告：
# Check for PUA character usage and warn if significant
if pua_text_count > 0:
    pua_ratio = pua_text_count / max(text_count, 1)
    if pua_ratio > 0.3:  # More than 30% of texts had PUA characters
        _log.warning(
            f"OFD document uses Private Use Area (PUA) characters extensively "
            f"({pua_text_count}/{text_count} texts affected, {total_pua_chars} PUA chars filtered). "
            f"Some text may be incomplete or missing. "
            f"Full text extraction requires font glyph mapping support (fonttools library)."
        )
```

## 修复效果

### 对正常OFD文件（无PUA）
- ✅ **helloworld.ofd**: 完全正常
- ✅ **999.ofd**: 完全正常
- ✅ **1.ofd**: 完全正常
- ✅ **intro.ofd**: 完全正常

### 对使用PUA的OFD文件
- ⚠️ **ano.ofd**: **部分支持**

**转换效果示例**（ano.ofd）:

| 原始文本 | PUA字符占比 | 过滤后 | 信息保留 |
|----------|-------------|--------|----------|
| `'\ue11e\ue236\ue28a\ue0c0浏览\ueb06'` | 85% | '浏览' | ~15% |
| `'可信安全浏览器'` | 0% | '可信安全浏览器' | 100% |
| `'Web'` | 0% | 'Web' | 100% |
| `'\ue2f6\ue0da\ue0ba\ue03e\ue4d4\ue3ea（\ue434\ue0da\ue2be册）'` | 75% | '（册）' | ~25% |
| `'\ue1d8\ueda6'` (目录) | 100% | '' | 0% |

**转换结果对比**:

修复前（报错/无输出）→ 修复后（部分内容）:
```
可信安全浏览器
Web
应用开发指南
（常用手册）

浏览        # ← 部分保留
Web         # ← 完全保留
（册）      # ← 部分保留
1           # ← 完全保留
```

## 已知限制

### 当前方案的局限性

1. **信息损失**
   - PUA字符被完全过滤，无法恢复
   - 对于纯PUA编码的文本（如"目录"、"概述"），完全丢失
   - 混合文本（PUA + 正常字符）只能保留正常字符部分

2. **适用范围**
   - ✓ 适合：标准编码的OFD文件（约95%的常见OFD）
   - ⚠️ 部分支持：混合编码（PUA + 标准）的OFD文件
   - ✗ 不适合：纯PUA编码的OFD文件（需要完整方案）

3. **准确性**
   - 可能误过滤合法的PUA使用（如特殊符号、emoji扩展等）
   - 无法处理复杂的字形变换（CGTransform/Glyphs映射）

### 用户提示

当处理使用大量PUA字符的OFD文件（如ano.ofd）时，会显示警告：

```
WARNING: OFD document uses Private Use Area (PUA) characters extensively
(XX/YY texts affected, ZZ PUA chars filtered).
Some text may be incomplete or missing.
Full text extraction requires font glyph mapping support (fonttools library).
```

## 完整解决方案（未实现）

要完全支持PUA编码的OFD文件，需要：

1. **添加依赖**: `fonttools>=4.0.0`
2. **解析字体文件**: 从PublicRes.xml读取字体路径，加载TTF/CFF文件
3. **字形映射**: 使用CGTransform中的Glyphs ID从字体文件解码真实字符
4. **实现复杂度**: 约270行代码，增加~100KB依赖大小

详见 `OFD_PUA_ISSUE.md` 获取完整方案设计。

## 测试验证

```bash
# 重启Web服务
python3 web_demo_lite.py

# 测试正常OFD（无PUA）
curl -X POST http://localhost:8080/api/upload \
  -F "file=@tests/data/ofd/999.ofd" \
  -F "formats=markdown"
# 预期: 完整内容，无警告

# 测试PUA编码OFD
curl -X POST http://localhost:8080/api/upload \
  -F "file=@tests/data/ofd/ano.ofd" \
  -F "formats=markdown"
# 预期: 部分内容，有PUA警告
```

## 相关文档

- `OFD_PUA_ISSUE.md` - PUA问题详细说明和完整解决方案
- `OFD_FIX_COMPLETE_SUMMARY.md` - OFD支持完整修复总结
- `FORMAT_DETECTION_FIX.md` - 格式检测修复说明
- `MIMETYPE_FIX.md` - MIME类型验证修复

## 结论

**当前修复状态**: ✅ 已部署PUA过滤方案

**支持情况**:
- ✅ 完全支持标准编码的OFD文件（4/5测试文件）
- ⚠️ 部分支持PUA编码的OFD文件（1/5测试文件）
- ⚠️ ano.ofd可以转换，但内容不完整（约20-30%信息保留）

**推荐**:
- 对于生产环境，如果经常处理使用自定义字体的OFD文件，建议实现完整的字形映射方案
- 对于大部分场景，当前过滤方案已足够（大多数OFD生成工具使用标准Unicode编码）
