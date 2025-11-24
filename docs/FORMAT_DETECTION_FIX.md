# OFD格式检测修复说明

## 问题描述

用户报告OFD文件无法解析，日志显示：
```
detected formats: []
ERROR: Input document e17db584-3f1e-4a10-8e0c-4e7d512aef28_999.ofd with format None does not match any allowed format
```

## 根本原因

OFD文件是ZIP压缩包格式（类似DOCX/PPTX/XLSX），但在格式检测时缺少特殊处理：

1. `filetype.guess_mime()` 识别OFD文件为 `"application/zip"`
2. DOCX/PPTX/XLSX有特殊处理，将ZIP MIME类型重写为各自的Office MIME类型
3. **OFD缺少这种特殊处理**，导致：
   - OFD文件保持通用的 `"application/zip"` MIME类型
   - 无法通过扩展名识别（`_mime_from_extension`缺少OFD支持）
   - 最终fallback到 `"text/plain"` MIME类型
   - `MimeTypeToFormat.get("text/plain")` 返回空列表
   - 格式检测失败

## 修复内容

### 文件：`docling/datamodel/document.py`

#### 修复1：Path分支的ZIP检测 (第299-300行)
```python
if mime is not None and mime.lower() == "application/zip":
    if obj.suffixes[-1].lower() == ".xlsx":
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif obj.suffixes[-1].lower() == ".docx":
        mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif obj.suffixes[-1].lower() == ".pptx":
        mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    elif obj.suffixes[-1].lower() == ".ofd":  # 新增
        mime = "application/ofd"
```

#### 修复2：DocumentStream分支的ZIP检测 (第321-322行)
```python
if mime is not None and mime.lower() == "application/zip":
    objname = obj.name.lower()
    if objname.endswith(".xlsx"):
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif objname.endswith(".docx"):
        mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif objname.endswith(".pptx"):
        mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    elif objname.endswith(".ofd"):  # 新增
        mime = "application/ofd"
```

#### 修复3：_mime_from_extension方法 (第403-404行)
```python
@staticmethod
def _mime_from_extension(ext):
    mime = None
    if ext in FormatToExtensions[InputFormat.ASCIIDOC]:
        mime = FormatToMimeType[InputFormat.ASCIIDOC][0]
    # ... 其他格式 ...
    elif ext in FormatToExtensions[InputFormat.VTT]:
        mime = FormatToMimeType[InputFormat.VTT][0]
    elif ext in FormatToExtensions[InputFormat.OFD]:  # 新增
        mime = FormatToMimeType[InputFormat.OFD][0]

    return mime
```

## 修复后的工作流程

1. **文件上传**: Web Demo接收 `999.ofd` 文件
2. **MIME检测**: `filetype.guess_mime()` 返回 `"application/zip"`
3. **OFD识别**: 检测到 `.ofd` 扩展名，重写MIME为 `"application/ofd"` ✓
4. **格式映射**: `MimeTypeToFormat["application/ofd"]` 返回 `[InputFormat.OFD]` ✓
5. **Backend调用**: DocumentConverter使用OFDDocumentBackend处理 ✓
6. **内容转换**: OFD backend提取文本并使用PARAGRAPH标签 ✓
7. **结果导出**: 正确生成Markdown/HTML/JSON等格式 ✓

## 相关文件修改清单

所有OFD支持修改的完整列表：

1. ✅ `docling/datamodel/base_models.py` - OFD格式定义（InputFormat.OFD）
2. ✅ `docling/backend/ofd_backend.py` - OFD解析实现（使用PARAGRAPH标签）
3. ✅ `docling/document_converter.py` - OFD backend注册
4. ✅ `docling/datamodel/document.py` - **OFD格式检测（本次修复）**
5. ✅ `web_demo.py` - Web界面OFD支持
6. ✅ `web_demo_lite.py` - 轻量级版本OFD支持

## 测试验证

### 方法1：使用诊断脚本
```bash
python3 diagnose_ofd.py
```

### 方法2：使用格式检测测试
```bash
python3 test_format_detection.py
```

### 方法3：Web Demo测试
```bash
# 启动Web Demo
python3 web_demo_lite.py

# 上传OFD文件
curl -X POST http://localhost:8080/api/upload \
  -F "file=@tests/data/ofd/999.ofd" \
  -F "formats=markdown"
```

### 预期日志输出
```
detected formats: [<InputFormat.OFD: 'ofd'>]  # 而不是 []
Successfully converted OFD document with X pages and Y text items
```

## 重要提示

1. **必须重启服务**: 修改代码后需要重启Web Demo才能生效
2. **NumPy版本问题**: 如果仍有NumPy 2.x错误，需要环境级别修复：
   ```bash
   pip install "numpy<2.0,>=1.24.0"
   ```
3. **完整性检查**: 确保所有6个文件的修改都已正确应用

## 技术细节

### 为什么ZIP格式需要特殊处理？

很多现代文档格式都是ZIP压缩包：
- `.docx` = Office Open XML Word (ZIP)
- `.pptx` = Office Open XML PowerPoint (ZIP)
- `.xlsx` = Office Open XML Excel (ZIP)
- `.ofd` = Open Fixed-layout Document (ZIP)

`filetype`库只能识别到外层的ZIP包装，无法区分内部的文档类型。因此需要通过文件扩展名进行二次识别，将通用的`application/zip` MIME类型重写为格式专用的MIME类型。

### 格式检测的三层防护

1. **第一层**: `filetype.guess_mime()` - 基于文件magic bytes
2. **第二层**: ZIP格式特殊处理 - 检查扩展名重写MIME
3. **第三层**: `_mime_from_extension()` - 纯粹基于扩展名

OFD在所有三层都需要正确处理才能保证可靠识别。

## 下一步

如果OFD转换仍有问题，请检查：

1. **格式检测**: 日志应显示 `detected formats: [<InputFormat.OFD: 'ofd'>]`
2. **Backend加载**: 确认OFDDocumentBackend正确导入
3. **文本标签**: 确认使用DocItemLabel.PARAGRAPH而非TEXT
4. **依赖安装**: 运行 `pip install -e .` 重新安装docling

## 修复验证清单

- [x] document.py添加Path分支OFD检测
- [x] document.py添加DocumentStream分支OFD检测
- [x] document.py的_mime_from_extension添加OFD
- [x] Git diff确认所有修改正确
- [x] 创建测试脚本test_format_detection.py
- [ ] 用户测试：重启Web Demo并上传OFD文件
- [ ] 用户验证：检查日志显示正确格式
- [ ] 用户验证：确认转换返回完整内容
