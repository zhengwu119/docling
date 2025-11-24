# DocumentOrigin MIME类型验证错误修复

## 问题描述

OFD文件转换时遇到Pydantic验证错误：

```
ValidationError: 1 validation error for DocumentOrigin
mimetype
  Value error, 'application/ofd' is not a valid MIME type
```

## 根本原因

`docling_core.types.doc.DocumentOrigin` 类有MIME类型验证器，只接受已注册的标准MIME类型。`application/ofd` 是我们为OFD格式自定义的MIME类型，但不在IANA注册的MIME类型列表中，因此被Pydantic验证器拒绝。

## 解决方案

### 修改文件：`docling/backend/ofd_backend.py`

**第196-200行** - 将MIME类型从 `application/ofd` 改为 `application/zip`：

```python
# 之前（导致验证错误）：
origin = DocumentOrigin(
    filename=self.file.name or "file",
    mimetype="application/ofd",  # ❌ 未注册的MIME类型
    binary_hash=self.document_hash,
)

# 修复后（使用标准MIME类型）：
origin = DocumentOrigin(
    filename=self.file.name or "file",
    mimetype="application/zip",  # ✓ 标准MIME类型
    binary_hash=self.document_hash,
)
```

## 技术细节

### 为什么使用 `application/zip`？

1. **OFD格式本质**: OFD文件是ZIP压缩包，包含XML文件和资源
2. **类似格式先例**: Office Open XML格式（DOCX/PPTX/XLSX）也是ZIP包，但它们使用专用MIME类型如 `application/vnd.openxmlformats-officedocument.wordprocessingml.document`
3. **验证器限制**: DocumentOrigin只接受IANA注册的MIME类型
4. **实用选择**: 使用 `application/zip` 作为通用ZIP文档类型

### 其他Backend的MIME类型对比

| Backend | MIME类型 | 状态 |
|---------|----------|------|
| DOCX | `application/vnd.openxmlformats-officedocument.wordprocessingml.document` | ✓ IANA注册 |
| PPTX | `application/vnd.openxmlformats-officedocument.presentationml.presentation` | ✓ IANA注册 |
| XLSX | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` | ✓ IANA注册 |
| AsciiDoc | `text/asciidoc` | ✓ 被接受 |
| Markdown | `text/markdown` | ✓ IANA注册 |
| VTT | `text/vtt` | ✓ IANA注册 |
| CSV | `text/csv` | ✓ IANA注册 |
| HTML | `text/html` | ✓ IANA注册 |
| **OFD** | ~~`application/ofd`~~ → `application/zip` | ✓ IANA注册 |

### DocumentOrigin的作用

`DocumentOrigin` 保存文档的元数据：
- `filename`: 原始文件名
- `mimetype`: MIME类型（用于标识文档格式）
- `binary_hash`: 文件哈希值（用于去重和缓存）

这些信息会包含在转换后的DoclingDocument中，但MIME类型设置为 `application/zip` 不影响：
1. **格式检测**: 已经在document.py中完成，正确识别为InputFormat.OFD
2. **Backend选择**: 已经路由到OFDDocumentBackend
3. **内容转换**: OFD解析逻辑不依赖origin.mimetype
4. **导出结果**: Markdown/HTML/JSON导出不受影响

## 注意事项

### base_models.py中的定义不需要修改

虽然我们在 `docling/backend/ofd_backend.py` 中使用 `application/zip`，但 `base_models.py` 中的定义仍然保持 `application/ofd`：

```python
# docling/datamodel/base_models.py
FormatToMimeType: dict[InputFormat, list[str]] = {
    # ...
    InputFormat.OFD: ["application/ofd"],  # 保持不变
}
```

这是正确的，因为：
1. **格式检测阶段**: document.py将检测到的 `application/ofd` MIME类型映射到 `InputFormat.OFD`
2. **Backend创建阶段**: OFDDocumentBackend初始化，使用 `application/zip` 创建DocumentOrigin
3. **两个阶段独立**: 格式检测用的MIME类型和DocumentOrigin存储的MIME类型可以不同

### 为什么不使用 `application/vnd.ofd`？

虽然按照IANA的vendor tree命名规范，OFD应该注册为类似 `application/vnd.ofd` 的MIME类型，但：
1. 目前OFD格式没有正式的IANA MIME类型注册
2. DocumentOrigin的验证器不接受未注册的vendor MIME类型
3. 使用标准的 `application/zip` 是最安全的选择

## 测试验证

修复后，OFD转换应该成功：

```bash
# 重启Web服务器
python3 web_demo_lite.py

# 上传测试
curl -X POST http://localhost:8080/api/upload \
  -F "file=@tests/data/ofd/999.ofd" \
  -F "formats=markdown"
```

**预期日志**（成功）:
```
INFO - detected formats: [<InputFormat.OFD: 'ofd'>]
INFO - Processing document xxx_999.ofd
DEBUG - Converting OFD document...
DEBUG - Valid OFD file detected
DEBUG - Found 1 pages
INFO - Successfully converted OFD document with 1 pages and X text items
INFO - 任务转换成功，耗时: 0.XX秒
```

**不再出现的错误**:
```
✗ ValidationError: 'application/ofd' is not a valid MIME type
```

## 完整修复清单

至此，OFD支持的所有修复已完成：

1. ✅ `docling/datamodel/base_models.py` - OFD格式定义
2. ✅ `docling/backend/ofd_backend.py` - OFD解析 + MIME类型修复
3. ✅ `docling/document_converter.py` - Backend注册
4. ✅ `docling/datamodel/document.py` - 格式检测
5. ✅ `web_demo.py` - Web集成
6. ✅ `web_demo_lite.py` - 轻量级版本

现在OFD文件应该能够完整转换并返回正确内容！

## 故障排除

如果仍有问题，检查：

1. **Web服务器已重启**
   ```bash
   pkill -f web_demo
   python3 web_demo_lite.py
   ```

2. **OFD文件有效**
   ```bash
   unzip -l tests/data/ofd/999.ofd | grep OFD.xml
   ```

3. **依赖已安装**
   ```bash
   pip install -e .
   ```

4. **查看详细日志**
   - 格式检测: `detected formats: [<InputFormat.OFD: 'ofd'>]`
   - OFD解析: `Valid OFD file detected`
   - 内容提取: `Found X pages and Y text items`
   - 转换成功: `任务转换成功`
