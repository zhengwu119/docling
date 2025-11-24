# 多格式文档转换Demo说明

这个Demo展示了如何使用Docling库进行多种格式文档的转换。

## 功能特性

### 支持的输入格式
- **PDF文档** - 包括原生PDF和扫描版PDF
- **Microsoft Office文档** - Word (.docx), PowerPoint (.pptx), Excel (.xlsx)
- **Web格式** - HTML网页
- **文本格式** - Markdown (.md), AsciiDoc
- **图片格式** - PNG, JPEG, TIFF等
- **数据格式** - CSV文件
- **字幕格式** - WebVTT (.vtt)

### 支持的输出格式
- **Markdown** (.md) - 结构化文本格式
- **HTML** (.html) - 网页格式，包含嵌入图片
- **JSON** (.json) - 结构化数据格式
- **YAML** (.yaml) - 人类可读的数据格式
- **纯文本** (.txt) - 去除格式的文本
- **DocTags** (.doctags.txt) - 文档标记格式

## 安装依赖

```bash
# 安装Docling
pip install docling

# 安装YAML支持（可选）
pip install pyyaml
```

## 使用方法

### 1. 运行完整演示
```bash
python multi_format_converter_demo.py
```

### 2. 使用转换器类
```python
from multi_format_converter_demo import MultiFormatDocumentConverter

# 创建转换器实例
converter = MultiFormatDocumentConverter(output_dir="my_outputs")

# 转换单个文档
result = converter.convert_single_document(
    "path/to/document.pdf", 
    output_formats=['markdown', 'html', 'json']
)

# 批量转换
results = converter.convert_multiple_documents([
    "document1.pdf",
    "document2.docx", 
    "https://example.com/document.pdf"
])
```

### 3. 自定义配置

```python
# 自定义PDF处理选项
from docling.datamodel.pipeline_options import PdfPipelineOptions

pdf_options = PdfPipelineOptions()
pdf_options.generate_page_images = True  # 生成页面图片
pdf_options.generate_picture_images = True  # 提取文档中的图片
pdf_options.ocr_enabled = True  # 启用OCR识别

# 使用自定义选项创建转换器
converter = MultiFormatDocumentConverter()
# 在创建时传入自定义选项...
```

## 输出示例

转换后的文件将保存在指定的输出目录中：

```
converted_outputs/
├── document.md          # Markdown格式
├── document.html        # HTML格式（包含图片）
├── document.json        # JSON结构化数据
├── document.yaml        # YAML格式
├── document.txt         # 纯文本
└── document.doctags.txt # DocTags格式
```

## 高级功能

### 1. 在线文档转换
支持直接转换在线PDF文档：
```python
result = converter.convert_single_document("https://arxiv.org/pdf/2408.09869")
```

### 2. 错误处理
```python
result = converter.convert_single_document("path/to/document.pdf")

if result['status'] == 'success':
    print(f"转换成功! 输出文件: {result['output_files']}")
    print(f"文档信息: {result['document_info']}")
else:
    print(f"转换失败: {result['error']}")
```

### 3. 批量处理进度跟踪
```python
results = converter.convert_multiple_documents(file_list)

success_count = sum(1 for r in results if r['status'] == 'success')
print(f"成功转换 {success_count}/{len(results)} 个文档")
```

## 性能优化建议

1. **批量处理** - 对于大量文档，使用 `convert_multiple_documents()` 方法
2. **格式选择** - 只选择需要的输出格式以减少处理时间
3. **图片处理** - 根据需要开启/关闭图片生成功能
4. **内存管理** - 对于大文档，考虑分批处理

## 常见问题

### Q: 如何处理扫描版PDF？
A: Docling自动检测并使用OCR技术处理扫描版PDF。

### Q: 支持中文文档吗？
A: 是的，支持多语言文档，包括中文。

### Q: 如何提取文档中的图片？
A: 设置 `generate_picture_images=True` 即可提取文档中的图片。

### Q: 转换速度慢怎么办？
A: 可以调整pipeline选项，关闭不需要的功能如图片生成。

## 扩展应用

这个Demo可以很容易地集成到更大的应用中：

- **文档管理系统** - 自动转换上传的文档
- **内容分析系统** - 提取文档结构化数据
- **搜索引擎** - 为文档建立索引
- **知识库系统** - 统一文档格式