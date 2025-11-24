# 快速开始指南

## 1. 安装依赖

### 方法一：使用安装脚本（推荐）
```bash
chmod +x install_demo.sh
./install_demo.sh
```

### 方法二：使用pip安装
```bash
# 最小安装（推荐）
pip install -r requirements-minimal.txt

# 或完整安装
pip install -r requirements.txt
```

### 方法三：手动安装核心库
```bash
pip install docling PyYAML
```

## 2. 运行Demo

```bash
python3 multi_format_converter_demo.py
```

## 3. 使用自己的文档

编辑 `multi_format_converter_demo.py` 文件，修改以下部分：

```python
# 在 demo_single_conversion() 函数中
online_pdf = "你的文档路径或URL"

# 在 demo_batch_conversion() 函数中  
test_documents = [
    "文档1.pdf",
    "文档2.docx",
    "https://example.com/文档3.pdf"
]
```

## 4. 输出文件

转换后的文件将保存在 `converted_outputs/` 目录中：

```
converted_outputs/
├── document.md      # Markdown格式
├── document.html    # HTML格式
├── document.json    # JSON格式
├── document.yaml    # YAML格式
├── document.txt     # 纯文本格式
└── document.doctags.txt # DocTags格式
```

## 5. 故障排除

### 常见问题

**Q: 提示找不到python命令**
```bash
# 尝试使用python3
python3 multi_format_converter_demo.py

# 或者创建别名
alias python=python3
```

**Q: 安装docling时出错**
```bash
# 升级pip
pip install --upgrade pip

# 清理缓存重新安装
pip cache purge
pip install docling
```

**Q: 转换PDF时很慢**
A: 这是正常的，PDF处理需要进行布局分析和OCR识别。可以通过以下方式优化：
- 关闭图片生成：`generate_page_images = False`
- 使用简单pipeline：`SimplePipeline`

**Q: 内存不足**
A: 对于大文档，可以：
- 分批处理文档
- 只选择需要的输出格式
- 增加系统内存

### 系统要求

- Python 3.9+
- 至少2GB内存
- 网络连接（用于下载模型和在线文档）

### 支持的平台

- ✅ macOS (Intel/Apple Silicon)
- ✅ Linux (x86_64/arm64)  
- ✅ Windows (x86_64)

## 6. 高级使用

### 自定义输出格式
```python
converter = MultiFormatDocumentConverter()
result = converter.convert_single_document(
    "document.pdf",
    output_formats=['markdown', 'json', 'html']  # 只输出这些格式
)
```

### 批量处理
```python
files = ["doc1.pdf", "doc2.docx", "doc3.pptx"]
results = converter.convert_multiple_documents(files)

for result in results:
    if result['status'] == 'success':
        print(f"✅ {result['input_path']} 转换成功")
    else:
        print(f"❌ {result['input_path']} 转换失败: {result['error']}")
```

### 处理在线文档
```python
# 支持各种在线文档
online_docs = [
    "https://arxiv.org/pdf/2408.09869",
    "https://example.com/document.pdf"
]
```