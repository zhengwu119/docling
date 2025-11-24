# OFD格式支持说明

## 🎯 OFD格式支持

我们已经为Docling Web Demo添加了对OFD（开放文档格式）的支持！OFD是中国国家标准的电子文档格式，广泛用于政府和企事业单位。

## ✨ 功能特性

### 📄 OFD文档解析
- **文档结构解析** - 完整解析OFD文档的XML结构
- **文本内容提取** - 提取所有文本内容并保持层次结构
- **图片提取** - 提取文档中的图片资源
- **元数据获取** - 提取文档标题、作者、创建时间等信息
- **多页面支持** - 支持多页面文档的分页处理

### 🔄 格式转换
- **Markdown** - 转换为结构化的Markdown格式
- **HTML** - 生成完整的HTML文档
- **JSON** - 导出结构化的JSON数据
- **纯文本** - 提取纯文本内容

## 🏗️ 技术实现

### 核心组件

1. **OFDParser** - OFD文档解析器
   - 解析OFD的ZIP容器结构
   - 处理XML文档结构
   - 提取文本和图片内容

2. **OFDDocument** - 文档表示类
   - 统一的文档数据模型
   - 页面内容管理
   - 元数据存储

3. **OFDToMarkdownConverter** - 格式转换器
   - Markdown格式转换
   - HTML格式生成
   - 智能段落处理

### 解析流程

```
OFD文件 → ZIP解压 → XML解析 → 内容提取 → 格式转换 → 输出文件
```

## 📁 文件结构

```
docling/
├── ofd_parser.py           # OFD解析器核心模块
├── web_demo.py            # 集成OFD支持的Web应用
├── templates/index.html   # 更新的前端界面
└── requirements-web.txt   # 包含Pillow依赖
```

## 🚀 使用方法

### 1. Web界面使用

1. 启动Web Demo:
   ```bash
   ./start_web_demo.sh
   ```

2. 访问: http://localhost:8080

3. 上传OFD文件:
   - 拖拽OFD文件到上传区域
   - 或点击"选择文件"按钮选择OFD文件

4. 选择输出格式并开始转换

### 2. API调用

```python
# 使用OFD解析器
from ofd_parser import convert_ofd_to_formats

# 转换OFD文档
results = convert_ofd_to_formats(
    'document.ofd', 
    ['markdown', 'json', 'html']
)

print(results['markdown'])  # Markdown内容
print(results['json'])      # JSON数据
print(results['html'])      # HTML内容
```

### 3. 独立使用

```python
from ofd_parser import OFDParser, OFDToMarkdownConverter

# 直接解析OFD
parser = OFDParser()
document = parser.parse('document.ofd')

# 转换为Markdown
converter = OFDToMarkdownConverter()
markdown = converter.convert('document.ofd')
```

## 🔧 配置说明

### 依赖要求

- **Python 3.9+**
- **Pillow** - 图片处理库
- **标准库** - xml.etree.ElementTree, zipfile, pathlib等

### 支持的OFD版本

- OFD 1.0 标准
- OFD 1.1 标准
- 兼容大部分OFD生成软件

## 📊 处理能力

### 支持的内容类型

✅ **文本内容**
- 标题和段落
- 字体和样式信息
- 文本布局位置

✅ **图片内容**
- PNG、JPEG、BMP等格式
- 图片位置信息
- Base64编码导出

✅ **文档元数据**
- 文档标题和作者
- 创建时间和修改时间
- 文档版本信息

⚠️ **部分支持**
- 表格结构（基础支持）
- 复杂排版布局
- 自定义字体

❌ **暂不支持**
- 表单字段
- 数字签名
- 加密文档
- 3D内容

## 🐛 常见问题

### Q: OFD文件转换失败怎么办？

A: 请检查：
1. 文件是否为有效的OFD格式
2. 文件是否损坏
3. 查看控制台错误信息

### Q: 转换的Markdown格式不理想？

A: OFD到Markdown的转换是基于文本提取的，可能会丢失一些格式信息。建议：
1. 尝试HTML格式，保留更多样式
2. 检查原始OFD文档的结构
3. 使用JSON格式获取完整数据

### Q: 图片无法显示？

A: 图片以Base64格式嵌入JSON中，HTML格式会正确显示。Markdown格式中图片为引用形式。

### Q: 处理大文件很慢？

A: OFD解析需要时间，特别是包含大量图片的文档。建议：
1. 耐心等待处理完成
2. 避免同时处理多个大文件
3. 考虑只选择需要的输出格式

## 🔮 未来改进

1. **性能优化**
   - 异步图片处理
   - 内存使用优化
   - 并行解析支持

2. **功能增强**
   - 表格结构识别
   - 复杂布局处理
   - 批注和标记支持

3. **格式扩展**
   - YAML格式输出
   - DocTags格式支持
   - 自定义模板

## 🤝 贡献指南

欢迎贡献OFD解析的改进：

1. **报告问题** - 提供具体的OFD文件和错误信息
2. **功能建议** - 描述期望的解析功能
3. **代码贡献** - 优化解析算法和转换质量

现在您可以在Docling Web Demo中无缝使用OFD文档了！🎉