# OFD Backend 重要修复

## 问题
用户报告OFD文件转换后只显示元数据（如"**创建时间**: 2020-11-20"），但正文内容为空。

## 根本原因
在OFD backend的`convert()`方法中，我使用了错误的标签：
```python
doc.add_text(label=DocItemLabel.TEXT, text=text)  # 错误！
```

应该使用：
```python
doc.add_text(label=DocItemLabel.PARAGRAPH, text=text)  # 正确！
```

## 原因分析
查看其他backend（如asciidoc_backend.py:244-248）的实现，它们在添加普通文本内容时使用`DocItemLabel.PARAGRAPH`标签。`DocItemLabel.TEXT`可能用于其他特殊用途，而`PARAGRAPH`才是正文段落的正确标签，会被正确导出到Markdown。

## 已修复
已在`docling/backend/ofd_backend.py:240-243`修改为使用`DocItemLabel.PARAGRAPH`。

## 测试建议
请重新测试OFD文件转换：

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert('tests/data/ofd/helloworld.ofd')
markdown = result.document.export_to_markdown()

print(markdown)
```

预期输出应包含：
```markdown
## 文档信息

**创建时间**: 2020-11-20

你好呀，OFD Reader&Writer！
```

而不是只有元数据部分。

## 额外改进
同时添加了更详细的debug日志，便于追踪转换过程：
- 显示找到的页数
- 显示每页的文本对象数量
- 显示成功转换的文本项总数

日志示例：
```
DEBUG: Found 1 pages
DEBUG: Processing page 1: Doc_0/Pages/Page_0/Content.xml
DEBUG: Page 1 has 1 text objects
INFO: Successfully converted OFD document with 1 pages and 1 text items
```
