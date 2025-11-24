#!/usr/bin/env python3
"""
Complete example of OFD conversion using the fixed backend
This shows exactly what the output should be
"""

print("Testing OFD conversion with updated backend...")
print("="*60)

# Since we can't import docling without dependencies, let's show what SHOULD happen:

print("""
Expected behavior with the fixed OFD backend:

from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert('tests/data/ofd/helloworld.ofd')

# The result.document should contain:
# 1. Metadata (CreationDate: 2020-11-20)
# 2. Text items with content: "你好呀，OFD Reader&Writer!"

markdown = result.document.export_to_markdown()

# Expected markdown output:
---
## 文档信息

**创建时间**: 2020-11-20

你好呀，OFD Reader&Writer！
---

If you only see the metadata section but no text content,
it means the text items are being added to the document
but not being exported to markdown properly.

Possible causes:
1. The DocItemLabel.TEXT items are not being included in markdown export
2. The items are being added to the wrong parent or section
3. There's a formatting issue in how TEXT items are rendered

To debug:
1. Check the logs for "Successfully converted OFD document with X pages and Y text items"
2. If Y > 0, then text is being extracted
3. Check result.document._export_to_markdown_content() or similar methods
4. Verify TEXT label items are included in the export

""")

print("\n" + "="*60)
print("To see actual conversion, you need to run in an environment with docling installed:")
print("="*60)

print("""
# Method 1: Using test script (needs docling installed)
python3 test_ofd_conversion.py

# Method 2: Direct Python code
import logging
logging.basicConfig(level=logging.DEBUG)  # Enable debug logs

from docling.document_converter import DocumentConverter
converter = DocumentConverter()
result = converter.convert('tests/data/ofd/helloworld.ofd')

# Check internal document structure
print(f"Document has {len(result.document.texts)} text items")
print(f"Document has {len(result.document.pictures)} pictures")

# Export and print
markdown = result.document.export_to_markdown()
print(f"Markdown length: {len(markdown)}")
print("Markdown content:")
print(markdown)
""")
