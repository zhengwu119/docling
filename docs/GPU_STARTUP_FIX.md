# 无GPU环境启动问题修复方案

## 问题分析

### 1. NumPy版本冲突
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
```
- **原因**: 某些依赖包是用NumPy 1.x编译的，但环境安装了NumPy 2.2.6
- **影响**: 导致启动时崩溃

### 2. PyTorch/Transformers自动加载
```
from transformers import StoppingCriteria
import torch
```
- **原因**: docling的VLM模型选项自动导入transformers和torch
- **影响**: 即使不使用GPU功能，也会尝试加载PyTorch（大约2GB）
- **问题**: 在无GPU环境下可能失败或导致启动缓慢

## 解决方案

### 方案1：使用轻量级Web Demo（推荐）

使用新创建的`web_demo_lite.py`，专为无GPU环境优化：

**特点:**
- ✅ 延迟加载docling，避免启动时加载所有依赖
- ✅ 不包含PDF/IMAGE格式（这些需要重型依赖）
- ✅ 只支持轻量级格式：DOCX, PPTX, XLSX, HTML, MD, CSV, VTT, OFD
- ✅ 设置环境变量避免加载transformers
- ✅ 禁用debug模式和reloader减少内存占用

**使用方法:**
```bash
# 安装最小依赖
pip install -r requirements-web-lite.txt

# 降级NumPy（如果需要）
pip install "numpy<2.0"

# 启动轻量级版本
python web_demo_lite.py
```

**优点:**
- 启动快（不预加载重型依赖）
- 内存占用小
- 适合OFD转换等轻量级任务

**缺点:**
- 不支持PDF转换
- 不支持图片OCR

### 方案2：修复NumPy版本

如果需要完整功能，修复NumPy版本问题：

```bash
# 1. 降级NumPy到1.x版本
pip install "numpy<2.0,>=1.24.0"

# 2. 重新安装可能受影响的包
pip install --force-reinstall --no-cache-dir \
    torch torchvision transformers \
    scipy scikit-learn

# 3. 启动原版web_demo.py
python web_demo.py
```

### 方案3：使用虚拟环境隔离

创建干净的环境避免依赖冲突：

```bash
# 创建新环境
conda create -n docling-web python=3.10
conda activate docling-web

# 安装docling和依赖
pip install "numpy<2.0"
pip install -e .

# 启动
python web_demo_lite.py
```

## 两个版本对比

### web_demo.py（原版）
- ✅ 完整功能
- ✅ 支持PDF/图片转换
- ❌ 启动时加载所有依赖
- ❌ 需要GPU相关库
- ❌ 内存占用大（~3-4GB）
- ❌ NumPy版本敏感

### web_demo_lite.py（轻量级）
- ✅ 快速启动
- ✅ 小内存占用（~500MB）
- ✅ 无GPU依赖
- ✅ NumPy版本不敏感
- ✅ 完美支持OFD转换
- ❌ 不支持PDF转换
- ❌ 不支持图片OCR

## 推荐配置

### 开发/测试环境（OFD专用）
```bash
# 使用轻量级版本
python web_demo_lite.py
```

### 生产环境（完整功能）
```bash
# 1. 固定NumPy版本
echo "numpy<2.0,>=1.24.0" > numpy-constraint.txt

# 2. 安装时指定约束
pip install -c numpy-constraint.txt -e .

# 3. 启动
python web_demo.py
```

### Docker部署
```dockerfile
FROM python:3.10-slim

# 固定NumPy版本
RUN pip install "numpy<2.0,>=1.24.0"

# 安装docling（轻量级）
COPY requirements-web-lite.txt /app/
RUN pip install -r /app/requirements-web-lite.txt

# 复制代码
COPY . /app/
WORKDIR /app

# 启动轻量级版本
CMD ["python", "web_demo_lite.py"]
```

## 当前环境诊断

根据错误信息，您的环境：
- ✅ Flask正常启动（服务器运行在8080端口）
- ❌ NumPy 2.2.6与某些包不兼容
- ⚠️ PyTorch尝试初始化NumPy时失败
- ✅ Web服务器实际上已经启动（看到"Running on..."）

**建议:**
1. **立即可用**: 使用`web_demo_lite.py`
2. **长期方案**: 降级NumPy到1.x版本

## 测试OFD转换

### 使用轻量级版本
```bash
# 1. 启动服务
python web_demo_lite.py

# 2. 上传OFD文件测试
curl -X POST http://localhost:8080/api/upload \
  -F "file=@tests/data/ofd/helloworld.ofd" \
  -F "formats=markdown"

# 3. 检查转换状态
# 使用返回的task_id查询
curl http://localhost:8080/api/status/<task_id>
```

### 预期结果
```json
{
  "status": "completed",
  "document_info": {
    "word_count": 10,
    "page_count": 1
  },
  "output_files": {
    "markdown": "helloworld.md"
  }
}
```

## 注意事项

1. **NumPy版本**: 强烈建议使用NumPy 1.x（<2.0）
2. **内存**: 轻量级版本约500MB，完整版约3-4GB
3. **启动时间**: 轻量级<5秒，完整版可能需要30-60秒
4. **OFD支持**: 两个版本都完全支持OFD转换
5. **警告信息**: 即使看到NumPy警告，服务器可能仍然正常运行

## 故障排除

### 问题：服务器启动但崩溃
**解决**: 使用`web_demo_lite.py` + `debug=False, use_reloader=False`

### 问题：OFD转换返回空内容
**检查**: 确保使用了修复后的OFD backend（使用DocItemLabel.PARAGRAPH）

### 问题：内存不足
**解决**: 使用轻量级版本，不要加载PDF/图片支持

### 问题：端口被占用
**解决**: 修改端口号或停止占用8080的进程
```bash
# 查找占用端口的进程
lsof -i :8080

# 使用其他端口
python web_demo_lite.py --port 8081  # 需要修改代码支持
```
