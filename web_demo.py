#!/usr/bin/env python3
"""
多格式文档转换Web Demo

基于Flask的Web应用，支持在线文档转换和预览功能。

功能特性:
- 支持多种文档格式上传 (PDF, DOCX, PPTX, XLSX, HTML, MD, 图片等)
- 实时转换进度显示
- 多种输出格式 (Markdown, HTML, JSON, 纯文本)
- 在线预览转换结果
- RESTful API接口

使用方法:
    python web_demo.py
    然后访问 http://localhost:5000
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional, Any
import mimetypes

from flask import Flask, request, jsonify, render_template, send_file, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename
import yaml

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
    PowerpointFormatOption
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask应用配置
app = Flask(__name__)
app.config['SECRET_KEY'] = 'docling-web-demo-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB上传限制
CORS(app)  # 允许跨域请求

# 全局配置
UPLOAD_FOLDER = Path('web_uploads')
OUTPUT_FOLDER = Path('web_outputs')
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# 支持的文件扩展名
ALLOWED_EXTENSIONS = {
    'pdf', 'docx', 'pptx', 'xlsx', 'html', 'htm', 'md', 
    'csv', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'webp',
    'asciidoc', 'adoc', 'asc', 'vtt', 'ofd'  # 添加OFD支持
}

# 全局转换任务状态存储
conversion_tasks: Dict[str, Dict[str, Any]] = {}


class DoclingWebConverter:
    """Web版文档转换器"""
    
    def __init__(self):
        """初始化转换器"""
        # 配置PDF处理选项
        pdf_options = PdfPipelineOptions()
        pdf_options.generate_page_images = True  # 为HTML输出生成页面图片
        pdf_options.generate_picture_images = True  # 提取图片
        
        # 创建文档转换器
        self.converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.PPTX,
                InputFormat.XLSX,
                InputFormat.HTML,
                InputFormat.MD,
                InputFormat.IMAGE,
                InputFormat.CSV,
                InputFormat.ASCIIDOC,
                InputFormat.VTT,
                InputFormat.OFD,  # 添加OFD支持
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline,
                    backend=PyPdfiumDocumentBackend,
                    pipeline_options=pdf_options
                ),
                InputFormat.DOCX: WordFormatOption(
                    pipeline_cls=SimplePipeline
                ),
                InputFormat.PPTX: PowerpointFormatOption(
                    pipeline_cls=SimplePipeline
                ),
            }
        )
    
    def convert_document(self, task_id: str, input_path: str, output_formats: List[str]):
        """
        异步转换文档

        Args:
            task_id: 任务ID
            input_path: 输入文件路径
            output_formats: 输出格式列表
        """
        try:
            # 更新任务状态
            conversion_tasks[task_id]['status'] = 'processing'
            conversion_tasks[task_id]['progress'] = 10

            logger.info(f"开始转换任务 {task_id}: {input_path}")
            start_time = time.time()

            # 使用统一的Docling处理流程（包括OFD）
            self._process_docling_document(task_id, input_path, output_formats, start_time)

        except Exception as e:
            # 转换异常
            conversion_tasks[task_id].update({
                'status': 'error',
                'progress': 100,
                'error': str(e)
            })
            logger.error(f"任务 {task_id} 转换异常: {str(e)}")

    def _process_docling_document(self, task_id: str, input_path: str, output_formats: List[str], start_time: float):
        """处理Docling支持的文档"""
        try:
            # 执行转换
            result = self.converter.convert(input_path)
            conversion_tasks[task_id]['progress'] = 60
            
            if result.status == ConversionStatus.SUCCESS:
                # 获取文档基本名称
                doc_name = Path(input_path).stem
                task_output_dir = OUTPUT_FOLDER / task_id
                task_output_dir.mkdir(exist_ok=True)
                
                # 保存不同格式的输出
                output_files = self._save_outputs(result, doc_name, output_formats, task_output_dir)
                conversion_tasks[task_id]['progress'] = 90
                
                conversion_time = time.time() - start_time
                
                # 更新任务完成状态
                conversion_tasks[task_id].update({
                    'status': 'completed',
                    'progress': 100,
                    'output_files': output_files,
                    'conversion_time': conversion_time,
                    'document_info': {
                        'page_count': len(result.document.pages) if hasattr(result.document, 'pages') else 0,
                        'word_count': self._count_words(result.document.export_to_markdown()),
                    },
                    'completed_at': time.time()
                })
                
                logger.info(f"任务 {task_id} 转换成功，耗时: {conversion_time:.2f}秒")
                
            else:
                # 转换失败
                conversion_tasks[task_id].update({
                    'status': 'failed',
                    'progress': 100,
                    'error': str(result.status),
                    'errors': [error.error_message for error in result.errors] if result.errors else []
                })
                logger.error(f"任务 {task_id} 转换失败: {result.status}")
                
        except Exception as e:
            conversion_tasks[task_id].update({
                'status': 'failed',
                'progress': 100,
                'error': str(e)
            })
            logger.error(f"任务 {task_id} Docling转换失败: {str(e)}")
    
    def _save_outputs(self, result, doc_name: str, output_formats: List[str], output_dir: Path) -> Dict[str, str]:
        """保存不同格式的输出文件"""
        output_files = {}
        
        for format_name in output_formats:
            try:
                if format_name.lower() == 'markdown':
                    file_path = output_dir / f"{doc_name}.md"
                    with file_path.open('w', encoding='utf-8') as f:
                        f.write(result.document.export_to_markdown())
                    output_files['markdown'] = str(file_path)
                
                elif format_name.lower() == 'html':
                    file_path = output_dir / f"{doc_name}.html"
                    result.document.save_as_html(file_path)
                    output_files['html'] = str(file_path)
                
                elif format_name.lower() == 'json':
                    file_path = output_dir / f"{doc_name}.json"
                    with file_path.open('w', encoding='utf-8') as f:
                        json.dump(result.document.export_to_dict(), f, ensure_ascii=False, indent=2)
                    output_files['json'] = str(file_path)
                
                elif format_name.lower() == 'yaml':
                    file_path = output_dir / f"{doc_name}.yaml"
                    with file_path.open('w', encoding='utf-8') as f:
                        yaml.safe_dump(result.document.export_to_dict(), f, allow_unicode=True, default_flow_style=False)
                    output_files['yaml'] = str(file_path)
                
                elif format_name.lower() == 'text':
                    file_path = output_dir / f"{doc_name}.txt"
                    with file_path.open('w', encoding='utf-8') as f:
                        f.write(result.document.export_to_markdown(strict_text=True))
                    output_files['text'] = str(file_path)
                
                elif format_name.lower() == 'doctags':
                    file_path = output_dir / f"{doc_name}.doctags.txt"
                    with file_path.open('w', encoding='utf-8') as f:
                        f.write(result.document.export_to_document_tokens())
                    output_files['doctags'] = str(file_path)
                
            except Exception as e:
                logger.warning(f"保存 {format_name} 格式时出错: {str(e)}")
        
        return output_files
    
    def _count_words(self, text: str) -> int:
        """统计文本中的单词数量"""
        return len(text.split())


# 全局转换器实例
converter = DoclingWebConverter()


def allowed_file(filename):
    """检查文件是否为允许的格式"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    文件上传API
    
    Returns:
        JSON响应包含任务ID和状态
    """
    try:
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({'error': '没有选择文件'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'不支持的文件格式。支持的格式: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # 获取输出格式
        output_formats = request.form.getlist('formats')
        if not output_formats:
            output_formats = ['markdown', 'html']  # 默认格式
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 保存上传的文件
        filename = secure_filename(file.filename)
        file_path = UPLOAD_FOLDER / f"{task_id}_{filename}"
        file.save(str(file_path))
        
        # 创建转换任务
        conversion_tasks[task_id] = {
            'task_id': task_id,
            'filename': filename,
            'file_path': str(file_path),
            'status': 'uploaded',
            'progress': 0,
            'output_formats': output_formats,
            'created_at': time.time()
        }
        
        # 启动异步转换
        thread = Thread(target=converter.convert_document, args=(task_id, str(file_path), output_formats))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'task_id': task_id,
            'filename': filename,
            'status': 'uploaded',
            'message': '文件上传成功，开始转换...'
        })
        
    except Exception as e:
        logger.error(f"文件上传错误: {str(e)}")
        return jsonify({'error': f'上传失败: {str(e)}'}), 500


@app.route('/api/status/<task_id>')
def get_status(task_id):
    """
    获取转换任务状态
    
    Args:
        task_id: 任务ID
        
    Returns:
        JSON响应包含任务状态和进度
    """
    if task_id not in conversion_tasks:
        return jsonify({'error': '任务不存在'}), 404
    
    task = conversion_tasks[task_id]
    
    # 清理敏感信息
    safe_task = {
        'task_id': task['task_id'],
        'filename': task['filename'],
        'status': task['status'],
        'progress': task['progress'],
        'output_formats': task['output_formats']
    }
    
    # 添加结果信息
    if task['status'] == 'completed':
        safe_task.update({
            'conversion_time': task.get('conversion_time', 0),
            'document_info': task.get('document_info', {}),
            'output_files': {k: Path(v).name for k, v in task.get('output_files', {}).items()}
        })
    elif task['status'] in ['failed', 'error']:
        safe_task['error'] = task.get('error', '未知错误')
        safe_task['errors'] = task.get('errors', [])
    
    return jsonify(safe_task)


@app.route('/api/preview/<task_id>/<format_name>')
def preview_file(task_id, format_name):
    """
    预览转换结果
    
    Args:
        task_id: 任务ID
        format_name: 格式名称 (markdown, html, json, text, yaml, doctags)
        
    Returns:
        文件内容或文件下载
    """
    if task_id not in conversion_tasks:
        abort(404)
    
    task = conversion_tasks[task_id]
    
    if task['status'] != 'completed':
        abort(400)
    
    if format_name not in task.get('output_files', {}):
        abort(404)
    
    file_path = Path(task['output_files'][format_name])
    
    if not file_path.exists():
        abort(404)
    
    # 根据格式返回不同的响应
    if format_name in ['html']:
        # HTML格式直接在浏览器中显示
        return send_file(str(file_path))
    else:
        # 其他格式返回文本内容
        try:
            with file_path.open('r', encoding='utf-8') as f:
                content = f.read()
            
            # 设置适当的Content-Type
            if format_name == 'json':
                content_type = 'application/json'
            elif format_name == 'yaml':
                content_type = 'text/yaml'
            elif format_name == 'markdown':
                content_type = 'text/markdown'
            else:
                content_type = 'text/plain'
            
            return app.response_class(
                content,
                mimetype=content_type,
                headers={'Content-Disposition': f'inline; filename="{file_path.name}"'}
            )
            
        except Exception as e:
            logger.error(f"预览文件错误: {str(e)}")
            abort(500)


@app.route('/api/download/<task_id>/<format_name>')
def download_file(task_id, format_name):
    """
    下载转换结果
    
    Args:
        task_id: 任务ID
        format_name: 格式名称
        
    Returns:
        文件下载响应
    """
    if task_id not in conversion_tasks:
        abort(404)
    
    task = conversion_tasks[task_id]
    
    if task['status'] != 'completed':
        abort(400)
    
    if format_name not in task.get('output_files', {}):
        abort(404)
    
    file_path = Path(task['output_files'][format_name])
    
    if not file_path.exists():
        abort(404)
    
    return send_file(str(file_path), as_attachment=True, download_name=file_path.name)


@app.route('/api/supported-formats')
def get_supported_formats():
    """
    获取支持的文件格式
    
    Returns:
        JSON响应包含支持的输入和输出格式
    """
    return jsonify({
        'input_formats': {
            'pdf': 'PDF文档',
            'docx': 'Word文档',
            'pptx': 'PowerPoint演示文稿',
            'xlsx': 'Excel电子表格',
            'html': 'HTML网页',
            'md': 'Markdown文档',
            'csv': 'CSV数据文件',
            'png': 'PNG图片',
            'jpg': 'JPEG图片',
            'jpeg': 'JPEG图片',
            'tiff': 'TIFF图片',
            'bmp': 'BMP图片',
            'webp': 'WebP图片',
            'asciidoc': 'AsciiDoc文档',
            'vtt': 'WebVTT字幕文件',
            'ofd': 'OFD开放文档格式'  # 添加OFD支持
        },
        'output_formats': {
            'markdown': 'Markdown格式',
            'html': 'HTML格式',
            'json': 'JSON格式',
            'yaml': 'YAML格式',
            'text': '纯文本格式',
            'doctags': 'DocTags格式'
        }
    })


@app.route('/api/tasks')
def list_tasks():
    """
    获取所有转换任务列表
    
    Returns:
        JSON响应包含任务列表
    """
    # 只返回任务的基本信息
    tasks = []
    for task_id, task in conversion_tasks.items():
        tasks.append({
            'task_id': task_id,
            'filename': task['filename'],
            'status': task['status'],
            'progress': task['progress'],
            'created_at': task['created_at']
        })
    
    # 按创建时间排序（最新的在前）
    tasks.sort(key=lambda x: x['created_at'], reverse=True)
    
    return jsonify({'tasks': tasks})


@app.errorhandler(413)
def too_large(e):
    """处理文件过大错误"""
    return jsonify({'error': '文件太大，最大支持100MB'}), 413


@app.errorhandler(500)
def internal_error(e):
    """处理内部服务器错误"""
    logger.error(f"内部服务器错误: {str(e)}")
    return jsonify({'error': '服务器内部错误'}), 500


if __name__ == '__main__':
    # 确保模板目录存在
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    print("🚀 启动Docling Web Demo")
    print("=" * 50)
    print(f"📁 上传目录: {UPLOAD_FOLDER.absolute()}")
    print(f"📁 输出目录: {OUTPUT_FOLDER.absolute()}")
    print(f"🌐 访问地址: http://localhost:8080")
    print(f"📝 支持格式: {', '.join(ALLOWED_EXTENSIONS)}")
    print("=" * 50)
    
    # 启动Flask应用
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)