import os
import subprocess
import time
import requests
import pytest

BASE_URL = "http://127.0.0.1:8080"
UPLOAD_URL = f"{BASE_URL}/api/upload"
STATUS_URL = f"{BASE_URL}/api/status"

@pytest.fixture(scope="module")
def flask_process():
    # Start the web demo lite server in a subprocess
    proc = subprocess.Popen(["python", "web_demo_lite.py"], cwd="/Users/admin/Documents/Dev/docling")
    # Wait a bit for server to start
    time.sleep(5)
    yield proc
    # Terminate the server
    proc.terminate()
    proc.wait()

def test_allowed_docx_conversion(flask_process):
    # Create a minimal DOCX file
    docx_path = "/tmp/test.docx"
    # Use python-docx to create a simple docx
    from docx import Document
    doc = Document()
    doc.add_paragraph("Hello World")
    doc.save(docx_path)
    with open(docx_path, "rb") as f:
        files = {"file": ("test.docx", f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        resp = requests.post(UPLOAD_URL, files=files)
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]
    # Poll status
    for _ in range(20):
        status_resp = requests.get(f"{STATUS_URL}/{task_id}")
        if status_resp.status_code == 200 and status_resp.json()["status"] == "SUCCESS":
            break
        time.sleep(1)
    else:
        pytest.fail("DOCX conversion did not succeed")

def test_pdf_not_allowed(flask_process):
    # Create a dummy PDF file
    pdf_path = "/tmp/test.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.4\n%EOF")
    with open(pdf_path, "rb") as f:
        files = {"file": ("test.pdf", f, "application/pdf")}
        resp = requests.post(UPLOAD_URL, files=files)
    # The server should reject PDF as not allowed (400 Bad Request)
    assert resp.status_code == 400
