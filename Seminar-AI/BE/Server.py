"""
app.py

Ứng dụng phân tích cảm xúc tiếng Việt dùng mô hình PhoBERT.
Tính năng chính:

- Dùng `pipeline("sentiment-analysis")` với model tiếng Việt.
- Chuẩn hoá nhãn cảm xúc thành 3 nhãn chuẩn: POSITIVE / NEGATIVE / NEUTRAL.
- Làm sạch + thêm dấu tiếng Việt tự động bằng Gemini API.
- Lưu lịch sử vào SQLite.
- FastAPI API để frontend gọi.

Cách chạy:
    python -m uvicorn Server:app --reload --port 8000

Yêu cầu cài đặt:
    pip install -r requirements.txt
"""

# =======================================================
# IMPORT THƯ VIỆN
# =======================================================
from datetime import datetime
import sqlite3
from typing import Optional

from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformers import pipeline, AutoTokenizer
from underthesea import word_tokenize

import requests

# =======================================================
# CẤU HÌNH MẶC ĐỊNH
# =======================================================
DB_PATH = "vi_sentiment_history.db"
DEFAULT_MODEL = "wonrax/phobert-base-vietnamese-sentiment"

app = FastAPI(title="Phân tích cảm xúc tiếng Việt (PhoBERT)", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép mọi domain truy cập API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Biến giữ pipeline đã load để tránh load lại nhiều lần
_PIPELINE = None
_MODEL_NAME = DEFAULT_MODEL


# =======================================================
# KHỞI TẠO DATABASE SQLITE
# =======================================================
def init_db(path: str = DB_PATH):
    """Tạo database và bảng lịch sử nếu chưa có."""
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            tokenized_text TEXT,
            label TEXT NOT NULL,
            score REAL NOT NULL,
            model TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


def save_history(text: str, tokenized: str, label: str, score: float, model_name: str, path: str = DB_PATH):
    """Lưu lịch sử dự đoán để tiện thống kê hoặc xem lại."""
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute(
        "INSERT INTO history (text, tokenized_text, label, score, model, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (text, tokenized, label, float(score), model_name, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_history(limit: int = 100, path: str = DB_PATH):
    """Lấy danh sách lịch sử gần nhất."""
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute(
        "SELECT id, text, tokenized_text, label, score, model, created_at FROM history ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = c.fetchall()
    conn.close()
    keys = ["id", "text", "tokenized_text", "label", "score", "model", "created_at"]
    return [dict(zip(keys, r)) for r in rows]


# =======================================================
# RESPONSE MODEL CHO FASTAPI
# =======================================================
class AnalyzeResponse(BaseModel):
    text: str
    tokenized_text: str
    label: str
    score: float
    model: str


# =======================================================
# LOAD PIPELINE PHOBERT
# =======================================================
def get_pipeline(model_name: str = None):
    """
    Load pipeline xử lý cảm xúc.
    - Dùng global để không load lại mỗi lần request.
    - PhoBERT yêu cầu tokenizer riêng và phải bật `use_fast=False`.
    """
    global _PIPELINE, _MODEL_NAME

    if model_name is None:
        model_name = _MODEL_NAME

    if _PIPELINE is None or _MODEL_NAME != model_name:
        print(f"Đang load pipeline cho model: {model_name} ...")

        # Mọi model PhoBERT đều cần tokenizer đặc biệt
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        _PIPELINE = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=tokenizer
        )
        _MODEL_NAME = model_name

    return _PIPELINE


# =======================================================
# THÊM DẤU TỰ ĐỘNG + CHUẨN HOÁ CÂU
# =======================================================
def call_gemini(prompt: str):
    """
    Gọi Gemini API để:
    - Thêm dấu tiếng Việt nếu bị mất dấu
    - Mở rộng từ viết tắt, teen code
    - Chuẩn hoá câu

    Trả về câu tiếng Việt hoàn chỉnh, giữ nguyên nghĩa.
    """
    api_key = "AIzaSyAxyX2EICyfWwn2anU3SCrUkkHAB7Btwpo"
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    data = {
        "contents": [
            {
                "parts": [
                    {"text": f"Kiểm tra câu sau, nếu câu không có dấu thì hãy tự động thêm dấu tiếng Việt đầy đủ;"
                     "nếu đã có dấu thì giữ nguyên. Tiếp theo, hãy kiểm tra và mở rộng mọi từ viết tắt thành dạng đầy đủ theo đúng nghĩa trong ngữ cảnh. " 
                     "Cuối cùng, chỉ trả về câu tiếng Việt đã được chuẩn hoá và định dạng hoàn chỉnh, không giải thích gì thêm: {prompt}"}
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        raise Exception(f"Lỗi Gemini API {response.status_code}: {response.text}")

    result = response.json()
    return result["candidates"][0]["content"]["parts"][0]["text"]


def vietnamese_segment(text: str) -> str:

    """
    Chuẩn hoá câu tiếng Việt:
    - Gọi Gemini để thêm dấu + sửa từ viết tắt
    - Không dùng word_tokenize vì model fine-tuned của bạn đã dùng văn bản thô
    """
    normalized = call_gemini(text)
    return normalized


# =======================================================
# CHUẨN HOÁ LABEL
# =======================================================
def normalize_label(raw_label: str) -> str:
    """
    Chuẩn hoá nhãn cảm xúc từ model về 3 nhãn chuẩn:
    - POSITIVE
    - NEGATIVE
    - NEUTRAL
    """
    label = raw_label.strip().upper()

    mapping = {
        # POSITIVE
        "POS": "POSITIVE",
        "POSITIVE": "POSITIVE",
        "1": "POSITIVE",
        "LABEL_1": "POSITIVE",
        "__LABEL__1": "POSITIVE",

        # NEGATIVE
        "NEG": "NEGATIVE",
        "NEGATIVE": "NEGATIVE",
        "0": "NEGATIVE",
        "LABEL_0": "NEGATIVE",
        "__LABEL__0": "NEGATIVE",

        # NEUTRAL
        "NEU": "NEUTRAL",
        "NEUTRAL": "NEUTRAL",
        "2": "NEUTRAL",
        "LABEL_2": "NEUTRAL",
        "__LABEL__2": "NEUTRAL",
    }

    return mapping.get(label, raw_label)  # fallback nếu không match


# =======================================================
# ENDPOINT PHÂN TÍCH CẢM XÚC
# =======================================================
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(text: str = Form(...), model: Optional[str] = Form(None)):
    """
    Phân tích cảm xúc của câu đầu vào.
    - Nhận text
    - Chuẩn hoá tiếng Việt (thêm dấu)
    - Đưa vào model
    - Chuẩn hoá nhãn và trả kết quả
    """
    model_name = model or _MODEL_NAME
    pipe = get_pipeline(model_name)

    tokenized = vietnamese_segment(text)

    result = pipe(tokenized)[0]
    raw_label = result.get("label")
    label = normalize_label(raw_label)
    score = float(result.get("score", 0.0))

    # Lưu vào DB
    save_history(text, tokenized, label, score, model_name)

    return AnalyzeResponse(
        text=text,
        tokenized_text=tokenized,
        label=label,
        score=score,
        model=model_name
    )


# =======================================================
# ROUTE LẤY LỊCH SỬ
# =======================================================
@app.get("/history")
async def history(limit: int = 50):
    return get_history(limit)

# =======================================================
# CHẠY BẰNG CLI
# =======================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test phân tích cảm xúc CLI.")
    parser.add_argument("--model", default=None, help="Model HuggingFace dùng để phân tích")
    parser.add_argument("--text", default=None, help="Câu cần phân tích")
    args = parser.parse_args()

    if args.text:
        pipe = get_pipeline(args.model)
        tok = vietnamese_segment(args.text)
        print("Câu đã chuẩn hoá:", tok)
        print("Kết quả:", pipe(tok))
    else:
        print("Không có --text. Hãy chạy FastAPI bằng lệnh:")
        print("uvicorn vi_sentiment_app:app --reload --port 8000")
