# VoXuanMai_Seminar_AI

Dự án này bao gồm mô hình AI phân tích cảm xúc tiếng Việt và API + giao diện web để sử dụng mô hình PhoBERT.

---

## 1. Chuẩn bị môi trường

### **1.1. Yêu cầu hệ thống**

* Python **3.10 – 3.11** (khuyến nghị 3.11)
* Pip
* Git (nếu muốn push mô hình lên HuggingFace)

---

## 2. Tạo môi trường ảo

### **Windows (PowerShell/CMD)**

```bash
python -m venv venv
venv\Scripts\activate
```

### **macOS/Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Cài đặt thư viện từ `requirements.txt`

```bash

pip install -r requirements.txt

```


## 4. Chạy API (app.py)

Đảm bảo đã cập nhật đường dẫn

```python
pipe = pipeline("text-classification", model="./models/vietnamese-sentiment-model")
```

Sau đó chạy Server:

```bash

python -m uvicorn Server:app --reload --port 8000

```

API sẽ chạy tại:

```

http://localhost:8000

```

---

## 5. Chạy giao diện Web

Sau khi API đang chạy, mở file HTML UI hoặc chạy server tĩnh:

```bash
python -m http.server 8080
```

Rồi truy cập:

```
http://localhost:8080/home.html
```

---

## 7. Cấu trúc Pipeline xử lý khi người dùng gọi API

```
User → UI → FastAPI (app.py) → Text cleaning
                                 ↓
                         underthesea normalize
                                 ↓
                         Tokenizer (model tokenizer)
                                 ↓
                         Model inference (pipeline)
                                 ↓
                      Sentiment + score + metadata
                                 ↓
                              Trả về UI
```

## 8. Tác giả

* **Xuân Mai**

---

## Giấy phép

MIT License
