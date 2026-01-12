# 📄 FormExtract AI - OCR Form Data Extraction System

> **Intelligent form processing with Chandra OCR + Gemini 2.5 Flash Lite**

Extract structured key-value data from handwritten and printed forms with AI-powered accuracy.

---

## ⚡ Quick Start

### 1. Clone & Setup Environment

```bash
git clone <repository-url>
cd OCR-System

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings:
# - GEMINI_API_KEY=your_key_here
# - DB_PASSWORD=your_password
```

### 3. Start Database

**Option A: Docker (Recommended)**
```bash
docker-compose up -d postgres
```

**Option B: Local PostgreSQL**
- Install PostgreSQL 16
- Create database: `CREATE DATABASE ocr_system;`
- Update `.env` with your credentials

### 4. Initialize Database

```bash
python scripts/init_db.py
```

### 5. Run Application

```bash
# Terminal 1: Backend API
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2: Frontend UI
cd frontend
streamlit run app.py --server.port 8501
```

### 6. Open Application

- **Frontend**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **pgAdmin** (optional): http://localhost:5050

---

## 🏗️ Project Structure

```
OCR-System/
├── backend/          # FastAPI REST API
│   ├── api/          # Route handlers
│   ├── services/     # Business logic (OCR, Gemini, exports)
│   ├── database/     # PostgreSQL models & CRUD
│   └── schemas/      # Pydantic models
├── frontend/         # Streamlit UI
│   ├── pages/        # Multi-page app
│   └── components/   # Reusable widgets
├── storage/          # Local file storage
│   ├── uploads/      # Uploaded documents
│   ├── processed/    # OCR outputs
│   └── exports/      # Generated exports
├── scripts/          # Utility scripts
└── tests/            # Test suite
```

---

## 🔧 Database Commands

```bash
# Initialize (create tables + seed data)
python scripts/init_db.py

# Reset database (DESTRUCTIVE)
python scripts/init_db.py --reset

# Check connection only
python scripts/init_db.py --check

# Skip seed data
python scripts/init_db.py --no-seed
```

---

## 📋 Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Backend | FastAPI |
| OCR | Chandra OCR |
| LLM | Gemini 2.5 Flash Lite |
| Database | PostgreSQL |
| Exports | Excel, JSON, PDF |

---

## 📝 License

MIT License - See LICENSE file for details.
