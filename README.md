# 🏥 MediSuite Agent

**AI-powered multi-agent system for automated medical claim generation and processing.**

MediSuite Agent uses a pipeline of 5 specialized AI agents to validate patient data, look up medical codes, verify insurance coverage, generate CMS-1500 claim forms, and make triage decisions — all powered by Azure AI Foundry with local fallback support.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![Azure](https://img.shields.io/badge/Azure_AI-Foundry-0078D4?logo=microsoft-azure)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- **5 Specialized AI Agents** — Each handles a distinct step in the claim processing pipeline
- **Interactive Web Dashboard** — Real-time stats, claim management, and agent monitoring
- **Chat with Agent** — Conversational chatbot for code lookups, claim queries, and policy checks
- **CMS-1500 PDF Generation** — Automated claim form creation via ReportLab
- **RAG Knowledge Base** — Azure Cognitive Search-backed knowledge retrieval
- **Azure Blob Storage** — Cloud-based artifact storage with local fallback
- **6 Sample Patient Datasets** — Demonstrates Approve, Deny, and Review outcomes

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Web UI / CLI                         │
├─────────────────────────────────────────────────────────┤
│                    Orchestrator                         │
├────────┬────────┬──────────┬───────────┬────────────────┤
│Patient │Document│Coverage  │Claim      │Triage          │
│Data    │Code    │Validation│Generation │Agent           │
│Agent   │Agent   │Agent     │Agent      │                │
├────────┴────────┴──────────┴───────────┴────────────────┤
│              RAG Knowledge Base                         │
├─────────────────────────────────────────────────────────┤
│  Azure AI Foundry  │  Blob Storage  │  Cognitive Search │
└─────────────────────────────────────────────────────────┘
```

### Agent Pipeline

| Step | Agent | Purpose |
|------|-------|---------|
| 1 | 🧑‍⚕️ **Patient Data Agent** | Validates patient info, extracts diagnoses and procedures from clinical notes |
| 2 | 📋 **Document Code Agent** | Looks up ICD-10 and CPT-4 codes, calculates charges |
| 3 | 🛡️ **Coverage Validation Agent** | Checks insurance policy validity, coverage status, and service eligibility |
| 4 | 📄 **Claim Generation Agent** | Creates CMS-1500 PDF forms, uploads to Azure Blob Storage |
| 5 | ⚖️ **Triage Agent** | Makes approve/deny/review decisions with risk assessment and confidence scoring |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- (Optional) Azure AI Foundry subscription for AI-powered responses

### Setup

```bash
# Clone the repository
git clone https://github.com/ponkumarpandian/capstone-ai-agentic-training.git
cd capstone-ai-agentic-training

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configure Azure (Optional)

Copy `.env.example` to `.env` and fill in your Azure credentials:

```bash
cp .env.example .env
```

```env
PROJECT_ENDPOINT=https://your-project.cognitiveservices.azure.com/
MODEL_DEPLOYMENT_NAME=gpt-4o
STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
BLOB_CONTAINER_NAME=medisuite-claims
SEARCH_ENDPOINT=https://your-search.search.windows.net
SEARCH_INDEX_NAME=medisuite-knowledge-base
```

> **Note:** Without Azure credentials, all agents use built-in local fallback logic. The system is fully functional offline.

### Run

```bash
# Web Server (recommended)
python server.py
# Open http://localhost:8000

# CLI
python main.py --patient data/sample_patient.json --notes data/sample_clinical_notes.txt
```

---

## 📸 Screenshots

### Dashboard
Stats overview with total claims, approval rates, and recent claims table.

### Claims List
All processed claims with decision badges, validation status, ICD-10 codes, and timestamps.

### Claim Detail
Drill into any claim to see agent-by-agent execution details, downloadable artifacts, and audit logs.

### Chat with Agent
Interactive chatbot for querying medical codes, checking insurance policies, and reviewing claim summaries.

---

## 📂 Project Structure

```
Capstone/
├── agents/                     # AI Agent implementations
│   ├── base_agent.py           # Abstract base class with Azure AI integration
│   ├── patient_data_agent.py   # Patient validation & diagnosis extraction
│   ├── document_code_agent.py  # ICD-10 / CPT-4 code lookup
│   ├── coverage_validation_agent.py  # Insurance policy validation
│   ├── claim_generation_agent.py     # CMS-1500 PDF generation
│   ├── triage_agent.py         # Approve/Deny/Review decisions
│   └── chat_handler.py         # Chatbot intent routing
├── data/                       # Sample data & lookup databases
│   ├── sample_patient.json     # Sample patient (John Doe)
│   ├── patient_jane_smith.json # Migraine case (Approve)
│   ├── patient_bob_johnson.json# Expired policy (Deny)
│   ├── patient_maria_garcia.json # Unknown insurance (Deny)
│   ├── patient_robert_williams.json # Surgical case (Review)
│   ├── icd10_codes.json        # 23 ICD-10 diagnosis codes
│   ├── cpt4_codes.json         # 18 CPT-4 procedure codes
│   └── policy_database.json    # 3 insurance policies
├── rag/                        # RAG Knowledge Base
│   └── knowledge_base.py       # Azure Cognitive Search integration
├── storage/                    # Blob Storage client
│   └── blob_storage.py         # Azure Blob Storage wrapper
├── templates/                  # Web UI
│   └── index.html              # Single Page Application (SPA)
├── utils/                      # Utilities
│   └── pdf_generator.py        # CMS-1500 PDF via ReportLab
├── config.py                   # Configuration & settings
├── orchestrator.py             # Agent pipeline orchestration
├── server.py                   # FastAPI web server
├── main.py                     # CLI entry point
├── requirements.txt            # Python dependencies
└── .env.example                # Environment variable template
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI (SPA) |
| `GET` | `/api/dashboard` | Dashboard statistics |
| `GET` | `/api/claims` | List all claims |
| `GET` | `/api/claims/{id}` | Claim details |
| `GET` | `/api/claims/{id}/runs` | Agent execution details |
| `GET` | `/api/claims/{id}/artifacts` | Claim artifacts (PDFs, blobs) |
| `GET` | `/api/claims/{id}/pdf` | Download CMS-1500 PDF |
| `POST` | `/api/workflow` | Process a new claim (multipart form) |
| `POST` | `/api/chat` | Send a chat message |
| `GET` | `/api/knowledge` | Knowledge base entries |
| `GET` | `/api/audit` | Audit log |
| `GET` | `/api/settings` | System configuration |
| `GET` | `/api/health` | Health check |

---

## 🧪 Sample Claims

| Patient | Insurance | Expected Outcome | Why |
|---------|-----------|-------------------|-----|
| John Doe | HealthCare Inc. (Valid) | ✅ Approve | Valid policy, all checks pass |
| Jane Smith | BlueCross Shield (Valid) | 🟡 Review | Valid policy, migraine case flagged |
| Bob Johnson | Aetna Health (Expired) | ❌ Deny | Policy expired |
| Maria Garcia | Unknown Insurance | ❌ Deny | Policy not in database |
| Robert Williams | HealthCare Inc. (Valid) | 🟡 Review | Valid policy, high-cost surgery ($28.5K) |
| Malformed Data | HealthCare Inc. | 🟡 Review | Missing clinical documentation |

---

## 🛠️ Tech Stack

- **Backend:** Python 3.12, FastAPI, Uvicorn
- **AI:** Azure AI Foundry (GPT-4o), Azure AI Agent Service
- **Search:** Azure Cognitive Search (RAG)
- **Storage:** Azure Blob Storage
- **PDF:** ReportLab
- **Frontend:** Vanilla HTML/CSS/JS (Single Page Application)
- **Auth:** Azure DefaultAzureCredential

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
