# Safe Multilingual Travel Assistant

Team G24 | KLH CSE | Generative AI (25CS2207E) | 2026-27
Members: Haasini Reddy B (2520030265), Arjun Tomar (2520030285). Supervisor: Dr. D. Bheekya

A travel assistant using RAG (English, Hindi, Telugu), prompt-injection defense, and allow-listed tool calling (MCP) with mock data.

## Status
- [x] Milestone 1: FastAPI skeleton, LLMClient (Gemini + fake), tests
- [ ] RAG, multilingual, security, tools, MCP, evaluation, frontend

## Run locally
1. `python -m venv .venv` then activate it (Windows: `.venv\Scripts\Activate.ps1`, Mac/Linux: `source .venv/bin/activate`)
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` (leave `LLM_PROVIDER=fake` to run without a key)
4. `uvicorn src.main:app --reload`
5. Open http://127.0.0.1:8000/docs
6. Tests: `python -m pytest`

Never commit `.env`.
