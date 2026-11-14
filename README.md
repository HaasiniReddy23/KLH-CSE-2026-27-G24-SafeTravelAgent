# Safe Multilingual Travel Booking Agent

**Approved project heading:** P16 – Safe Tool-Calling Travel Booking Agent with Allow-Lists  
**Extended project scope:** RAG, multilingual tokenization, prompt-injection defense, MCP tool execution, and LLM-as-a-Judge evaluation.

## Team details

| Role | Name | University ID |
|---|---|---|
| Team member | Haasini Reddy B | 2520030265 |
| Team member | Arjun Tomar | 2520030285 |

**Team ID:** G24  
**Branch:** CSE  
**Academic year:** 2026–2027  
**Supervisor:** Dr. D. Bheekya

## Abstract

This project proposes a secure, multilingual AI travel assistant that answers travel-related questions using Retrieval-Augmented Generation (RAG) over trusted travel documents and can perform selected travel operations through controlled MCP tools. The assistant is intended to understand supported multilingual queries, retrieve relevant information such as travel policies and cancellation rules, and provide grounded responses. Because user prompts and retrieved documents may contain malicious or irrelevant instructions, the system will include prompt-injection screening and security tests. Tool calls will be restricted through an explicit allow-list, input validation, permission checks, and user confirmation for sensitive actions. An LLM-as-a-Judge component will assist in evaluating response quality and detecting unsafe outputs; it will not replace deterministic security controls.

The initial implementation will use simulated or sandbox travel services unless a real provider integration is explicitly approved and configured.

## Core objectives

1. Build a multilingual query interface for travel-related questions.
2. Retrieve grounded answers from a curated collection of travel documents using RAG.
3. Detect and test prompt-injection attempts in user inputs and retrieved content.
4. Expose a limited set of travel operations through MCP tools.
5. Enforce allow-listed tools, schema validation, authorization checks, and confirmation before consequential actions.
6. Evaluate retrieval quality, answer quality, injection resistance, and tool-call safety.

## Planned modules

- `src/multilingual/` – language identification and multilingual text handling.
- `src/rag/` – document ingestion, retrieval, and grounded response flow.
- `src/security/` – input screening, untrusted-content handling, and injection test cases.
- `src/mcp_tools/` – MCP server and allow-listed travel tools.
- `src/evaluation/` – LLM-as-a-Judge and evaluation metrics.
- `src/main.py` – application entry point (to be implemented).

## Repository structure

```text
.
├── src/
│   ├── evaluation/
│   ├── mcp_tools/
│   ├── multilingual/
│   ├── rag/
│   ├── security/
│   └── main.py
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   └── problem_statement.md
├── data/
│   └── README.md
├── results/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── tests/
│   └── README.md
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Requirements:
- Python 3.10 or newer
- Git
- An LLM provider API key only if using a hosted model
- Optional: a local model runtime or sandbox travel API, depending on the implementation

Create and activate a virtual environment:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local environment file from the example:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Add configuration values to `.env` locally. **Never commit `.env`, API keys, passwords, tokens, or private data.**

## Execution

The application entry point and modules are starter placeholders at this stage. Once implementation is added, the intended launch command is:

```bash
python -m src.main
```

Run tests after they are implemented:

```bash
pytest
```

## Current project phase

**Phase:** Initial repository setup / project planning  
**Status:** Starter structure created; implementation and evaluation are pending.

```bash
git tag review-1
git push origin review-1
```

- Keep the repository accessible to the supervisor and Course Coordinator.
- Do not rename or transfer the repository after submitting its URL in Section D without written approval.
