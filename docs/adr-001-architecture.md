# ADR-001: Core architecture decisions

Status: Accepted (Team G24, November 2026)

## Context
We are building a multilingual travel assistant (English, Hindi, Telugu) that answers from documents (RAG), resists prompt injection, and performs limited actions through tools. It must be achievable by two students in about 11 weeks.

## Decisions
1. LLM-as-a-Judge runs offline only. It never authorizes actions.
2. Security screening runs before the main LLM call and again at later stages. Unicode script detection is a hint only, not a language classifier.
3. The LLM returns one Pydantic-validated object: an answer with citations, or a proposed tool call. Pydantic checks shape, not whether an action is allowed.
4. A Python tool gateway enforces: allow-list, argument schema, permissions, business rules, confirmation.
5. Booking changes use server-side pending actions: tied to the user, expiring, single-use, bound to exact parameters, consumed atomically in SQLite.
6. Retrieved chunks are untrusted data. They are scanned and isolated in the prompt. They can never authorize an action.
7. Tool arguments must come from the user's message or from our own trusted server data (e.g. earlier search results by ID), never from raw document text.
8. User identity comes only from the server session, never from the LLM.
9. Security layers: normalisation, rules, injection classifier, structural isolation, output/leak checks, tool gateway. Layers are measured separately.
10. Stack: Python, FastAPI, Pydantic v2, Gemini behind an LLMClient interface, sentence-transformers (multilingual-e5-small), ChromaDB, SQLite, pytest, React + Vite later. LangChain is skipped initially.
11. Five tools: search_hotels, check_availability, get_booking_details, create_test_booking, cancel_test_booking. Plain Python first, MCP after.
12. Separate evaluation datasets: normal RAG, multilingual, direct injection, indirect injection, unauthorized tool requests, harmless look-alike prompts.

## Consequences
- More code than a simple chatbot, but each security claim can be tested.
- MCP is part of the final demonstration but built after the plain tools work.
- Authentication is simulated (fake users); the report must say so.
