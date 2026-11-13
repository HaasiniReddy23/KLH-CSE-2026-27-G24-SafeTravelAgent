# Methodology and Implementation Plan

## Phase 1 – Planning and baseline

- Finalize supported languages, travel use cases, document sources, and mock tool scope.
- Prepare a small, licensed/public travel-document collection.
- Define normal, ambiguous, and adversarial test queries.
- Establish baseline answer and retrieval tests.

## Phase 2 – Multilingual RAG

- Implement document loading, cleaning, chunking, embeddings, and vector retrieval.
- Add language identification and multilingual query handling.
- Generate answers grounded in retrieved passages.
- Display source references and handle insufficient evidence safely.

## Phase 3 – Security and red-teaming

- Build a test set of direct prompt injections, indirect injections in documents, role-confusion attempts, and requests for secrets or unauthorized actions.
- Add input screening and clear separation between trusted instructions and untrusted retrieved content.
- Record true positives, false positives, and missed attacks.
- Add safe fallback behavior for suspicious or uncertain cases.

## Phase 4 – MCP tools and policy enforcement

- Implement a small MCP server with mock travel operations.
- Create a server-side allow-list and strict argument schemas.
- Add authorization, booking ownership checks, business-rule validation, and confirmation for consequential actions.
- Test that blocked or invalid calls cannot reach the mock service.

## Phase 5 – LLM-as-a-Judge and evaluation

- Define a documented rubric for grounding, relevance, language quality, and safety.
- Compare judge scores with a manually reviewed sample.
- Treat judge output as an evaluation signal, not an authorization mechanism.
- Run regression tests after prompt, model, retrieval, or tool changes.

## Phase 6 – Integration and reporting

- Integrate the modules behind the API/UI.
- Run end-to-end tests for normal and adversarial scenarios.
- Document limitations, metrics, screenshots, and reproducible setup steps.
- Prepare review and final reports.

## Minimum demonstration

1. Ask a travel-policy question in English, Hindi, or Telugu.
2. Show the retrieved document references supporting the answer.
3. Submit a prompt-injection attempt and show safe handling.
4. Include a malicious instruction in a test document and show that it is treated as untrusted content.
5. Search mock stays or destinations through an MCP tool.
6. Attempt an unapproved tool call and show that the allow-list blocks it.
7. Demonstrate a test booking that requires confirmation.
8. Show the evaluation report and security test results.
