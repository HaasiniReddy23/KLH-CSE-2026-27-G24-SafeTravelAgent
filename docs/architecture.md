# System Architecture

## High-level flow

1. **Query intake:** Receive the user's question and identify the selected/supported language.
2. **Input screening:** Apply deterministic checks and a prompt-injection classifier or judge to flag suspicious instructions. Do not rely on this step alone for security.
3. **RAG retrieval:** Search the approved travel-document collection and return relevant passages with source metadata.
4. **Context isolation:** Mark retrieved passages as untrusted reference material. Instructions inside documents must not override system policy or authorize actions.
5. **Response generation:** Generate a grounded answer with citations or document references where available.
6. **Response evaluation:** Use an LLM-as-a-Judge rubric to assess relevance, factual support, and policy compliance. A failed evaluation can trigger regeneration or a safe fallback.
7. **Action planning:** If an operation is needed, map the request to a known tool from the allow-list.
8. **Tool validation:** Validate the tool name, argument schema, user authorization, and business rules in ordinary application code.
9. **Confirmation:** Ask the user to confirm consequential actions.
10. **Execution and audit:** Execute only the validated tool and record a privacy-conscious audit event.

## Component boundaries

| Component | Responsibility |
|---|---|
| Multilingual layer | Language identification and query normalization |
| RAG layer | Document ingestion, chunking, embedding, retrieval, source references |
| Security layer | Prompt-injection screening and untrusted-content handling |
| MCP layer | Expose only approved travel operations |
| Policy/validation layer | Enforce allow-lists, schemas, authorization, and confirmation |
| Judge/evaluation layer | Score answer grounding, relevance, and safety |
| API/UI layer | Accept queries and present answers or confirmation requests |

## Initial safe tool allow-list

The team should implement only the tools needed for the prototype. Suggested mock tools:

- `search_destinations`
- `search_stays`
- `get_travel_policy`
- `get_test_booking`
- `create_test_booking` (confirmation required)
- `cancel_test_booking` (confirmation and ownership check required)

The names above are proposals, not existing integrations. Implement each tool with strict input schemas and mock data first.

## Key security rules

- The model may propose a tool call; application code decides whether it is permitted.
- Never execute arbitrary tool names, shell commands, SQL, or URLs supplied by the model or user.
- Validate every argument against a strict schema and business constraints.
- Retrieved documents are data, not instructions.
- Require authentication/ownership checks and explicit confirmation for consequential actions.
- Store secrets outside source control.
- Keep audit logs free of passwords, API keys, and unnecessary personal data.

## Evaluation signals

- Retrieval: Precision@k / Recall@k on a small labeled query set.
- Answers: factual support, citation correctness, relevance, and language appropriateness.
- Security: attack detection rate, false-positive rate, and successful unauthorized tool-call rate.
- Tools: schema-validation failures, blocked unauthorized calls, and confirmation enforcement.
