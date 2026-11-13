# Problem Statement

## Title

**P16 – Safe Tool-Calling Travel Booking Agent with Allow-Lists**

**Extended scope:** A secure multilingual travel assistant integrating Retrieval-Augmented Generation (RAG), prompt-injection defense, MCP-based tool execution, and LLM-as-a-Judge evaluation.

## Problem

Travel assistants based on large language models can produce unsupported answers when travel information is incomplete or outdated. They may also misunderstand multilingual queries, and they face security risks when malicious instructions appear in user prompts or retrieved documents. When connected to external services, an inadequately controlled agent may attempt unauthorized or incorrectly parameterized actions.

## Proposed solution

Develop a multilingual AI travel assistant that retrieves answers from a curated set of trusted travel documents, screens user inputs and retrieved content for prompt-injection attempts, and performs only explicitly permitted travel operations through MCP tools. Tool execution will be protected by allow-lists, strict argument schemas, authorization checks, and user confirmation for consequential actions. An LLM-as-a-Judge component will help assess answer relevance, grounding, and safety during evaluation.

## Scope

- Answer travel questions using RAG over approved travel documents.
- Support selected languages, initially English, Hindi, and Telugu, subject to model and tokenizer capability.
- Identify and safely handle suspicious instructions in user input and retrieved documents.
- Provide a controlled set of simulated or sandbox travel tools, such as searching stays, checking itinerary details, and managing a test booking.
- Evaluate answer quality, retrieval performance, prompt-injection resistance, and tool-call safety.

## Out of scope for the initial version

- Processing real payments.
- Making real bookings or cancellations without an approved provider integration and explicit confirmation.
- Guaranteeing detection of every prompt-injection attack.
- Treating an LLM judge as the sole security boundary.

## Expected outcome

A working prototype that demonstrates grounded multilingual travel assistance, safe handling of adversarial prompts and documents, and restricted MCP tool execution with auditable validation and confirmation steps.
