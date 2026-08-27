# Qwen Coding Rules

Follow these rules every session:

1. Inspect the existing project, dependencies, configuration, and Git state before writing code.
2. State what you tested. Never claim something works without running a relevant check.
3. Verify imports, installed package versions, and library APIs before using them.
4. Never invent classes, functions, package names, parameters, or model identifiers.
5. Prefer stable, documented libraries and existing project patterns over guessed APIs.
6. Make small, focused edits. Do not rewrite working code or add unnecessary files.
7. Keep configuration connected to runtime code; remove or wire unused settings.
8. After every feature or major change, run a focused test or smoke check.
9. Separate “implemented” from “verified working.” Mark a feature verified only after testing it.
10. Do not call a project complete when only scaffolding exists.
11. Update README and status documents only after verifying the behavior they describe.
12. Make small commits with meaningful messages. Ask before large architectural changes.

## RAG Rules

- Verify both retrieval and generation before calling a RAG chatbot complete.
- Preserve document metadata and source provenance.
- Prevent duplicate vector ingestion with stable IDs or equivalent replacement logic.
- Test Chroma persistence by writing, reloading, and retrieving.
- Keep the retrieval layer independent from any specific generation provider.
- Handle errors explicitly. Do not hide failures behind misleading fallback output.