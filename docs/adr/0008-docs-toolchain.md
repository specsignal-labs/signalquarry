# 0008. Documentation toolchain

Status: accepted (2026-09-25)

## Context

Docs must serve humans and coding agents and stay Python-only.

## Decision

MkDocs Material with mkdocstrings; content is plain Markdown so the generator can be swapped. `llms.txt` and `llms-full.txt` are generated from the docs, CLI catalog and schemas, and served offline by `sqy docs --llms`.

## Consequences

No Node toolchain in this repository.
