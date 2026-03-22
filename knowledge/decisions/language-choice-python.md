---
id: decision-language-python
type: decision
project: corc
repos: [corc]
tags: [language, python, performance, ai-code-generation]
created: 2026-03-21T00:00:00Z
updated: 2026-03-21T00:00:00Z
source: human
status: active
---

# Language Choice: Python as Primary Language

## Context

CORC needs a primary implementation language for the orchestration system. The system is CLI-based, relies heavily on SQLite, integrates with AI/ML libraries (sentence-transformers for embeddings), and will be primarily written and maintained by Claude Code agents. The choice of language directly affects both human developer productivity and AI code generation speed/cost.

A March 2026 benchmark by Yusuke Endoh ("Which Programming Language Is Best for Claude Code?") tested Claude Code Opus 4.6 implementing a simplified Git in 13 languages, with 20 trials each. This provided quantitative data on AI code generation performance across languages.

## Decision

**Python is the primary language for CORC.**

### Benchmark Results

- **Top 3**: Ruby (73s, $0.36), Python (75s, $0.38), JavaScript (81s, $0.39) — fast, cheap, low variance
- **Mid tier**: Go (102s, $0.50), Java (115s, $0.50), Rust (114s, $0.54) — 1.4-2x slower, much higher variance
- **Bottom tier**: TypeScript (133s, $0.62), C (156s, $0.74), Haskell (174s, $0.74)

### Key Insight

Dynamic languages consistently outperform static ones for AI code generation. Type checking adds 1.6-3.2x overhead in generation time and cost. Python specifically sits in the sweet spot: fast generation, low variance, and the best ecosystem for our needs.

### Why Python Over Ruby

While Ruby was marginally faster in the benchmark (73s vs 75s), Python wins on ecosystem:

- **Best AI/ML ecosystem**: sentence-transformers, PyTorch, numpy — all native Python
- **Excellent SQLite tooling**: built-in sqlite3 module, well-tested
- **CLI frameworks**: Click is mature and well-suited for our needs
- **TUI libraries**: Rich and Textual for the operator dashboard
- **Ubiquitous**: every developer and every AI model knows Python well

## Consequences

### What Becomes Easier
- AI agents generate Python code faster and cheaper than most alternatives
- Direct access to the ML ecosystem for embeddings and future features
- SQLite integration is trivial with the built-in module
- Rich ecosystem of CLI and TUI libraries
- Lower variance in AI output quality — more predictable agent behavior

### What Becomes Harder
- No compile-time type safety (mitigated by tests and runtime validation)
- Performance ceiling for compute-heavy operations (mitigated: our bottleneck is LLM latency, not CPU)
- Dependency management can be fragile (mitigated by minimal dependencies and optional extras)

## Alternatives Considered

- **Ruby**: Slightly faster AI generation but weaker ML ecosystem. No sentence-transformers equivalent.
- **JavaScript/TypeScript**: Good ecosystem but TypeScript's type system adds 1.6x overhead for AI generation. Node.js SQLite story is less mature.
- **Go**: Strong for CLI tools but 1.4x slower/more expensive for AI generation. No ML ecosystem.
- **Rust**: Excellent performance but 1.5x slower for AI generation with much higher variance. Wrong trade-off for a system whose bottleneck is LLM latency.
