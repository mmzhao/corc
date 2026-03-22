"""Dispatch abstraction layer.

Abstracts agent dispatch behind an interface so the underlying LLM CLI
can be swapped (Claude Code → Gemini → Codex) by implementing one class.
"""

import json
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Constraints:
    allowed_tools: list[str] = field(default_factory=lambda: ["Read", "Edit", "Write", "Bash", "Grep", "Glob"])
    max_budget_usd: float = 3.0
    max_turns: int = 50
    output_format: str | None = None
    json_schema: str | None = None


@dataclass
class AgentResult:
    output: str
    exit_code: int
    duration_s: float


class AgentDispatcher(ABC):
    @abstractmethod
    def dispatch(self, prompt: str, system_prompt: str, constraints: Constraints) -> AgentResult:
        ...


class ClaudeCodeDispatcher(AgentDispatcher):
    def dispatch(self, prompt: str, system_prompt: str, constraints: Constraints) -> AgentResult:
        cmd = ["claude", "-p", prompt]

        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        if constraints.allowed_tools:
            cmd.extend(["--allowedTools", ",".join(constraints.allowed_tools)])

        if constraints.max_turns:
            cmd.extend(["--max-turns", str(constraints.max_turns)])

        start = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            duration = time.time() - start
            output = result.stdout
            if result.stderr:
                output += "\n--- STDERR ---\n" + result.stderr
            return AgentResult(
                output=output,
                exit_code=result.returncode,
                duration_s=duration,
            )
        except subprocess.TimeoutExpired:
            duration = time.time() - start
            return AgentResult(
                output="[TIMEOUT: agent exceeded 600s limit]",
                exit_code=-1,
                duration_s=duration,
            )


def get_dispatcher(provider: str = "claude-code") -> AgentDispatcher:
    dispatchers = {
        "claude-code": ClaudeCodeDispatcher,
    }
    if provider not in dispatchers:
        raise ValueError(f"Unknown dispatch provider: {provider}. Available: {list(dispatchers.keys())}")
    return dispatchers[provider]()
