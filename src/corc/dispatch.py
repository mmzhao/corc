"""Dispatch abstraction layer.

Abstracts agent dispatch behind an interface so the underlying LLM CLI
can be swapped (Claude Code → Gemini → Codex) by implementing one class.

Streaming dispatch: ClaudeCodeDispatcher uses --output-format stream-json
to parse stdout line-by-line as JSON events, enabling real-time visibility
into agent reasoning and tool calls.
"""

import json
import logging
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


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


# Type alias for the streaming event callback
EventCallback = Callable[[dict], None]


class AgentDispatcher(ABC):
    @abstractmethod
    def dispatch(self, prompt: str, system_prompt: str, constraints: Constraints,
                 pid_callback=None, event_callback: EventCallback | None = None,
                 cwd: str | None = None) -> AgentResult:
        """Dispatch an agent with the given prompt and constraints.

        Args:
            pid_callback: Optional callable(pid: int) called with the process
                PID once the agent subprocess starts. Used for PID tracking
                during reconciliation on restart.
            event_callback: Optional callable(event: dict) called for each
                streaming event parsed from the agent's stdout. Enables
                real-time session logging, audit logging, and TUI updates.
            cwd: Optional working directory for the agent subprocess.
                Used for git worktree isolation — each agent runs in its
                own worktree so parallel agents don't conflict.
        """
        ...


class ClaudeCodeDispatcher(AgentDispatcher):
    def dispatch(self, prompt: str, system_prompt: str, constraints: Constraints,
                 pid_callback=None, event_callback: EventCallback | None = None,
                 cwd: str | None = None) -> AgentResult:
        cmd = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose"]

        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        if constraints.allowed_tools:
            cmd.extend(["--allowedTools", ",".join(constraints.allowed_tools)])

        if constraints.max_turns:
            cmd.extend(["--max-turns", str(constraints.max_turns)])

        start = time.time()
        timed_out = False

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
        )

        # Report PID so it can be tracked for reconciliation
        if pid_callback:
            pid_callback(proc.pid)

        # Read stderr in a background thread to prevent pipe deadlock
        # (stdout is read line-by-line in the main loop; if stderr fills
        # its pipe buffer the process would block writing to it)
        stderr_chunks: list[str] = []

        def _drain_stderr():
            for line in proc.stderr:
                stderr_chunks.append(line)

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        # Watchdog: kill process if it exceeds timeout
        def _kill_on_timeout():
            nonlocal timed_out
            timed_out = True
            try:
                proc.kill()
            except OSError:
                pass

        timer = threading.Timer(1800, _kill_on_timeout)
        timer.start()

        result_text = ""
        event_count = 0
        try:
            # Read stdout line-by-line for real-time streaming.
            # iter(readline, '') avoids internal buffering that
            # `for line in proc.stdout` can introduce.
            for line in iter(proc.stdout.readline, ""):
                line = line.strip()
                if not line:
                    continue
                logger.debug("stream raw stdout: %s", line[:500])
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("stream non-JSON line (skipped): %s", line[:200])
                    continue

                event_type = event.get("type", "unknown")
                logger.debug("stream event #%d type=%s", event_count, event_type)
                event_count += 1

                if event_callback:
                    event_callback(event)

                # Extract final output from the result event
                if event_type == "result":
                    result_text = event.get("result", "")

            proc.wait()
        finally:
            timer.cancel()

        stderr_thread.join(timeout=5)
        logger.info("stream dispatch finished: %d events parsed, exit_code=%s",
                     event_count, proc.returncode)

        if timed_out:
            return AgentResult(
                output="[TIMEOUT: agent exceeded 1800s limit]",
                exit_code=-1,
                duration_s=time.time() - start,
            )

        duration = time.time() - start
        output = result_text
        stderr = "".join(stderr_chunks)
        if stderr:
            output += "\n--- STDERR ---\n" + stderr

        return AgentResult(
            output=output,
            exit_code=proc.returncode,
            duration_s=duration,
        )


def get_dispatcher(provider: str = "claude-code") -> AgentDispatcher:
    dispatchers = {
        "claude-code": ClaudeCodeDispatcher,
    }
    if provider not in dispatchers:
        raise ValueError(f"Unknown dispatch provider: {provider}. Available: {list(dispatchers.keys())}")
    return dispatchers[provider]()
