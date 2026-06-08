from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from openrouter import OpenRouter
except ImportError:  # pragma: no cover - optional runtime dependency
    OpenRouter = None  # type: ignore[assignment]

from core.config import MemoryConfig
from main import Memory
from utils.logger import get_logger, log_exception


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(PROJECT_ROOT / ".env")


MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
SYSTEM_INSTRUCTION = MemoryConfig.load_system_prompt()


def format_context(results: list[dict[str, object]]) -> str:
    """Format retrieved chunks for inclusion in the chat prompt."""

    if not results:
        return "No relevant context was found."

    formatted: list[str] = []
    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
        content = metadata.get("content", "") if isinstance(metadata, dict) else ""
        score = result.get("score", 0.0) if isinstance(result, dict) else 0.0
        document_id = metadata.get("document_id", "") if isinstance(metadata, dict) else ""
        chunk_index = metadata.get("chunk_index", "") if isinstance(metadata, dict) else ""

        header_parts = [f"[{index}] score={score:.4f}"]
        if document_id != "":
            header_parts.append(f"document_id={document_id}")
        if chunk_index != "":
            header_parts.append(f"chunk_index={chunk_index}")

        formatted.append(f"{' '.join(header_parts)}\n{content}")

    return "\n\n".join(formatted)


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue

            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
                continue

            if isinstance(item, dict):
                maybe_text = item.get("text")
                if isinstance(maybe_text, str):
                    parts.append(maybe_text)

        return "".join(parts).strip()

    return str(content).strip()


def build_client() -> OpenRouter | None:
    """Create a configured OpenRouter client from environment variables.

    Returns None when OpenRouter is not configured so the CLI can still run in
    local-context mode instead of failing at startup.
    """

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

    if OpenRouter is None:
        return None

    return OpenRouter(
        api_key=api_key,
        http_referer=os.getenv("OPENROUTER_HTTP_REFERER"),
        x_open_router_title=os.getenv("OPENROUTER_APP_TITLE"),
        x_open_router_categories=os.getenv("OPENROUTER_APP_CATEGORIES"),
    )


def ask_llm(client: OpenRouter | None, *, user_prompt: str, context: str) -> str:
    """Send a single chat request and return the generated text."""

    if client is None:
        if context.strip() and context != "No relevant context was found.":
            return (
                "OpenRouter is not configured, so I can only surface the retrieved context.\n\n"
                f"{context}"
            )

        return "OpenRouter is not configured and no relevant context was found."

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
    ]

    if context.strip() and context != "No relevant context was found.":
        messages.append({"role": "system", "content": f"Retrieved context:\n{context}"})

    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.send(
        model=MODEL_NAME,
        messages=messages,
    )

    if not response.choices:
        raise RuntimeError("OpenRouter returned no choices")

    answer = _message_content_to_text(response.choices[0].message.content)
    if not answer:
        raise RuntimeError("OpenRouter returned an empty response")

    return answer


def _remember_turn(memory: Memory, *, turn_index: int, user_prompt: str, assistant_reply: str) -> None:
    turn_document = (
        f"Conversation turn {turn_index}\n"
        f"User: {user_prompt}\n"
        f"Assistant: {assistant_reply}"
    )

    asyncio.run(
        memory.save(
            document_id=f"conversation-turn-{turn_index}",
            content=turn_document,
        )
    )


def main() -> None:
    """Run an interactive retrieval-augmented chat loop."""

    logger = get_logger(__name__, subsystem="app.cli")
    memory = Memory()
    client = build_client()

    seed_document = os.getenv("MEMORYOS_SEED_DOCUMENT", "")
    if seed_document.strip():
        asyncio.run(
            memory.save(
                document_id="seed-document",
                content=seed_document,
            )
        )

    turn_index = 0

    while True:
        user = input("USER: ").strip()

        if not user:
            continue

        if user.lower() == "exit":
            break

        retrieval_results = memory.retrieve(
            query=user,
            top_k=10,
            score_threshold=0.1,
        )
        context = format_context(retrieval_results)

        try:
            answer = ask_llm(client, user_prompt=user, context=context)
            # UX: print to console for interactive users
            print("AI:", answer)
            # Structured log for observability
            logger.info("ai.response", extra={"answer": answer})
            turn_index += 1
            _remember_turn(
                memory,
                turn_index=turn_index,
                user_prompt=user,
                assistant_reply=answer,
            )
        except Exception as exc:
            log_exception(logger, message="ai.error", exc=exc)
            print(f"AI error: {exc}")


if __name__ == "__main__":
    main()

