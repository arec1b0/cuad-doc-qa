"""Answer a question from retrieved chunks: one chat call, parsed into an Answer.

Citations are returned exactly as the model claims them. Whether a quote really occurs in its
chunk is checked downstream, never here.
"""

import json
from collections.abc import Sequence

from cuad_doc_qa.domain import Answer, Chunk, Citation
from cuad_doc_qa.generation.ollama import ChatModel
from cuad_doc_qa.generation.prompt import answer_schema, build_messages, source_labels

CHARS_PER_TOKEN_FLOOR = 3
"""Conservative: English contract text averages about 4 characters per token."""


class Generator:
    def __init__(self, chat: ChatModel, num_ctx: int, reply_tokens: int = 1024) -> None:
        self._chat = chat
        self._max_prompt_chars = (num_ctx - reply_tokens) * CHARS_PER_TOKEN_FLOOR

    def answer(self, question: str, chunks: Sequence[Chunk]) -> Answer:
        """Raises ValueError if the prompt may not fit the context window or the reply is invalid.

        With no chunks there is nothing to cite, so the model is not called.
        """
        if not chunks:
            return Answer(found=False, text="", citations=())

        messages = build_messages(question, chunks)
        prompt_chars = sum(len(message["content"]) for message in messages)
        if prompt_chars > self._max_prompt_chars:
            raise ValueError(
                f"prompt has {prompt_chars} characters, over the {self._max_prompt_chars} "
                "that safely fit the context window; pass fewer chunks or raise num_ctx"
            )

        labels = source_labels(len(chunks))
        reply = self._chat.complete(messages, answer_schema(labels))
        return _parse(reply, dict(zip(labels, (chunk.chunk_id for chunk in chunks), strict=True)))


def _parse(reply: str, chunk_by_label: dict[str, str]) -> Answer:
    try:
        data = json.loads(reply)
        citations = tuple(
            Citation(chunk_id=chunk_by_label[item["source"]], quote=item["quote"])
            for item in data["citations"]
        )
        found, text = data["found"], data["answer"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"reply does not follow the answer schema: {reply[:200]!r}") from error
    if not isinstance(found, bool) or not isinstance(text, str):
        raise ValueError(f"reply does not follow the answer schema: {reply[:200]!r}")
    return Answer(found=found, text=text, citations=citations)
