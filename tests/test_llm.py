"""Tests for the OpenAI wrappers using a mocked client (no network).

Python note: ``unittest.mock.MagicMock`` creates an object that accepts any attribute
access or call and records it - similar to ``jest.fn()`` / ``vi.fn()``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from docmind.llm import OpenAIChatModel, OpenAIEmbedder


def _embedding_response(vectors_with_index):
    data = [SimpleNamespace(index=i, embedding=v) for i, v in vectors_with_index]
    return SimpleNamespace(data=data)


def test_embedder_batches_and_preserves_order():
    client = MagicMock()
    # Second batch returns items out of order to prove we sort by index.
    client.embeddings.create.side_effect = [
        _embedding_response([(0, [0.0]), (1, [1.0])]),
        _embedding_response([(1, [3.0]), (0, [2.0])]),
    ]
    embedder = OpenAIEmbedder(client, model="text-embedding-3-small", batch_size=2)

    vectors = embedder.embed(["a", "b", "c", "d"])

    assert vectors == [[0.0], [1.0], [2.0], [3.0]]
    assert client.embeddings.create.call_count == 2
    first_call = client.embeddings.create.call_args_list[0].kwargs
    assert first_call == {"model": "text-embedding-3-small", "input": ["a", "b"]}


def test_chat_model_sends_system_and_user_messages():
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="  Answer [S1]  "))]
    )
    model = OpenAIChatModel(client, model="gpt-4o-mini")

    assert model.complete("SYS", "USER") == "Answer [S1]"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0.0
    assert kwargs["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
