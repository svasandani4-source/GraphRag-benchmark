"""OllamaLLM against a mocked Ollama server (no Ollama needed)."""
import json

import httpx

from grb.eval.judge import JudgeVerdict
from grb.llm import OllamaLLM, make_llm


def _llm(handler):
    llm = OllamaLLM("qwen2.5:7b-instruct", host="http://ollama.test")
    llm.http = httpx.Client(transport=httpx.MockTransport(handler))
    return llm


def test_complete_returns_first_line_and_zero_cost():
    def handler(req):
        body = json.loads(req.content)
        assert body["options"]["temperature"] == 0 and "format" not in body
        return httpx.Response(200, json={"message": {"content": "Brennock\nbecause..."},
                                         "prompt_eval_count": 120, "eval_count": 5})
    text, usage = _llm(handler).complete("sys", "Question: x")
    assert text == "Brennock"
    assert (usage.input_tokens, usage.output_tokens, usage.cost_usd) == (120, 5, 0.0)


def test_parse_sends_schema_and_retries_bad_json():
    calls = []

    def handler(req):
        body = json.loads(req.content)
        assert body["format"]["type"] == "object"
        calls.append(1)
        content = "{not json" if len(calls) == 1 else json.dumps({"rationale": "same", "correct": True})
        return httpx.Response(200, json={"message": {"content": content}, "prompt_eval_count": 10, "eval_count": 3})
    verdict, usage = _llm(handler).parse("sys", "u", JudgeVerdict)
    assert verdict.correct is True and usage.calls == 2


def test_make_llm_prefixes():
    assert make_llm("ollama:llama3.1:8b").model == "llama3.1:8b"
    assert make_llm("fake").model == "fake"
