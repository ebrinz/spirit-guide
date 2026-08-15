import json
from spiritbench.stimuli.render import render_prompt, ingest_renders


def _stim():
    return {"id": "s1", "constructor": "valley", "generator": "word-template",
            "target": "calm", "target_va": [0.75, 0.2],
            "params": {"length": "medium", "intensity": "plain", "style": "unfiltered"},
            "waypoints": [{"node": 0, "v": 0.6, "a": 0.4}],
            "lines": ["stone", "river", "rest"], "text": "stone.\nriver.\nrest"}


def test_render_prompt_contains_words_and_target():
    p = render_prompt(_stim())
    assert "stone" in p and "river" in p and "rest" in p
    assert "calm" in p


def test_ingest_creates_render_records(tmp_path):
    stim = _stim()
    prompts = tmp_path / "prompts.jsonl"
    responses = tmp_path / "responses.jsonl"
    out = tmp_path / "renders.jsonl"
    prompts.write_text(json.dumps({"stimulus_id": "s1", "prompt": "x"}) + "\n")
    responses.write_text(json.dumps(
        {"stimulus_id": "s1", "text": "line one\nline two\nline three\nline four"}) + "\n")
    ingest_renders(prompts, responses, {"s1": stim}, out)
    rec = json.loads(out.read_text().strip())
    assert rec["generator"] == "claude-render"
    assert rec["id"] == "render-s1"
    assert rec["lines"] == ["line one", "line two", "line three", "line four"]
    assert rec["waypoints"] == stim["waypoints"]
