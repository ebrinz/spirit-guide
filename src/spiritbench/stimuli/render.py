import json

META_PROMPT = (
    "Rewrite the following word sequence as a guided meditation in free verse. "
    "Preserve the emotional arc: the words are ordered waypoints from the current "
    "state toward a state of {target}. Use each waypoint word in order, one line "
    "per waypoint, weaving it into an evocative image. Do not add instructions, "
    "titles, or commentary — output only the poem lines, one per waypoint.\n\n"
    "Waypoint words, in order: {words}\n"
)


def render_prompt(stim: dict) -> str:
    return META_PROMPT.format(target=stim["target"], words=", ".join(stim["lines"]))


def make_prompt_batch(stims, out_path):
    rows = [s for s in stims if s["generator"] == "word-template"
            and s["params"].get("length") == "medium"
            and s["params"].get("intensity") == "plain"]
    with open(out_path, "w") as f:
        for s in rows:
            f.write(json.dumps({"stimulus_id": s["id"], "prompt": render_prompt(s)}) + "\n")
    return len(rows)


def ingest_renders(prompts_path, responses_path, stims_by_id, out_path):
    with open(responses_path) as f:
        responses = [json.loads(l) for l in f]
    n = 0
    with open(out_path, "w") as f:
        for r in responses:
            src = stims_by_id.get(r["stimulus_id"])
            lines = [l.strip() for l in r.get("text", "").splitlines() if l.strip()]
            if src is None or len(lines) < 4:
                print(f"SKIPPED render for {r.get('stimulus_id')}")
                continue
            rec = dict(src)
            rec.update({"id": "render-" + src["id"], "generator": "claude-render",
                        "lines": lines, "text": "\n".join(lines)})
            f.write(json.dumps(rec) + "\n")
            n += 1
    return n
