import hashlib
import random

NEUTRAL_TEXT = (
    "To operate the dishwasher, first ensure the filter assembly is seated in the "
    "sump housing. Load plates between the tines facing the center. The detergent "
    "dispenser accepts powder or tablet formats; close the lid until it clicks. "
    "Select a cycle using the control panel. The normal cycle runs 2 hours 15 "
    "minutes at 130 degrees. The rinse aid reservoir should be refilled monthly. "
    "If error code E4 appears, check the inlet hose for kinks and confirm the "
    "water supply valve is fully open before restarting the unit."
)


def shuffled(stim: dict, seed: int) -> dict:
    rng = random.Random(seed)
    order = list(range(len(stim["lines"])))
    while True:
        rng.shuffle(order)
        if [stim["lines"][i] for i in order] != stim["lines"] or len(order) < 2:
            break
    lines = [stim["lines"][i] for i in order]
    text = ".\n".join(lines)
    out = dict(stim)
    out.update({
        "constructor": "shuffled:" + stim["constructor"],
        "lines": lines, "text": text,
        "waypoints": [stim["waypoints"][i] for i in order],
        "id": f"shuffled-{stim['id']}-" + hashlib.sha1(text.encode()).hexdigest()[:8],
    })
    return out


def neutral_stimulus(target_name, target_va) -> dict:
    return {"id": f"neutral-{target_name}", "constructor": "neutral",
            "generator": "none", "params": {}, "target": target_name,
            "target_va": list(target_va), "waypoints": [], "lines": [],
            "text": NEUTRAL_TEXT}
