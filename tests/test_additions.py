from spiritbench.stimuli.adapter import antipode, negate_lines, gregory_wrap


def test_antipode_reflects_through_center():
    assert antipode((0.75, 0.20)) == (0.25, 0.80)
    assert antipode((0.5, 0.5)) == (0.5, 0.5)


def test_negate_lines_negates_every_line_in_order():
    out = negate_lines(["grasping at straws", "fleeing the dark"])
    assert out == ["not grasping at straws", "nor fleeing the dark"]


def test_gregory_wrap_blocks_preserve_order():
    lines = [f"phrase {i}" for i in range(7)]
    out = gregory_wrap(lines)
    assert out[0] == "there is phrase 0"
    assert out[1] == "and phrase 1"
    assert out[4] == "this is phrase 4"      # closing of first block of 5
    assert out[5] == "there is phrase 5"     # second block opens
    assert out[-1] == "this is phrase 6"
    # every source phrase appears exactly once, in order
    def strip_frame(l):
        for pre in ("there is ", "this is ", "and "):
            if l.startswith(pre):
                return l[len(pre):]
        return l
    core = [strip_frame(l) for l in out]
    assert core == [f"phrase {i}" for i in range(7)]


def test_valley_steps_ladder(toy_artifact_dir):
    from spiritbench.stimuli.adapter import load_art, valley_steps
    art = load_art(str(toy_artifact_dir / "toy.json"))
    flat = valley_steps(art, (0.75, 0.2), 6, seed=1, n_steps=0)
    deep = valley_steps(art, (0.75, 0.2), 6, seed=1, n_steps=3)
    assert len(flat) == 6 and len(deep) == 6
    # flat litany stays in the target band; deep version visits other regions
    vas_flat = [art.va(i) for i in flat]
    assert all(abs(v - 0.75) <= 0.16 for v, a in vas_flat)
    assert flat != deep
