"""Engine reverse-engineering experiment: can an LLM agent rebuild an ARC-AGI-3
game engine from a recorded run's observations? See engine_re/README.md."""

PLAY_VERSION = "v12"  # the play harness's version, written to a run's config.json (run_play) with the git short sha
