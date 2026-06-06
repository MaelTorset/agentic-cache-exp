# Contributing

Thanks for your interest in Agentic Cache Lab.

This is an early research prototype. Small, measurable changes are preferred over large framework rewrites.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests
python scripts/run_benchmark.py
```

## Contribution Guidelines

- Keep the core package dependency-free unless a dependency is clearly necessary.
- Add or update a benchmark fixture when changing scoring or packing behavior.
- Keep prompt-routing decisions inspectable in JSON output.
- Avoid provider-specific assumptions in `segment_store.py`, `scorer.py`, and `packer.py`.
- Put runtime-specific integrations behind client or adapter boundaries.

## Pull Request Checklist

- Tests pass with `python -m unittest discover -s tests`.
- The smoke benchmark runs with `python scripts/run_benchmark.py`.
- New behavior is documented in `README.md` or `docs/`.
- No secrets, local machine paths, or private traces are committed.
