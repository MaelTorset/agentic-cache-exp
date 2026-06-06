# Security Policy

## Supported Versions

This project is pre-1.0. Security fixes should target the latest `main` branch.

## Reporting Issues

Please open a private security advisory on GitHub if the repository is hosted there. If private advisories are not available, open an issue with minimal reproduction details and avoid posting secrets, private prompts, credentials, or proprietary traces.

## Data Handling Notes

Agent traces can contain sensitive information. Before sharing examples or benchmark data:

- remove API keys, cookies, tokens, credentials, and hostnames;
- remove private source code unless it is intentionally public;
- avoid committing raw agent conversations from real production work;
- prefer synthetic fixtures under `examples/` and `benchmarks/`.
