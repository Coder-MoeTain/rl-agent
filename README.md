# Production Autonomous Pentester

**10/10 production-ready** RL autonomous web application pentester for authorized targets.

```bash
make install && make validate && make smoke
```

## Commands

```bash
autopentest validate --offline              # pre-deploy check
autopentest scan --mock -o auto             # offline smoke test
autopentest scan --authorized -t URL -o auto # live production scan
autopentest report -i reports/run_x/engagement.json
make live                                   # Docker target + live scan
```

## Outputs

Each run exports: `engagement.json`, `engagement.md`, `engagement.sarif`, `metrics.json`, `audit.json`

## Authorization

Live scans require `--authorized` or `PENTEST_AUTHORIZED=1`. Mock mode does not.

Full scorecard: **[PRODUCTION.md](PRODUCTION.md)**
