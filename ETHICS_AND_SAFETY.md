# Ethics and Safety

## Purpose Statement

This framework is designed exclusively for **defensive security research** and **authorized vulnerability assessment** of applications you own or have explicit written permission to test.

It is **not** intended for:
- Unauthorized testing of third-party systems
- Real-world attack automation
- Circumventing security controls on production systems
- Any activity violating applicable laws or regulations

## Authorized Targets

Default configuration permits only:

| Target Type | Examples |
|-------------|----------|
| Localhost | `http://localhost:3000` (Juice Shop) |
| Loopback | `127.0.0.1`, `::1` |
| Private lab networks | `192.168.x.x`, `10.x.x.x` (when enabled) |
| Intentionally vulnerable apps | OWASP Juice Shop, DVWA, WebGoat |

Public internet targets are **blocked by default** (`allow_public_internet: false`).

## Safety Controls

| Control | Default | Description |
|---------|---------|-------------|
| Safe mode | `true` | Enables all safety features |
| Target allowlist | localhost | Only approved hosts/ports |
| Rate limiting | 5 req/s | Prevents aggressive scanning |
| Max requests/episode | 200 | Request budget per episode |
| Emergency stop | 3 violations | Halts on repeated scope violations |
| Action logging | `true` | Full audit trail per episode |
| Scope guard | enabled | Blocks out-of-scope URLs |

## Configuration for Authorized Testing

To test on a private lab server:

```yaml
environment:
  base_url: "http://192.168.1.100:8080"

safety:
  safe_mode: true
  allow_public_internet: false
  allow_private_networks: true
  allowed_hosts:
    - "192.168.1.100"
    - "localhost"
```

**Never** set `allow_public_internet: true` unless operating in an isolated, authorized test environment.

## Responsible Disclosure

Findings from lab targets should be used to:
1. Understand vulnerability classes and detection methods
2. Improve defensive tooling and remediation workflows
3. Publish research results with appropriate ethical framing

Do not use findings from lab exercises to attack real systems.

## Research Ethics

When publishing results from this framework:
- Frame the work as defensive vulnerability assessment
- Document authorization and lab-only scope
- Include safety control descriptions
- Acknowledge limitations of heuristic detection
- Do not release pre-trained models as "attack tools"

## Reporting Issues

If you discover that this framework could be misused against unauthorized targets, please report via the project's issue tracker so additional safeguards can be implemented.

## Legal Notice

Users are solely responsible for ensuring compliance with all applicable laws, regulations, and organizational policies. The authors disclaim liability for misuse.
