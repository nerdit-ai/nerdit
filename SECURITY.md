# Security Policy

## Supported versions

Security fixes are made against the latest release line only. Older
releases do not receive backported fixes; please upgrade to the latest
version before reporting an issue.

## Reporting a vulnerability

Please report suspected security vulnerabilities privately, using GitHub's
private security advisory feature on this repository (Security tab →
"Report a vulnerability"). **Do not open a public issue or pull request
for a security report** — that discloses the problem before a fix exists.

Nerdit is maintained solo, so acknowledgment is best-effort: expect an
initial response within 72 hours, not a guarantee. If you haven't heard
back in that window, it's fine to follow up on the same advisory thread.

## Disclosure process

Once a report is triaged, we'll work with you through the private
advisory to confirm the issue, develop and test a fix, and agree on a
disclosure timeline. The default coordinated-disclosure window is 90
days from the initial report, after which the advisory is published
(with credit, if you'd like it) whether or not a fix has shipped. We'll
aim to ship a fix and publish before that deadline, and are happy to
disclose earlier by mutual agreement once a fix is available.

## Scope

This policy covers the code in this repository: the daemon, CLI, MCP
server, and dashboard. It does not cover third-party dependencies —
please report those upstream — or the hosted cloud service, which has
its own reporting channel.
