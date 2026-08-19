# Nerdit releases

Install with:

```
curl -fsSL https://get.nerdit.ai | sh
```

This repository carries **release assets only** — no source. Each tag publishes
the platform tarballs, a `SHA256SUMS` manifest, an ECDSA P-256 `SHA256SUMS.sig`
over it, and the verifying public key `nerdit-release.pub.pem`.

The installer verifies the signature **and** the checksum before extracting
anything. To check by hand:

```
V=0.5.0
BASE=https://github.com/nerdit-ai/releases/releases/download/v$V
curl -fsSLO $BASE/nerdit-$V-linux-x86_64.tar.gz
curl -fsSLO $BASE/SHA256SUMS
curl -fsSLO $BASE/SHA256SUMS.sig
curl -fsSLO $BASE/nerdit-release.pub.pem

openssl dgst -sha256 -verify nerdit-release.pub.pem \
  -signature SHA256SUMS.sig SHA256SUMS
sha256sum -c --ignore-missing SHA256SUMS
```

The verifying key is published twice on purpose: as the asset above, and inlined
in each release's notes. The installer embeds the same key.

| Platform | Artifact |
|---|---|
| Linux x86_64 | `nerdit-<version>-linux-x86_64.tar.gz` |
| Linux arm64 | `nerdit-<version>-linux-arm64.tar.gz` |
| macOS arm64 | `nerdit-<version>-macos-arm64.tar.gz` |
