"""Certificate helpers for CA retrieval and trust bootstrap.

Compute SHA-256 from certificate DER, never transport headers; authenticate
against an out-of-band fingerprint.
"""

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import Encoding

FINGERPRINT_PREFIX = "sha256:"


def parse_single_certificate(pem: str) -> x509.Certificate:
    """Parse exactly one X.509 certificate.

    Reject bundles: parsers may ignore trailing certificates while OS trust tools
    install them all, allowing an unverified root beside a verified one.

    Raises:
        ValueError: The input contains zero, multiple or unparseable certificates.
    """
    certs = x509.load_pem_x509_certificates(pem.encode())
    if len(certs) != 1:
        raise ValueError(f"expected exactly one certificate, got {len(certs)}")
    return certs[0]


def ca_fingerprint(pem: str) -> str:
    """Return sha256:<hex> of the certificate's DER encoding.

    Raises:
        ValueError: The PEM is not exactly one parseable certificate.
    """
    cert = parse_single_certificate(pem)
    return FINGERPRINT_PREFIX + cert.fingerprint(hashes.SHA256()).hex()


def canonical_pem(pem: str) -> str:
    """Serialize the verified single certificate to canonical PEM bytes.

    Write this representation to the trust store, never extra data from the fetched body.
    """
    return parse_single_certificate(pem).public_bytes(Encoding.PEM).decode()


def normalize_fingerprint(value: str) -> str:
    """Normalize a user-supplied fingerprint for comparison.

    Accepts optional `sha256:` prefix, colon separators, and mixed case —
    all the shapes `openssl x509 -fingerprint` and browsers display.
    """
    v = value.strip().lower()
    if v.startswith(FINGERPRINT_PREFIX):
        v = v[len(FINGERPRINT_PREFIX) :]
    return FINGERPRINT_PREFIX + v.replace(":", "")
