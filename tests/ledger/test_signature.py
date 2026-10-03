"""Signed ledger heads: the RFC 8032 test vectors, and the head check against a chain."""

import json

import pytest

from sentiment_agent.ledger.signature import (
    chain_hashes_from_jsonl,
    head_message,
    public_key,
    sign,
    sign_head,
    signed_heads_document,
    verify,
    verify_heads,
)

# RFC 8032 section 7.1, TEST 1 and TEST 2 (secret key, public key, message, signature).
VECTORS = [
    (
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
        "",
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
    ),
    (
        "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
        "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
        "72",
        "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
    ),
]


@pytest.mark.parametrize(("secret", "public", "message", "signature"), VECTORS)
def test_rfc8032_vectors(secret: str, public: str, message: str, signature: str) -> None:
    sk, msg = bytes.fromhex(secret), bytes.fromhex(message)
    assert public_key(sk).hex() == public
    assert sign(sk, msg).hex() == signature
    assert verify(bytes.fromhex(public), msg, bytes.fromhex(signature))


def test_tampering_fails_verification() -> None:
    sk = bytes.fromhex(VECTORS[1][0])
    pk = public_key(sk)
    sig = sign(sk, b"head")
    assert verify(pk, b"head", sig)
    assert not verify(pk, b"heaD", sig)
    flipped = bytes([sig[0] ^ 1]) + sig[1:]
    assert not verify(pk, b"head", flipped)
    assert not verify(pk, b"head", sig[:63])  # malformed: a failure, not an exception
    assert not verify(pk[:31], b"head", sig)


def test_heads_are_checked_against_their_key_and_the_chain() -> None:
    sk = bytes.fromhex(VECTORS[0][0])
    genesis = "ab" * 32
    ledger = [
        {"seq": 0, "hash": "00" * 32},
        {"seq": 1, "hash": "11" * 32},
        {"seq": 2, "hash": "22" * 32},
    ]
    chain = chain_hashes_from_jsonl("\n".join([json.dumps(e) for e in ledger] + ['{"seq": 3, "ha']))
    assert chain == {0: "00" * 32, 1: "11" * 32, 2: "22" * 32}  # the partial last line is skipped
    heads = [
        sign_head(sk, genesis, 1, "11" * 32, "2026-10-03T00:00:00Z"),
        sign_head(sk, genesis, 2, "22" * 32, "2026-10-03T01:00:00Z"),
    ]
    doc = signed_heads_document(public_key(sk), genesis, heads)
    assert all(c.ok for c in verify_heads(doc, chain))

    rewritten = {**chain, 2: "99" * 32}  # a chain rewritten after it was signed
    checks = verify_heads(doc, rewritten)
    assert checks[0].ok
    assert checks[1].signature_valid
    assert checks[1].matches_chain is False

    forged = json.loads(json.dumps(doc))
    forged["heads"][0]["hash"] = "33" * 32  # a head swapped without the key
    assert not verify_heads(forged, chain)[0].signature_valid

    assert verify_heads(doc, {1: "11" * 32})[1].matches_chain is None  # chain too short to say


def test_head_message_is_domain_separated() -> None:
    message = head_message("g", 7, "h")
    assert message == b"t2sa-ledger-head-v1\ng\n7\nh"
    with pytest.raises(ValueError, match="negative"):
        head_message("g", -1, "h")


def test_a_unicode_line_separator_inside_an_event_does_not_split_it() -> None:
    event = {"seq": 5, "hash": "55" * 32, "payload": {"text": "crowd post second line\x85"}}
    raw = json.dumps(event, ensure_ascii=False)
    assert len(raw.splitlines()) > 1  # what str.splitlines would have done
    assert chain_hashes_from_jsonl(raw + "\n") == {5: "55" * 32}
