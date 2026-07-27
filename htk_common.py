#!/usr/bin/env python3
"""Shared constants and helpers for the HTK miner toolchain.

Everything in here is derived from the verified on-chain source of
HashToken at 0xE5544a2A5fA9b175da60D8Eec67adD5582bB31b0 (Solidity 0.3.5, 2016):

    function mint(bytes32 value) {
        if (uint(sha3(value, prev_hash)) > max_value) { throw; }
        balances[msg.sender] += 10 ** 16;
        prev_hash = sha3(block.blockhash(block.number), prev_hash);
        max_value -= max_value / 100;
        Mint(msg.sender);
    }

Three consequences drive the whole design:

1. A nonce is valid iff  keccak256(nonce || prev_hash) <= max_value.
   Both operands are bytes32, so sha3(a, b) is plain concatenation.

2. `throw` in Solidity 0.3.5 compiles to the INVALID opcode, which burns the
   ENTIRE gas limit. A lost race costs the full limit, not just gas used.
   Verified on-chain: successful mints use exactly 38,721 gas; reverted mints
   consume 100% of their limit. Hence: pre-verify, and keep the limit tight.

3. `block.blockhash(block.number)` is ALWAYS zero in the EVM (you can only read
   hashes of the previous 256 blocks, not the current one). So the chain update
   is  prev_hash = keccak256(bytes32(0) || prev_hash)  -- fully deterministic.
   Verified against archive state across the mint in block 25,600,440.
"""
from __future__ import annotations

import json
import os
import time

CONTRACT = "0xE5544a2A5fA9b175da60D8Eec67adD5582bB31b0"

# Function selectors, all confirmed against the deployed bytecode.
SEL_MINT = "0xadf2cead"       # mint(bytes32)
SEL_PREV_HASH = "0xc69b5df2"  # prev_hash()
SEL_MAX_VALUE = "0x98597629"  # max_value()

# event Mint(address indexed minter)
TOPIC_MINT = "0x3c3284d117c92d0b1699230960384e794dcba184cc48ff114fe4fed20c9b0565"

DECIMALS = 16
REWARD_UNITS = 10 ** 16          # mint() credits 10**16 base units ...
REWARD_TOKENS = 1.0              # ... which is exactly 1.0 HTK at 16 decimals.

INITIAL_MAX_VALUE = 2 ** 255     # set in the 2016 constructor
GAS_SUCCESS = 38_721             # exact, observed on every successful mint
DEFAULT_GAS_LIMIT = 60_000       # tight on purpose: a revert burns all of it

NTFY_BASE = "https://ntfy.sh"
DEFAULT_NONCE_TOPIC = "htk-nonce-a303558a1aa9e6043d43531d"
DEFAULT_STATUS_TOPIC = "htk-status-35cb5d56831536e9924deb7b"

READ_RPCS = [
    "https://ethereum-rpc.publicnode.com",
    "https://eth.llamarpc.com",
    "https://cloudflare-eth.com",
    "https://eth.drpc.org",
    "https://1rpc.io/eth",
]

# Private relays. mint(bytes32) does not bind the nonce to msg.sender, so a
# public-mempool transaction lets anyone copy the nonce and front-run it.
# Sending through a private relay is the only real defence.
PRIVATE_RELAYS = {
    "flashbots": "https://rpc.flashbots.net/fast",
    "mevblocker": "https://rpc.mevblocker.io/fast",
    "beaver": "https://rpc.beaverbuild.org",
}


# --------------------------------------------------------------------------
# keccak256
# --------------------------------------------------------------------------
def keccak256(data: bytes) -> bytes:
    """keccak256 (the pre-NIST padding Ethereum uses), via whichever lib exists."""
    try:
        from Crypto.Hash import keccak as _k
        h = _k.new(digest_bits=256)
        h.update(data)
        return h.digest()
    except ImportError:
        pass
    try:
        from eth_utils import keccak as _ek
        return _ek(data)
    except ImportError:
        pass
    import sha3  # pysha3
    return sha3.keccak_256(data).digest()


def hash_value(nonce: bytes, prev_hash: bytes) -> int:
    """The exact quantity the contract compares against max_value."""
    if len(nonce) != 32 or len(prev_hash) != 32:
        raise ValueError("nonce and prev_hash must both be 32 bytes")
    return int.from_bytes(keccak256(nonce + prev_hash), "big")


def is_valid(nonce: bytes, prev_hash: bytes, max_value: int) -> bool:
    return hash_value(nonce, prev_hash) <= max_value


def next_prev_hash(prev_hash: bytes) -> bytes:
    """The prev_hash the contract will move to after the next successful mint.

    Deterministic because block.blockhash(block.number) is always zero.
    """
    return keccak256(b"\x00" * 32 + prev_hash)


def next_max_value(max_value: int) -> int:
    """Difficulty after one more mint: max_value -= max_value / 100 (integer div)."""
    return max_value - max_value // 100


def project_state(prev_hash: bytes, max_value: int, mints_ahead: int):
    """State after `mints_ahead` further mints by anyone. Lets you mine ahead."""
    for _ in range(mints_ahead):
        prev_hash = next_prev_hash(prev_hash)
        max_value = next_max_value(max_value)
    return prev_hash, max_value


# Verdicts from classify_nonce().
VALID = "valid"
PREMATURE = "premature"
STALE = "stale"


def classify_nonce(nonce: bytes, prev_hash: bytes, max_value: int, lookahead: int = 200):
    """Explain a nonce: (verdict, mints_ahead, detail).

    prev_hash only advances, and deterministically, so a nonce that is not valid
    right now is either aimed at a state that has not arrived yet (mined ahead,
    worth holding) or at one already gone (we lost the race). Distinguishing the
    two matters: one is worth keeping, the other is worth discarding.
    """
    if is_valid(nonce, prev_hash, max_value):
        return VALID, 0, "valid against current chain state"

    p, m = prev_hash, max_value
    for ahead in range(1, lookahead + 1):
        p, m = next_prev_hash(p), next_max_value(m)
        if is_valid(nonce, p, m):
            return PREMATURE, ahead, (
                f"valid but PREMATURE: targets the state {ahead} mint(s) ahead "
                f"of now - hold it until then")

    over = hash_value(nonce, prev_hash) / max_value if max_value else float("inf")
    return STALE, 0, (
        f"does not satisfy the current target (hash is {over:.3e}x max_value) - "
        f"stale: another miner minted first and prev_hash moved on")


# --------------------------------------------------------------------------
# difficulty / economics
# --------------------------------------------------------------------------
def expected_hashes(max_value: int) -> float:
    """Mean hashes to find one valid nonce: 2**256 / max_value."""
    return (2 ** 256) / max_value


def mints_so_far(max_value: int) -> float:
    """Approximate number of mints that have happened, from the 1%/mint ratchet."""
    import math
    return math.log(max_value / INITIAL_MAX_VALUE) / math.log(0.99)


def cost_per_token(max_value: int, hashrate: float, dollars_per_hour: float) -> float:
    """Expected USD to win one token at a given hashrate and rental rate."""
    if hashrate <= 0:
        return float("inf")
    seconds = expected_hashes(max_value) / hashrate
    return seconds / 3600.0 * dollars_per_hour


def fmt_hashrate(h: float) -> str:
    for unit, div in (("TH/s", 1e12), ("GH/s", 1e9), ("MH/s", 1e6), ("kH/s", 1e3)):
        if h >= div:
            return f"{h / div:.2f} {unit}"
    return f"{h:.0f} H/s"


def fmt_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds or seconds == float("inf"):
        return "never"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


# --------------------------------------------------------------------------
# chain reads (dependency-light: plain JSON-RPC over requests)
# --------------------------------------------------------------------------
def eth_call(rpc: str, data: str, block: str = "latest", timeout: int = 10) -> bytes:
    import requests
    r = requests.post(
        rpc,
        json={"jsonrpc": "2.0", "id": 1, "method": "eth_call",
              "params": [{"to": CONTRACT, "data": data}, block]},
        timeout=timeout,
    )
    r.raise_for_status()
    body = r.json()
    if body.get("error"):
        raise RuntimeError(body["error"])
    result = body.get("result", "0x")
    return bytes.fromhex(result[2:] if result.startswith("0x") else result)


def read_state(rpc: str):
    """(prev_hash: bytes32, max_value: int) from a single RPC."""
    prev = eth_call(rpc, SEL_PREV_HASH)
    mx = eth_call(rpc, SEL_MAX_VALUE)
    if len(prev) != 32 or len(mx) != 32:
        raise ValueError("unexpected return length from contract")
    return prev, int.from_bytes(mx, "big")


def read_state_failover(rpcs=None, start: int = 0):
    """(prev_hash, max_value, rpc_index) trying each RPC in turn."""
    rpcs = rpcs or READ_RPCS
    errors = []
    for i in range(len(rpcs)):
        idx = (start + i) % len(rpcs)
        try:
            prev, mx = read_state(rpcs[idx])
            return prev, mx, idx
        except Exception as e:  # noqa: BLE001 - failover is the point
            errors.append(f"{rpcs[idx]}: {type(e).__name__}")
    raise RuntimeError("all read RPCs failed - " + "; ".join(errors))


# --------------------------------------------------------------------------
# ntfy
# --------------------------------------------------------------------------
def ntfy_push(topic: str, body: str, title=None, tags=None, priority=None, tries: int = 4) -> bool:
    import requests
    if not topic:
        return False
    headers = {}
    if title:
        headers["Title"] = title
    if tags:
        headers["Tags"] = tags
    if priority:
        headers["Priority"] = str(priority)
    delay = 1.0
    for _ in range(tries):
        try:
            requests.post(f"{NTFY_BASE}/{topic}", data=body.encode(),
                          headers=headers, timeout=15).raise_for_status()
            return True
        except Exception:  # noqa: BLE001
            time.sleep(delay)
            delay = min(delay * 2, 15)
    return False


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------
def load_env(path=None):
    """Load .env next to this file. Uses python-dotenv when available."""
    from pathlib import Path
    path = Path(path) if path else Path(__file__).resolve().parent / ".env"
    try:
        from dotenv import load_dotenv
        load_dotenv(path)
        return
    except ImportError:
        pass
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def parse_nonce(text: str) -> bytes | None:
    """Pull a 32-byte nonce out of a message body or a found_nonces.jsonl line."""
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("{"):
        try:
            text = json.loads(text).get("nonce", "")
        except Exception:  # noqa: BLE001
            return None
    text = text.strip()
    if text.startswith("0x") or text.startswith("0X"):
        text = text[2:]
    if len(text) != 64:
        return None
    try:
        return bytes.fromhex(text)
    except ValueError:
        return None


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)
