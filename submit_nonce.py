#!/usr/bin/env python3
"""Local HTK mint submitter.

Runs on YOUR machine, never on a rented GPU. The miner finds nonces and pushes
them to ntfy; this script picks them up and sends the mint() transaction, so the
private key never leaves home.

Why it is built the way it is (all verified against the deployed contract):

  * mint(bytes32) does NOT bind the nonce to msg.sender. Anyone who sees your
    transaction in the public mempool can copy the nonce and front-run you for
    the token. So we send through a private relay by default.

  * A failed mint burns 100% of the gas limit -- Solidity 0.3.5 `throw` is the
    INVALID opcode. We therefore (a) verify the nonce locally, (b) re-read
    prev_hash immediately before sending, (c) run eth_estimateGas as a final
    pre-flight, and (d) keep the gas limit tight (60k vs the 38,721 actually
    needed) so a lost race is cheap.

  * prev_hash advances deterministically, so when a nonce turns out to be stale
    we can say exactly how many mints it missed by, rather than just "stale".

Usage:
    python3 submit_nonce.py                 # watch ntfy + found_nonces.jsonl
    python3 submit_nonce.py --once 0x<64hex> # submit a single nonce
    python3 submit_nonce.py --dry-run       # verify + simulate, never send
    python3 submit_nonce.py --status        # print wallet/chain status and exit
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import htk_common as H
from htk_common import log

SCRIPT_DIR = Path(__file__).resolve().parent
FOUND_LOG = SCRIPT_DIR / "found_nonces.jsonl"
MINTED_LOG = SCRIPT_DIR / "minted.jsonl"
STATE_FILE = SCRIPT_DIR / ".submit_state.json"

# How far ahead to look when deciding whether a nonce is merely stale.
STALE_LOOKAHEAD = 200


class Submitter:
    def __init__(self, args):
        H.load_env(args.env)
        self.args = args

        self.rpc_url = os.environ.get("MAINNET_RPC_URL", "").strip()
        if not self.rpc_url:
            raise SystemExit(
                "MAINNET_RPC_URL is not set. Copy .env.example to .env and fill it in."
            )

        raw_contract = os.environ.get("HTK_CONTRACT", H.CONTRACT).strip()
        self.nonce_topic = os.environ.get("NTFY_TOPIC", H.DEFAULT_NONCE_TOPIC).strip()
        self.status_topic = os.environ.get("NTFY_STATUS_TOPIC", H.DEFAULT_STATUS_TOPIC).strip()

        # Send path: private relay unless explicitly overridden.
        relay = (args.relay or os.environ.get("SUBMIT_RELAY", "flashbots")).strip()
        if relay == "public":
            self.send_url = self.rpc_url
            self.relay_name = "PUBLIC MEMPOOL"
        elif relay in H.PRIVATE_RELAYS:
            self.send_url = H.PRIVATE_RELAYS[relay]
            self.relay_name = relay
        elif relay.startswith("http"):
            self.send_url = relay
            self.relay_name = "custom"
        else:
            raise SystemExit(
                f"unknown relay {relay!r}; use one of "
                f"{', '.join(H.PRIVATE_RELAYS)}, 'public', or an https:// URL"
            )

        from web3 import Web3
        self.Web3 = Web3
        self.contract = Web3.to_checksum_address(raw_contract)
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url, request_kwargs={"timeout": 30}))
        self.w3_send = Web3(Web3.HTTPProvider(self.send_url, request_kwargs={"timeout": 60}))

        key = os.environ.get("MINER_PRIVATE_KEY", "").strip()
        self.account = None
        if key:
            from eth_account import Account
            if not key.startswith("0x"):
                key = "0x" + key
            self.account = Account.from_key(key)

        self.gas_limit = int(os.environ.get("GAS_LIMIT", H.DEFAULT_GAS_LIMIT))
        self.max_gas_gwei = float(os.environ.get("MAX_GAS_GWEI", "50"))
        self.priority_gwei = float(os.environ.get("PRIORITY_FEE_GWEI", "0.05"))

        self.seen = self._load_seen()
        # Nonces mined for a future chain state: valid, just not yet. Kept in
        # memory and retried as the chain advances.
        self.pending: dict[str, float] = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()

    # ---------------------------------------------------------------- state
    def _load_seen(self) -> set:
        if STATE_FILE.exists():
            try:
                return set(json.loads(STATE_FILE.read_text()).get("seen", []))
            except Exception:  # noqa: BLE001
                pass
        return set()

    def _save_seen(self):
        try:
            STATE_FILE.write_text(json.dumps({"seen": sorted(self.seen)}, indent=1))
        except Exception as e:  # noqa: BLE001
            log(f"warn: could not persist state: {e}")

    # ---------------------------------------------------------------- chain
    def chain_state(self):
        """(prev_hash, max_value) straight from the configured RPC."""
        prev = self.w3.eth.call({"to": self.contract, "data": H.SEL_PREV_HASH})
        mx = self.w3.eth.call({"to": self.contract, "data": H.SEL_MAX_VALUE})
        return bytes(prev), int.from_bytes(bytes(mx), "big")

    def classify(self, nonce: bytes, prev: bytes, mx: int):
        """(ok, verdict, detail) - explains *why* a nonce fails, not just that it does."""
        verdict, _ahead, detail = H.classify_nonce(nonce, prev, mx, STALE_LOOKAHEAD)
        return verdict == H.VALID, verdict, detail

    # ---------------------------------------------------------------- submit
    def submit(self, nonce: bytes, source: str = "manual") -> bool:
        key = nonce.hex()
        with self.lock:
            if key in self.seen:
                log(f"skip 0x{key[:16]}... already handled")
                return False

        log(f"--- nonce 0x{key} (from {source})")

        # 1. Re-read chain state right now. Never trust the miner's snapshot.
        try:
            prev, mx = self.chain_state()
        except Exception as e:  # noqa: BLE001
            log(f"  ERROR reading chain state: {e}")
            return False
        log(f"  prev_hash  0x{prev.hex()}")
        log(f"  max_value  0x{mx:064x}")

        # 2. Local validity check -- free, and saves a full gas limit on a revert.
        ok, verdict, detail = self.classify(nonce, prev, mx)
        if not ok:
            log(f"  REJECTED: {detail}")
            if verdict == H.PREMATURE:
                # Do NOT mark it seen: it becomes submittable once the chain
                # catches up, so park it and let the retry loop pick it up.
                with self.lock:
                    self.pending[key] = time.time()
                H.ntfy_push(self.status_topic,
                            f"nonce 0x{key[:16]}... is premature - holding it\n{detail}",
                            title="HTK nonce held", tags="hourglass")
            else:
                with self.lock:
                    self.seen.add(key)
                    self._save_seen()
                H.ntfy_push(self.status_topic,
                            f"nonce 0x{key[:16]}... rejected: {detail}",
                            title="HTK nonce stale", tags="warning")
            return False
        with self.lock:
            self.pending.pop(key, None)
        log(f"  local check OK: {detail}")

        if not self.account:
            log("  MINER_PRIVATE_KEY not set - cannot sign. Nonce is valid; set the key to submit.")
            return False

        data = H.SEL_MINT + key

        # 3. Pre-flight simulation against latest state.
        try:
            est = self.w3.eth.estimate_gas({
                "to": self.contract, "from": self.account.address, "data": data,
            })
            log(f"  estimate_gas OK: {est} (expect ~{H.GAS_SUCCESS})")
        except Exception as e:  # noqa: BLE001
            log(f"  REJECTED: estimate_gas reverted ({str(e)[:120]}) - would burn the full gas limit")
            with self.lock:
                self.seen.add(key)
                self._save_seen()
            return False

        if self.args.dry_run:
            log("  --dry-run: verified and simulated, not sending")
            return True

        return self._send(nonce, data)

    def _send(self, nonce: bytes, data: str) -> bool:
        key = nonce.hex()
        attempts = int(self.args.retries)
        bump = 1.0

        for attempt in range(1, attempts + 1):
            try:
                latest = self.w3.eth.get_block("latest")
                base = latest.get("baseFeePerGas", 0) or 0
                prio = int(self.priority_gwei * 1e9 * bump)
                max_fee = int(base * 2 + prio)
                cap = int(self.max_gas_gwei * 1e9)
                if max_fee > cap:
                    log(f"  gas too high: needs {max_fee/1e9:.2f} gwei > MAX_GAS_GWEI={self.max_gas_gwei}")
                    time.sleep(12)
                    continue

                tx_nonce = self.w3.eth.get_transaction_count(self.account.address, "pending")
                tx = {
                    "chainId": self.w3.eth.chain_id,
                    "to": self.Web3.to_checksum_address(self.contract),
                    "from": self.account.address,
                    "data": data,
                    "gas": self.gas_limit,
                    "maxFeePerGas": max_fee,
                    "maxPriorityFeePerGas": prio,
                    "nonce": tx_nonce,
                    "value": 0,
                }
                signed = self.account.sign_transaction(tx)
                raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction

                worst = self.gas_limit * max_fee / 1e18
                log(f"  sending via {self.relay_name}: tx_nonce={tx_nonce} "
                    f"maxFee={max_fee/1e9:.2f} gwei worst-case={worst:.5f} ETH (attempt {attempt}/{attempts})")

                tx_hash = self.w3_send.eth.send_raw_transaction(raw)
                h = tx_hash.hex()
                if not h.startswith("0x"):
                    h = "0x" + h
                log(f"  submitted: {h}")

                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=self.args.wait)
                gas_used = receipt["gasUsed"]
                eff = receipt.get("effectiveGasPrice", max_fee)
                cost_eth = gas_used * eff / 1e18

                if receipt["status"] == 1:
                    log(f"  *** MINTED *** block {receipt['blockNumber']} "
                        f"gas={gas_used} cost={cost_eth:.6f} ETH")
                    self._record(key, h, receipt, cost_eth, True)
                    H.ntfy_push(self.status_topic,
                                f"MINTED 1 HTK\ntx {h}\ncost {cost_eth:.6f} ETH",
                                title="HTK mint confirmed", tags="tada", priority=4)
                    with self.lock:
                        self.seen.add(key)
                        self._save_seen()
                    return True

                log(f"  REVERTED in block {receipt['blockNumber']} - "
                    f"burned {gas_used} gas ({cost_eth:.6f} ETH). Lost the race.")
                self._record(key, h, receipt, cost_eth, False)
                H.ntfy_push(self.status_topic,
                            f"mint reverted (lost race)\ntx {h}\nburned {cost_eth:.6f} ETH",
                            title="HTK mint failed", tags="warning", priority=4)
                with self.lock:
                    self.seen.add(key)
                    self._save_seen()
                return False

            except Exception as e:  # noqa: BLE001
                msg = str(e)[:200]
                log(f"  send attempt {attempt} failed: {msg}")
                if attempt < attempts:
                    # Someone else may already have minted; re-verify before retrying.
                    try:
                        prev, mx = self.chain_state()
                        if not H.is_valid(nonce, prev, mx):
                            log("  chain moved on - nonce no longer valid, abandoning")
                            with self.lock:
                                self.seen.add(key)
                                self._save_seen()
                            return False
                    except Exception:  # noqa: BLE001
                        pass
                    bump *= 1.5
                    time.sleep(min(4 * attempt, 20))

        log("  giving up after all retries")
        return False

    def _record(self, nonce_hex, tx_hash, receipt, cost_eth, success):
        rec = {
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "nonce": "0x" + nonce_hex,
            "tx": tx_hash,
            "block": receipt["blockNumber"],
            "gas_used": receipt["gasUsed"],
            "cost_eth": cost_eth,
            "success": success,
            "reward_htk": H.REWARD_TOKENS if success else 0.0,
        }
        with open(MINTED_LOG, "a", buffering=1) as f:
            f.write(json.dumps(rec) + "\n")

    # ---------------------------------------------------------------- watchers
    def watch_ntfy(self):
        """Stream the ntfy topic. Reconnects forever with backoff."""
        import requests
        url = f"{H.NTFY_BASE}/{self.nonce_topic}/json"
        backoff = 1.0
        while not self.stop.is_set():
            try:
                log(f"[ntfy] subscribing to {self.nonce_topic}")
                with requests.get(url, stream=True, timeout=(10, 90),
                                  params={"since": "10m"}) as r:
                    r.raise_for_status()
                    backoff = 1.0
                    for line in r.iter_lines():
                        if self.stop.is_set():
                            return
                        if not line:
                            continue
                        try:
                            msg = json.loads(line)
                        except Exception:  # noqa: BLE001
                            continue
                        if msg.get("event") != "message":
                            continue
                        nonce = H.parse_nonce(msg.get("message", ""))
                        if nonce:
                            self.submit(nonce, source="ntfy")
                        else:
                            log(f"[ntfy] ignoring non-nonce message: {msg.get('message','')[:60]}")
            except Exception as e:  # noqa: BLE001
                if self.stop.is_set():
                    return
                log(f"[ntfy] stream error ({type(e).__name__}), reconnecting in {backoff:.0f}s")
                self.stop.wait(backoff)
                backoff = min(backoff * 2, 60)

    def watch_file(self):
        """Tail found_nonces.jsonl, for when the miner runs on this same box."""
        pos = 0
        if FOUND_LOG.exists() and not self.args.replay:
            pos = FOUND_LOG.stat().st_size
            log(f"[file] tailing {FOUND_LOG.name} from byte {pos}")
        else:
            log(f"[file] watching {FOUND_LOG.name} (from start)" if self.args.replay
                else f"[file] waiting for {FOUND_LOG.name}")
        while not self.stop.is_set():
            try:
                if FOUND_LOG.exists():
                    size = FOUND_LOG.stat().st_size
                    if size < pos:
                        pos = 0
                    if size > pos:
                        with open(FOUND_LOG) as f:
                            f.seek(pos)
                            for line in f:
                                nonce = H.parse_nonce(line)
                                if nonce:
                                    self.submit(nonce, source="found_nonces.jsonl")
                            pos = f.tell()
            except Exception as e:  # noqa: BLE001
                log(f"[file] error: {e}")
            self.stop.wait(3.0)

    def retry_pending(self):
        """Re-check held (premature) nonces as the chain advances."""
        while not self.stop.is_set():
            if self.stop.wait(20.0):
                return
            with self.lock:
                due = list(self.pending)
            if not due:
                continue
            try:
                prev, mx = self.chain_state()
            except Exception:  # noqa: BLE001
                continue
            for key in due:
                nonce = bytes.fromhex(key)
                verdict, ahead, _ = H.classify_nonce(nonce, prev, mx, STALE_LOOKAHEAD)
                if verdict == H.VALID:
                    log(f"[pending] chain caught up to 0x{key[:16]}... submitting")
                    self.submit(nonce, source="pending")
                elif verdict == H.STALE:
                    log(f"[pending] 0x{key[:16]}... went stale, dropping")
                    with self.lock:
                        self.pending.pop(key, None)
                        self.seen.add(key)
                        self._save_seen()

    # ---------------------------------------------------------------- status
    def status(self):
        import math
        prev, mx = self.chain_state()
        exp = H.expected_hashes(mx)
        print()
        print("  HTK chain status")
        print(f"    contract    {self.contract}")
        print(f"    prev_hash   0x{prev.hex()}")
        print(f"    max_value   0x{mx:064x}")
        print(f"    mints so far ~{H.mints_so_far(mx):.0f}")
        print(f"    expected hashes/token  {exp:.3e}  (2^{math.log2(exp):.1f})")
        print(f"    next prev_hash (deterministic)  0x{H.next_prev_hash(prev).hex()}")
        print()
        print("  submission path")
        print(f"    read RPC    {self.rpc_url.split('/v2/')[0]}...")
        print(f"    send via    {self.relay_name} ({self.send_url})")
        print(f"    gas limit   {self.gas_limit}  (success costs {H.GAS_SUCCESS}; a revert burns the whole limit)")
        if self.account:
            bal = self.w3.eth.get_balance(self.account.address)
            tok = self.w3.eth.call({
                "to": self.contract,
                "data": "0x70a08231" + "00" * 12 + self.account.address[2:].lower(),
            })
            held = int.from_bytes(bytes(tok), "big") / 10 ** H.DECIMALS
            print(f"    wallet      {self.account.address}")
            print(f"    balance     {bal/1e18:.6f} ETH")
            print(f"    HTK held    {held:.4f}")
            try:
                base = self.w3.eth.get_block("latest").get("baseFeePerGas", 0) or 0
                worst = self.gas_limit * (base * 2 + self.priority_gwei * 1e9) / 1e18
                print(f"    worst-case cost per attempt  {worst:.6f} ETH @ {base/1e9:.2f} gwei base")
            except Exception:  # noqa: BLE001
                pass
        else:
            print("    wallet      (MINER_PRIVATE_KEY not set - read-only mode)")
        print()

    def run(self):
        log(f"HTK submitter up. relay={self.relay_name} "
            f"wallet={self.account.address if self.account else 'NONE (read-only)'}")
        if self.relay_name == "PUBLIC MEMPOOL":
            log("WARNING: public mempool. mint(bytes32) is not bound to msg.sender, "
                "so bots can copy your nonce and front-run you. Use --relay flashbots.")
        threads = [
            threading.Thread(target=self.watch_ntfy, daemon=True),
            threading.Thread(target=self.watch_file, daemon=True),
            threading.Thread(target=self.retry_pending, daemon=True),
        ]
        for t in threads:
            t.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            log("shutting down")
            self.stop.set()


def main():
    p = argparse.ArgumentParser(description="Submit HTK mint() transactions from found nonces")
    p.add_argument("--once", metavar="NONCE", help="submit one nonce (0x-prefixed 32 bytes) and exit")
    p.add_argument("--dry-run", action="store_true", help="verify and simulate, never broadcast")
    p.add_argument("--status", action="store_true", help="print chain/wallet status and exit")
    p.add_argument("--relay", help="flashbots | mevblocker | beaver | public | <url>")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--wait", type=int, default=300, help="seconds to wait for a receipt")
    p.add_argument("--replay", action="store_true", help="also process existing found_nonces.jsonl lines")
    p.add_argument("--env", help="path to .env")
    args = p.parse_args()

    s = Submitter(args)
    if args.status:
        s.status()
        return
    if args.once:
        nonce = H.parse_nonce(args.once)
        if not nonce:
            raise SystemExit("--once needs a 32-byte hex nonce, e.g. 0x1234...")
        raise SystemExit(0 if s.submit(nonce, source="cli") else 1)
    s.run()


if __name__ == "__main__":
    main()
