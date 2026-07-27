#!/usr/bin/env python3
"""Tests for the HTK toolchain.

The important ones are the vectors taken from real mainnet history: if
hash_value() or next_prev_hash() ever drift, these fail loudly rather than
quietly sending transactions that burn a full gas limit each.

    python3 test_htk.py           # offline vectors only
    python3 test_htk.py --live    # also check current chain state
"""
from __future__ import annotations

import sys

import htk_common as H
import monitor

PASS = FAIL = 0


def check(name, got, want):
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}\n         got  {got!r}\n         want {want!r}")


def check_true(name, cond):
    check(name, bool(cond), True)


# --------------------------------------------------------------------------
# Real mainnet vector: the winning mint in block 25,600,440
# tx 0xfaa0079a980c157a9e3229a02235ad7ec33a5d576ff3500309abc561b96f6cce
WIN_NONCE = bytes.fromhex("e8a2f59fc9013fa90f1055fb60ab50688f0fbcbf3c7454a79b2de03903a38b03")
PREV_BEFORE = bytes.fromhex("68cbde56ef31e472a3bdd141da4d58a7c48679018470b718f0ec83499ef1b14d")
PREV_AFTER = bytes.fromhex("b3f27efa4e02c1ffcead86589f95bbdd49a867795acc74c9ffc5b298d6ffd8ca")
MAX_BEFORE = int("00000000000046593edf1475d395d97b5bc12cd890512a49dd1b2a9717b4dd34", 16)


def test_keccak():
    print("\nkeccak256")
    check("empty string",
          H.keccak256(b"").hex(),
          "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470")
    check("'abc'",
          H.keccak256(b"abc").hex(),
          "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45")
    check("selector mint(bytes32)", "0x" + H.keccak256(b"mint(bytes32)")[:4].hex(), H.SEL_MINT)
    check("selector prev_hash()", "0x" + H.keccak256(b"prev_hash()")[:4].hex(), H.SEL_PREV_HASH)
    check("selector max_value()", "0x" + H.keccak256(b"max_value()")[:4].hex(), H.SEL_MAX_VALUE)
    check("topic Mint(address)", "0x" + H.keccak256(b"Mint(address)").hex(), H.TOPIC_MINT)


def test_validation():
    print("\nnonce validation (real winning mint, block 25600440)")
    v = H.hash_value(WIN_NONCE, PREV_BEFORE)
    check("hash_value matches on-chain outcome",
          f"{v:064x}",
          "00000000000011023eaea5cfaff6bbb300d4d755f1ebb69a367cc76d4cda4066")
    check_true("real winning nonce validates", H.is_valid(WIN_NONCE, PREV_BEFORE, MAX_BEFORE))
    check_true("same nonce fails against the post-mint state",
               not H.is_valid(WIN_NONCE, PREV_AFTER, H.next_max_value(MAX_BEFORE)))
    check_true("a random nonce fails", not H.is_valid(b"\x11" * 32, PREV_BEFORE, MAX_BEFORE))
    # boundary: hash == max_value is valid (contract throws only on >)
    check_true("hash exactly equal to max_value is accepted",
               H.is_valid(WIN_NONCE, PREV_BEFORE, v))
    check_true("hash one below max_value is rejected",
               not H.is_valid(WIN_NONCE, PREV_BEFORE, v - 1))


def test_chain_progression():
    print("\ndeterministic prev_hash / difficulty ratchet")
    check("next_prev_hash reproduces the real post-mint hash",
          H.next_prev_hash(PREV_BEFORE).hex(), PREV_AFTER.hex())
    check("next_max_value applies the 1% ratchet",
          H.next_max_value(1000), 990)
    check("ratchet uses integer division like Solidity",
          H.next_max_value(99), 99 - 0)  # 99//100 == 0, so no change at tiny values
    p, m = H.project_state(PREV_BEFORE, MAX_BEFORE, 1)
    check("project_state(1) == one step", (p.hex(), m),
          (PREV_AFTER.hex(), H.next_max_value(MAX_BEFORE)))
    p3, _ = H.project_state(PREV_BEFORE, MAX_BEFORE, 3)
    q = PREV_BEFORE
    for _ in range(3):
        q = H.next_prev_hash(q)
    check("project_state(3) == three steps", p3.hex(), q.hex())


def test_classification():
    print("\nnonce classification (valid / premature / stale)")
    verdict, ahead, _ = H.classify_nonce(WIN_NONCE, PREV_BEFORE, MAX_BEFORE)
    check("real winner is VALID", (verdict, ahead), (H.VALID, 0))

    verdict, _, detail = H.classify_nonce(WIN_NONCE, PREV_AFTER, H.next_max_value(MAX_BEFORE))
    check("the same nonce one mint later is STALE", verdict, H.STALE)
    check_true("stale detail explains the race", "minted first" in detail)

    # Construct a genuine premature case: an easy target makes some future state
    # satisfiable, so we can exercise the branch without doing 1e15 hashes.
    easy = 2 ** 252              # ~1 in 16 of states will accept a given nonce
    nonce = b"\x42" * 32
    base = b"\x07" * 32
    target_ahead = None
    p, m = base, easy
    for i in range(1, 60):
        p, m = H.next_prev_hash(p), H.next_max_value(m)
        if H.is_valid(nonce, p, m):
            target_ahead = i
            break
    check_true("found a future state for the premature fixture", target_ahead is not None)
    if target_ahead:
        # Not valid now, but valid target_ahead mints from now.
        if not H.is_valid(nonce, base, easy):
            verdict, ahead, detail = H.classify_nonce(nonce, base, easy, lookahead=60)
            check("classified PREMATURE", verdict, H.PREMATURE)
            check("reports the right distance ahead", ahead, target_ahead)
            check_true("premature detail says to hold", "hold it" in detail)

    # Lookahead must be respected: too small a window degrades to STALE.
    if target_ahead and target_ahead > 1 and not H.is_valid(nonce, base, easy):
        verdict, _, _ = H.classify_nonce(nonce, base, easy, lookahead=target_ahead - 1)
        check("short lookahead falls back to STALE", verdict, H.STALE)


def test_economics():
    print("\neconomics")
    check("expected_hashes(2^255) == 2", round(H.expected_hashes(2 ** 255)), 2)
    check("mints_so_far(initial) == 0", round(H.mints_so_far(H.INITIAL_MAX_VALUE)), 0)
    mv = H.INITIAL_MAX_VALUE
    for _ in range(100):
        mv = H.next_max_value(mv)
    check("mints_so_far recovers 100 mints", round(H.mints_so_far(mv)), 100)
    # 1e9 H/s against a 3.6e12-hash target = 3600s = 1h, so $1/hr -> $1/token
    target = 2 ** 256 // int(3.6e12)
    check("cost_per_token arithmetic", round(H.cost_per_token(target, 1e9, 1.0), 4), 1.0)
    check("zero hashrate is infinite cost", H.cost_per_token(target, 0, 1.0), float("inf"))


def test_parse_nonce():
    print("\nnonce parsing")
    h = "e8a2f59fc9013fa90f1055fb60ab50688f0fbcbf3c7454a79b2de03903a38b03"
    check("0x-prefixed", H.parse_nonce("0x" + h), WIN_NONCE)
    check("bare hex", H.parse_nonce(h), WIN_NONCE)
    check("whitespace tolerated", H.parse_nonce("  0x" + h + "\n"), WIN_NONCE)
    check("jsonl line", H.parse_nonce('{"ts":1,"nonce":"' + h + '","gpu":0}'), WIN_NONCE)
    check("too short rejected", H.parse_nonce("0xdeadbeef"), None)
    check("empty rejected", H.parse_nonce(""), None)
    check("non-hex rejected", H.parse_nonce("0x" + "z" * 64), None)
    check("prose rejected", H.parse_nonce("miner started up"), None)


def test_heartbeat_parsing():
    print("\nheartbeat parsing (watchdog -> monitor round trip)")
    body = "\n".join([
        "rig htk-32668906 [1x RTX 4090]",
        "uptime   3.2h",
        "hashrate 1.21 GH/s",
        "found    2 nonce(s)",
        "restarts 1",
        "spend    $0.43 @ $0.136/hr",
        "eta/token 9.9d",
        "cost/token $32.40",
    ])
    rigs = monitor.parse_heartbeats([{"event": "message", "message": body, "time": 1000}])
    check("one rig parsed", len(rigs), 1)
    r = rigs["htk-32668906"]
    check("gpu description", r["desc"], "1x RTX 4090")
    check("hashrate in H/s", r["hashrate"], 1.21e9)
    check("found count", r["found"], 2)
    check("spend", r["spend"], 0.43)
    check("uptime string", r["uptime"], "3.2h")
    check("uptime -> hours", monitor._hours("3.2h"), 3.2)
    check("minutes -> hours", round(monitor._hours("45m"), 4), 0.75)
    check("days -> hours", monitor._hours("1.5d"), 36.0)

    # newest heartbeat per rig wins
    older = body.replace("1.21 GH/s", "0.10 GH/s")
    rigs2 = monitor.parse_heartbeats([
        {"event": "message", "message": older, "time": 500},
        {"event": "message", "message": body, "time": 1000},
    ])
    check("latest heartbeat wins", rigs2["htk-32668906"]["hashrate"], 1.21e9)

    # MH/s units
    mh = body.replace("1.21 GH/s", "610 MH/s").replace("1x RTX 4090", "1x RTX 3090")
    rigs3 = monitor.parse_heartbeats([{"event": "message", "message": mh, "time": 1}])
    check("MH/s unit", rigs3["htk-32668906"]["hashrate"], 610e6)


def _offer(oid, gpu, n, dph, hashrate_per_gpu, src="prior"):
    hr = hashrate_per_gpu * n
    return {"id": oid, "gpu": gpu, "n": n, "dph": dph, "hashrate": hr, "src": src,
            "dph_per_ghs": dph / (hr / 1e9), "cost_per_token": 0.0, "eta": 0.0}


def test_speed_premium():
    print("\nspeed premium tolerance")
    import deploy
    # The rule, exactly: 20% more per hash is acceptable at 2x hashrate.
    check("1x speedup tolerates nothing", deploy.speed_tolerance(1.0, 0.20, 0.60), 1.0)
    check("slower than baseline tolerates nothing",
          deploy.speed_tolerance(0.5, 0.20, 0.60), 1.0)
    check("2x speedup tolerates exactly +20%",
          round(deploy.speed_tolerance(2.0, 0.20, 0.60), 10), 1.20)
    check("4x speedup tolerates +40% (two doublings)",
          round(deploy.speed_tolerance(4.0, 0.20, 0.60), 10), 1.40)
    check("cap bounds the premium",
          round(deploy.speed_tolerance(1024.0, 0.20, 0.60), 10), 1.60)
    check_true("1.5x sits between 1.0 and 1.2",
               1.0 < deploy.speed_tolerance(1.5, 0.20, 0.60) < 1.20)


def test_fleet_math():
    print("\nfleet arithmetic")
    import deploy
    f = deploy.Fleet([_offer(1, "RTX 3090", 2, 0.20, 1.5e9),
                      _offer(2, "RTX 3090", 2, 0.20, 1.5e9)])
    check("gpu count", f.gpus, 4)
    check("hashrate", f.hashrate, 6.0e9)
    check("dph", round(f.dph, 6), 0.40)
    check("$/GH", round(f.dollars_per_ghs, 6), round(0.40 / 6.0, 6))
    check("composition", f.composition(), "4x RTX 3090")
    check("uniform per-GPU price note", f.price_note(), "@ $0.1000/hr per GPU")

    mixed = deploy.Fleet([_offer(1, "RTX 3090", 4, 0.40, 1.5e9),
                          _offer(2, "RTX 4090", 2, 0.50, 2.5e9)])
    check("mixed composition", mixed.composition(), "4x RTX 3090 + 2x RTX 4090")
    check("mixed hashrate", mixed.hashrate, 6.0e9 + 5.0e9)

    # The invariant the whole ranking rests on: cost per token is exactly
    # proportional to $/GH, so ranking by either gives the same order.
    mv = 2 ** 256 // int(1e15)
    a = deploy.Fleet([_offer(1, "RTX 3090", 4, 0.40, 1.5e9)])
    b = deploy.Fleet([_offer(2, "RTX 4090", 4, 1.00, 2.5e9)])
    ratio_cost = a.cost_per_token(mv) / b.cost_per_token(mv)
    ratio_gh = a.dollars_per_ghs / b.dollars_per_ghs
    check("cost/token is proportional to $/GH",
          round(ratio_cost, 9), round(ratio_gh, 9))
    # 6 GH/s against 1e15 hashes = 1.667e5 s; at $0.40/hr that is ~$18.52
    check("cost_per_token value", round(a.cost_per_token(mv), 2), 18.52)


def test_fleet_selection():
    print("\nfleet selection")
    import deploy
    cheap = deploy.Fleet([_offer(1, "RTX 3090", 4, 0.40, 1.5e9)])      # 6 GH/s, $0.0667/GH
    fast_ok = deploy.Fleet([_offer(2, "RTX 3090", 8, 0.94, 1.5e9)])    # 12 GH/s, $0.0783/GH (+17%)
    fast_bad = deploy.Fleet([_offer(3, "RTX 4090", 8, 2.00, 2.5e9)])   # 20 GH/s, $0.1000/GH (+50%)

    chosen, base = deploy.select_fleet([cheap, fast_ok, fast_bad], 0.20, 0.60)
    check("baseline is the cheapest per hash", base is cheap, True)
    check("2x speed at +17% is accepted over the cheapest", chosen is fast_ok, True)

    # With no premium allowed, the cheapest must win.
    chosen, _ = deploy.select_fleet([cheap, fast_ok, fast_bad], 0.0, 0.60)
    check("zero premium selects the cheapest", chosen is cheap, True)

    # A big enough premium unlocks the fastest.
    chosen, _ = deploy.select_fleet([cheap, fast_ok, fast_bad], 0.50, 0.60)
    check("large premium selects the fastest", chosen is fast_bad, True)

    # Cap must bind: 0.50 premium would allow it, but a 0.10 cap must not.
    chosen, _ = deploy.select_fleet([cheap, fast_ok, fast_bad], 0.50, 0.10)
    check("cap overrides a generous premium", chosen is cheap, True)

    chosen, base = deploy.select_fleet([cheap], 0.20, 0.60)
    check("single fleet selects itself", (chosen is cheap, base is cheap), (True, True))
    check("empty input is handled", deploy.select_fleet([], 0.2, 0.6), (None, None))


def test_greedy_fleet():
    print("\nfleet construction")
    import deploy
    offers = [_offer(i, "RTX 3090", 2, 0.20, 1.5e9) for i in range(1, 6)]  # 5 boxes x2 GPUs
    by_value = lambda o: (o["dph_per_ghs"], -o["hashrate"])  # noqa: E731

    f = deploy.greedy_fleet(offers, 4, 4, by_value, "t")
    check("hits an exact GPU target", f.gpus, 4)
    f = deploy.greedy_fleet(offers, 8, 8, by_value, "t")
    check("fills a larger target", f.gpus, 8)
    check("never exceeds the ceiling",
          deploy.greedy_fleet(offers, 1, 3, by_value, "t").gpus, 2)
    check("returns None when the floor cannot be met",
          deploy.greedy_fleet(offers, 20, 20, by_value, "t"), None)
    check("returns None on no offers",
          deploy.greedy_fleet([], 1, 4, by_value, "t"), None)

    # A box too large to fit is skipped in favour of one that fits.
    mixed = [_offer(1, "RTX 3090", 8, 0.80, 1.5e9), _offer(2, "RTX 3090", 2, 0.20, 1.5e9)]
    check("oversized box skipped for a fitting one",
          deploy.greedy_fleet(mixed, 2, 2, by_value, "t").gpus, 2)


def test_gpu_matching():
    print("\nGPU hashrate lookup")
    import deploy
    check("exact model", deploy.gpu_hashrate("RTX 3090")[0], 1.50e9)
    check("4090 per spec", deploy.gpu_hashrate("RTX 4090")[0], 2.50e9)
    check("5090 per spec", deploy.gpu_hashrate("RTX 5090")[0], 4.00e9)
    check("3080 per spec", deploy.gpu_hashrate("RTX 3080")[0], 1.20e9)
    # Longest-match: a Ti must not be scored as the base model.
    check("3080 Ti is not scored as a 3080", deploy.gpu_hashrate("RTX 3080 Ti")[0], 1.35e9)
    check("3090 Ti is not scored as a 3090", deploy.gpu_hashrate("RTX 3090 Ti")[0], 1.70e9)
    # Vendor suffixes still resolve.
    check("A100 SXM4 resolves to A100", deploy.gpu_hashrate("A100 SXM4")[0], 2.50e9)
    check("H100 PCIE resolves to H100", deploy.gpu_hashrate("H100 PCIE")[0], 4.00e9)
    check("unknown model falls back", deploy.gpu_hashrate("RTX 2060")[1], "guess")
    check("provenance is reported", deploy.gpu_hashrate("RTX 4090")[1], "prior")


def test_competition_model():
    print("\ncompetitive drag model")
    import math
    import competition as C

    E0, h = 1.0346e15, 9.6e9
    check("no competition is the solo time",
          C.competitive_time_to_token(E0, h, 0), E0 / h)
    check("zero hashrate never finishes",
          C.competitive_time_to_token(E0, 0, 1e9), float("inf"))

    # The ratchet subtracts 1% of the network hashrate from yours.
    got = C.competitive_time_to_token(E0, h, 5.03e9)
    check("closed form is E0/(h - 0.01*net)", round(got, 6), round(E0 / (h - 0.01 * 5.03e9), 6))
    check_true("competition makes it slower, not faster", got > E0 / h)
    check_true("but only slightly (<1% at observed levels)", got / (E0 / h) < 1.01)

    # Divergence: once the field is 100x you, difficulty outruns your search.
    check("100x network diverges", C.competitive_time_to_token(E0, h, 100 * h), float("inf"))
    check("beyond 100x also diverges", C.competitive_time_to_token(E0, h, 200 * h), float("inf"))
    check_true("just under the threshold is finite and large",
               C.competitive_time_to_token(E0, h, 99 * h) < float("inf"))

    # Cross-check the closed form against numerical integration of the survival
    # function P(not found by t) = (1 + a t)^-b. This is the claim the whole
    # "competition barely matters" conclusion rests on.
    def numeric_mean(net, steps=400_000, horizon=60):
        a = 0.01 * net / E0
        b = h / (0.01 * net)
        T = horizon * E0 / h
        dt = T / steps
        tot = 0.0
        for i in range(steps):
            t = (i + 0.5) * dt
            tot += math.exp(-b * math.log1p(a * t)) * dt
        return tot

    for net in (1.17e9, 5.03e9, 23.59e9):
        num = numeric_mean(net)
        closed = C.competitive_time_to_token(E0, h, net)
        check_true(f"numeric integration agrees at {net/1e9:.2f} GH/s "
                   f"({num:.4e} vs {closed:.4e})",
                   abs(num - closed) / closed < 0.01)

    # Difficulty walk-back: N mints ago the target was 0.99^-N times today's.
    mv = 1e60
    check("difficulty_at(0) undoes one ratchet step",
          round(C.difficulty_at(mv, 0) * 0.99, 6), round(mv, 6))
    check_true("older mints faced an easier (larger) target",
               C.difficulty_at(mv, 100) > C.difficulty_at(mv, 10) > mv)


def test_no_secrets_on_instance():
    print("\nsecret isolation (rented box must never see the wallet)")
    import argparse
    import os
    import deploy

    # Plant recognisable secrets in the environment, then confirm none of them
    # reach the vast.ai bootstrap script.
    canaries = {
        "MINER_PRIVATE_KEY": "0x" + "de" * 32,
        "MAINNET_RPC_URL": "https://eth-mainnet.g.alchemy.com/v2/CANARYKEY123",
        "VASTAI_API_KEY": "CANARY-VAST-KEY-456",
        "ETHERSCAN_API_KEY": "CANARY-ETHERSCAN-789",
    }
    saved = {k: os.environ.get(k) for k in canaries}
    os.environ.update(canaries)
    try:
        # Every delivery mode must be secret-free, not just inline.
        modes = [
            argparse.Namespace(mode="inline", workdir="/opt/htk", heartbeat=900,
                               image=None, code_url="", _dph=0.1, _gpu_desc="1x RTX 3080"),
            argparse.Namespace(mode="image", workdir="/opt/htk", heartbeat=900,
                               image="you/htk-miner", code_url="", _dph=0.1, _gpu_desc="1x RTX 3080"),
            argparse.Namespace(mode="fetch", workdir="/opt/htk", heartbeat=900,
                               image=None, code_url="https://ex.com/htk.tgz",
                               _dph=0.1, _gpu_desc="1x RTX 3080"),
        ]
        scripts = {m.mode: deploy.build_onstart(m, "htk-test") for m in modes}
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    for mode, script in scripts.items():
        for name, value in canaries.items():
            check_true(f"[{mode}] {name} absent from onstart", value not in script)
        check_true(f"[{mode}] no 'PRIVATE_KEY' string in onstart", "PRIVATE_KEY" not in script)
        # The miner does legitimately need the ntfy topics.
        check_true(f"[{mode}] ntfy topic IS shipped (the miner needs it)", "NTFY_TOPIC" in script)


def test_onstart_size_and_delivery():
    print("\nonstart size vs vast.ai limits (image/args/label)")
    import argparse
    import deploy

    def mk(mode, image=None, code_url=""):
        return argparse.Namespace(mode=mode, image=image, code_url=code_url,
                                  workdir="/opt/htk", heartbeat=900,
                                  _dph=0.0556, _gpu_desc="8x RTX 3080")

    # This is the exact failure that was hit: inline embedding all four files
    # blows past the 16 KB args limit even after gzip.
    inline = deploy.build_onstart(mk("inline"), "htk-33362248")
    check_true("inline onstart still exceeds 16 KB (why it must be blocked)",
               len(inline) > deploy.VAST_MAX_ONSTART)
    probs = deploy.check_vast_limits(deploy.DEFAULT_IMAGE_INLINE, inline, "htk-33362248")
    check_true("guard flags the oversized inline onstart", len(probs) == 1)
    check_true("guard names the args/onstart field", "onstart/args" in probs[0])

    # image and fetch keep the script tiny and pass the guard.
    for mode, kw in [("image", {"image": "you/htk-miner"}),
                     ("fetch", {"code_url": "https://ex.com/htk-miner.tar.gz"}),
                     ("fetch", {"code_url": "https://github.com/x/htk.git"})]:
        s = deploy.build_onstart(mk(mode, **kw), "htk-33362248")
        img = kw.get("image") or deploy.DEFAULT_IMAGE_INLINE
        check_true(f"[{mode}] onstart well under 16 KB ({len(s)} B)",
                   len(s) < 2048)
        check("[%s] passes the preflight guard" % mode,
              deploy.check_vast_limits(img, s, "htk-33362248"), [])

    # The guard also catches the other two documented limits.
    check_true("guard flags an over-long image",
               any("image is" in p for p in
                   deploy.check_vast_limits("x" * 1100, "ok", "lbl")))
    check_true("guard flags an over-long label",
               any("label is" in p for p in
                   deploy.check_vast_limits("img", "ok", "L" * 300)))

    # The inline blob must reconstruct the files byte-for-byte.
    import base64
    import gzip
    import io
    import tarfile
    blob = gzip.decompress(base64.b64decode(deploy.miner_tarball_b64()))
    tf = tarfile.open(fileobj=io.BytesIO(blob))
    check("inline blob carries exactly the miner files",
          sorted(tf.getnames()), sorted(deploy.FILES_TO_SHIP))
    from pathlib import Path
    root = Path(deploy.__file__).resolve().parent
    ok = all(tf.extractfile(n).read() == (root / n).read_bytes() for n in tf.getnames())
    check_true("inline blob round-trips to identical source", ok)


def test_repo_has_no_secrets():
    print("\nrepo hygiene (no committed credentials)")
    import re
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parent
    tracked = subprocess.run(["git", "-C", str(root), "ls-files"],
                             capture_output=True, text=True).stdout.split()
    check_true(".env is not tracked by git", ".env" not in tracked)
    check_true(".gitignore covers .env",
               ".env" in (root / ".gitignore").read_text().split())

    # Credential shapes that must never appear in tracked files.
    patterns = [
        (r"[A-Z0-9]{34}", "etherscan-style 34-char key"),
        (r"alchemy\.com/v2/(?!YOUR_KEY_HERE)[A-Za-z0-9_-]{16,}", "live alchemy url"),
    ]
    offenders = []
    for rel in tracked:
        p = root / rel
        if not p.exists() or p.suffix not in (".py", ".md", ".txt", ".example", ""):
            continue
        text = p.read_text(errors="ignore")
        for pat, label in patterns:
            for m in re.finditer(pat, text):
                # Canaries in this test file are deliberate.
                if rel == "test_htk.py" or "CANARY" in m.group(0):
                    continue
                offenders.append(f"{rel}: {label} -> {m.group(0)[:24]}")
    check("no credential-shaped strings in tracked files", offenders, [])


def test_formatting():
    print("\nformatting")
    check("GH/s", H.fmt_hashrate(1.2e9), "1.20 GH/s")
    check("MH/s", H.fmt_hashrate(6e8), "600.00 MH/s")
    check("hours", H.fmt_duration(3600 * 3.2), "3.2h")
    check("days", H.fmt_duration(86400 * 1.5), "1.5d")
    check("minutes", H.fmt_duration(600), "10m")


def test_live():
    print("\nlive chain (network)")
    prev, mx, idx = H.read_state_failover()
    check_true("prev_hash is 32 bytes", len(prev) == 32)
    check_true("max_value below the 2016 starting value", mx < H.INITIAL_MAX_VALUE)
    check_true("difficulty is plausible (2^40..2^80 hashes)",
               2 ** 40 < H.expected_hashes(mx) < 2 ** 80)
    print(f"       prev_hash 0x{prev.hex()}")
    print(f"       expected hashes/HTK {H.expected_hashes(mx):.3e} via {H.READ_RPCS[idx]}")


def main():
    print("HTK toolchain tests")
    test_keccak()
    test_validation()
    test_chain_progression()
    test_classification()
    test_economics()
    test_parse_nonce()
    test_heartbeat_parsing()
    test_speed_premium()
    test_fleet_math()
    test_fleet_selection()
    test_greedy_fleet()
    test_gpu_matching()
    test_competition_model()
    test_no_secrets_on_instance()
    test_onstart_size_and_delivery()
    test_repo_has_no_secrets()
    test_formatting()
    if "--live" in sys.argv:
        test_live()
    print(f"\n{PASS} passed, {FAIL} failed\n")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
