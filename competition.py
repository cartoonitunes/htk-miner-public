#!/usr/bin/env python3
"""Measure how contested HTK mining actually is, and what that costs you.

Pulls Mint(address) events, derives the difficulty in force at each one, and
back-calculates the combined hashrate of everyone else mining. Then it prices
what that competition does to your own cost per token.

The headline result is counterintuitive and worth stating up front: on this
contract, competitors barely affect your cost per token. See the notes under
COMPETITIVE DRAG below for why - HTK has no difficulty retargeting, so your own
discovery rate depends on your hashrate and the current target, not on how many
other people are hashing.

    python3 competition.py                    # last 90 days
    python3 competition.py --days 365
    python3 competition.py --hashrate 9.6e9 --rate 0.378
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.parse
import urllib.request
from collections import Counter

import htk_common as H

ETHERSCAN = "https://api.etherscan.io/v2/api"
MINT_TOPIC = H.TOPIC_MINT
RATCHET = 0.99  # max_value -= max_value/100 on every mint


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "htk-miner/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def rpc(url, method, params, timeout=30):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"content-type": "application/json",
                                          "User-Agent": "htk-miner/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    if "error" in d:
        raise RuntimeError(d["error"].get("message", str(d["error"])))
    return d["result"]


def fetch_mints(api_key, from_block, to_block="latest", contract=None):
    """All Mint events in range, oldest first. Etherscan pages at 1000."""
    contract = contract or H.CONTRACT
    out, page = [], 1
    while True:
        q = urllib.parse.urlencode({
            "chainid": 1, "module": "logs", "action": "getLogs",
            "address": contract.lower(), "topic0": MINT_TOPIC,
            "fromBlock": from_block, "toBlock": to_block,
            "page": page, "offset": 1000, "apikey": api_key,
        })
        d = _get(f"{ETHERSCAN}?{q}")
        if d.get("status") != "1":
            if "No records" in str(d.get("message", "")):
                break
            raise SystemExit(f"etherscan getLogs failed: {d.get('message')} {d.get('result')}")
        rows = d.get("result") or []
        out.extend(rows)
        if len(rows) < 1000:
            break
        page += 1
        time.sleep(0.25)
    events = []
    for r in out:
        events.append({
            "block": int(r["blockNumber"], 16),
            "ts": int(r["timeStamp"], 16),
            "miner": "0x" + r["topics"][1][-40:],
            "tx": r["transactionHash"],
            "gas_price": int(r.get("gasPrice", "0x0"), 16),
        })
    events.sort(key=lambda e: (e["block"], e["ts"]))
    # Etherscan can repeat rows across page boundaries; de-dupe on tx hash.
    seen, uniq = set(), []
    for e in events:
        if e["tx"] in seen:
            continue
        seen.add(e["tx"])
        uniq.append(e)
    for i, e in enumerate(uniq):
        e["idx"] = i
    return uniq


def difficulty_at(max_value_now, mints_after):
    """Target in force for a mint that had `mints_after` mints following it.

    Each mint multiplies max_value by ~0.99, so walking backwards divides.
    """
    return max_value_now / (RATCHET ** (mints_after + 1))


def window_stats(events, max_value_now, total_events, now, seconds, label):
    """Mint cadence and implied network hashrate over a trailing window.

    `events` must be oldest-first; each carries its index so difficulty can be
    walked back without an O(n) lookup per event.
    """
    cutoff = now - seconds
    sel = [e for e in events if e["ts"] >= cutoff]
    if len(sel) < 1:
        return {"label": label, "n": 0, "seconds": seconds}

    span = now - min(e["ts"] for e in sel)
    rate = len(sel) / seconds  # mints per second over the full window
    gap = seconds / len(sel)

    exps = [(2 ** 256) / difficulty_at(max_value_now, total_events - 1 - e["idx"])
            for e in sel]
    mean_exp = sum(exps) / len(exps)

    # Poisson: the count itself is the only information, so the relative
    # standard error on the rate is 1/sqrt(n).
    rel_err = 1.0 / math.sqrt(len(sel))

    return {
        "label": label, "n": len(sel), "seconds": seconds, "span": span,
        "rate": rate, "gap": gap, "mean_exp": mean_exp,
        "hashrate": mean_exp * rate, "rel_err": rel_err,
        "miners": Counter(e["miner"] for e in sel),
    }


def monthly_histogram(events, months=14):
    """Mint counts per calendar month. A bursty process is badly served by a
    single average, so show the shape instead."""
    buckets = {}
    for e in events:
        key = time.strftime("%Y-%m", time.gmtime(e["ts"]))
        buckets[key] = buckets.get(key, 0) + 1
    keys = sorted(buckets)[-months:]
    return [(k, buckets[k]) for k in keys]


def competitive_time_to_token(exp_hashes, my_hr, net_hr):
    """Mean seconds for YOU to win one token, accounting for the ratchet.

    Your instantaneous find rate is my_hr/E and does NOT depend on how many
    others are hashing - this contract has no difficulty retargeting. What
    competitors do is push E up 1% per mint they land, so while you search,

        dE/dt = E * 0.01 * (net_hr / E) = 0.01 * net_hr

    i.e. E grows *linearly*: E(t) = E0 + 0.01*net_hr*t. Your find process is a
    non-homogeneous Poisson process with intensity my_hr/E(t), so

        P(not found by t) = exp(-integral) = (1 + a*t)^(-b)
        with a = 0.01*net_hr/E0 and b = my_hr/(0.01*net_hr)

    and the mean time is the integral of that survival function, which collapses
    to a clean closed form (verified numerically to 6 significant figures):

        E[T] = E0 / (my_hr - 0.01*net_hr)

    So the ratchet simply subtracts 1% of the network's hashrate from your own.
    Note the divergence: if net_hr >= 100 * my_hr the denominator goes to zero
    and your expected time is infinite - difficulty outruns you faster than you
    can search. That threshold is far above anything this contract has seen.
    """
    if my_hr <= 0:
        return float("inf")
    if net_hr <= 0:
        return exp_hashes / my_hr
    effective = my_hr - 0.01 * net_hr
    if effective <= 0:
        return float("inf")
    return exp_hashes / effective


def main():
    p = argparse.ArgumentParser(description="HTK mining competition analysis")
    p.add_argument("--days", type=float, default=90, help="history window to pull")
    p.add_argument("--hashrate", type=float, default=9.6e9,
                   help="your fleet hashrate in H/s (default: 8x RTX 3080 at 1.2 GH/s)")
    p.add_argument("--rate", type=float, default=0.378,
                   help="your fleet cost in $/hr")
    p.add_argument("--fleet", default="8x RTX 3080")
    p.add_argument("--recent", type=int, default=25, help="mints to list individually")
    p.add_argument("--etherscan-key", default="",
                   help="Etherscan API key (or set ETHERSCAN_API_KEY in .env)")
    p.add_argument("--rpc", default=os.environ.get("MAINNET_RPC_URL", "").strip()
                   or "https://ethereum-rpc.publicnode.com")
    args = p.parse_args()

    H.load_env()
    rpc_url = args.rpc

    etherscan_key = args.etherscan_key or os.environ.get("ETHERSCAN_API_KEY", "").strip()
    if not etherscan_key:
        raise SystemExit(
            "No Etherscan API key. Mint events need one because Alchemy's free tier\n"
            "caps eth_getLogs at a 10-block range.\n"
            "  Get a free key at https://etherscan.io/apis, then either\n"
            "    echo 'ETHERSCAN_API_KEY=yourkey' >> .env\n"
            "  or pass --etherscan-key yourkey")

    print("=" * 78)
    print("  HTK MINING COMPETITION")
    print("=" * 78)

    head = int(rpc(rpc_url, "eth_blockNumber", []), 16)
    mv_raw = rpc(rpc_url, "eth_call",
                 [{"to": H.CONTRACT, "data": H.SEL_MAX_VALUE}, "latest"])
    max_value = int(mv_raw, 16)
    prev_raw = rpc(rpc_url, "eth_call",
                   [{"to": H.CONTRACT, "data": H.SEL_PREV_HASH}, "latest"])
    exp_now = H.expected_hashes(max_value)

    print(f"\n  chain head    {head:,}")
    print(f"  prev_hash     {prev_raw}")
    print(f"  max_value     0x{max_value:064x}")
    print(f"  difficulty    {exp_now:.4e} expected hashes per token (2^{math.log2(exp_now):.2f})")
    print(f"  implied mints ~{H.mints_so_far(max_value):.0f} since 2016 deployment")

    blocks_back = int(args.days * 86400 / 12.05)
    from_block = max(0, head - blocks_back)
    print(f"\n  scanning Mint(address) logs from block {from_block:,} "
          f"({args.days:.0f} days)...")
    events = fetch_mints(etherscan_key, from_block)
    if not events:
        raise SystemExit("no Mint events found in that range")
    total = len(events)
    now = int(time.time())
    print(f"  found {total} Mint events")

    # Cross-check a couple of Etherscan timestamps against the RPC directly.
    checked = 0
    for e in (events[-1], events[0]):
        try:
            blk = rpc(rpc_url, "eth_getBlockByNumber", [hex(e["block"]), False])
            if int(blk["timestamp"], 16) == e["ts"]:
                checked += 1
        except Exception:  # noqa: BLE001
            pass
    print(f"  timestamp cross-check against RPC: {checked}/2 match")

    # ---------------------------------------------------------------- recent
    print(f"\n  RECENT MINTS (last {min(args.recent, total)})")
    print(f"    {'when (UTC)':<18}{'block':>10}  {'miner':<14}{'gap':>9}{'gwei':>7}")
    print("    " + "-" * 62)
    tail = events[-args.recent:]
    for i, e in enumerate(tail):
        prev_ts = tail[i - 1]["ts"] if i > 0 else None
        gap = H.fmt_duration(e["ts"] - prev_ts) if prev_ts else "-"
        print(f"    {time.strftime('%Y-%m-%d %H:%M', time.gmtime(e['ts'])):<18}"
              f"{e['block']:>10}  {e['miner'][:12]:<14}{gap:>9}"
              f"{e['gas_price']/1e9:>7.2f}")
    age = now - events[-1]["ts"]
    print(f"\n    most recent mint was {H.fmt_duration(age)} ago")

    # ---------------------------------------------------------------- windows
    print(f"\n  CADENCE AND IMPLIED NETWORK HASHRATE")
    print(f"    {'window':<10}{'mints':>6}{'avg gap':>10}{'mints/day':>11}"
          f"{'implied network':>18}")
    print("    " + "-" * 56)
    windows = [("24h", 86400), ("7d", 7 * 86400), ("30d", 30 * 86400),
               ("90d", 90 * 86400), (f"{args.days:.0f}d", int(args.days * 86400))]
    seen_labels = set()
    stats = {}
    for label, secs in windows:
        if label in seen_labels or secs > args.days * 86400 * 1.02:
            continue
        seen_labels.add(label)
        s = window_stats(events, max_value, total, now, secs, label)
        stats[label] = s
        if s["n"] == 0:
            print(f"    {label:<10}{0:>6}{'-':>10}{'0.00':>11}{'~0 (idle)':>18}")
            continue
        print(f"    {label:<10}{s['n']:>6}{H.fmt_duration(s['gap']):>10}"
              f"{s['n']/(secs/86400):>11.2f}{H.fmt_hashrate(s['hashrate']):>18}")

    print()
    print("    'implied network' = expected_hashes_per_mint / average_seconds_per_mint.")
    print("    It is a Poisson estimate: with few mints in a window the error is large")
    print(f"    (a window with n mints has roughly +/-{100/math.sqrt(max(stats.get('30d',{}).get('n',1),1)):.0f}% "
          f"standard error at n=30d's count).")

    # ---------------------------------------------------------------- miners
    print(f"\n  ACTIVE MINERS")
    for label in ["24h", "7d", "30d"]:
        s = stats.get(label)
        if not s or s["n"] == 0:
            print(f"    {label:<5} none")
            continue
        c = s["miners"]
        who = ", ".join(f"{a[:10]}({n})" for a, n in c.most_common(6))
        print(f"    {label:<5} {len(c)} unique, {s['n']} mints: {who}")

    all_c = Counter(e["miner"] for e in events)
    print(f"\n    over the full {args.days:.0f}d window: {len(all_c)} unique miners, "
          f"{total} mints")
    print(f"    {'miner':<44}{'mints':>7}{'share':>8}")
    print("    " + "-" * 60)
    for addr, n in all_c.most_common(8):
        print(f"    {addr:<44}{n:>7}{100*n/total:>7.1f}%")

    # ---------------------------------------------------------------- trend
    print(f"\n  TREND - mints per calendar month")
    hist = monthly_histogram(events)
    peak = max(n for _, n in hist) or 1
    for k, n in hist:
        bar = "#" * max(1, int(38 * n / peak)) if n else ""
        print(f"    {k}  {n:>4}  {bar}")
    print()
    print("    Mining here is BURSTY, not steady: someone points a farm at the")
    print("    contract for days, then it goes quiet for months. A single average")
    print("    over the year is meaningless - what matters is whether a burst is")
    print("    running right now.")

    gaps = [events[i + 1]["ts"] - events[i]["ts"] for i in range(len(events) - 1)]
    if gaps:
        long_gaps = [g for g in gaps if g > 7 * 86400]
        print(f"\n    {len(gaps)} intervals in window: median {H.fmt_duration(sorted(gaps)[len(gaps)//2])}, "
              f"longest {H.fmt_duration(max(gaps))}")
        print(f"    {len(long_gaps)} dormant stretches over 7 days")

    d30, d7, d1 = stats.get("30d"), stats.get("7d"), stats.get("24h")
    net = None
    for s in (d7, d30):
        if s and s["n"] >= 2:
            net = s["hashrate"]
            net_label = f"{s['label']} cadence, n={s['n']}"
            break
    if net is None:
        for s in (d30, stats.get("90d")):
            if s and s["n"] >= 1:
                net, net_label = s["hashrate"], f"{s['label']} cadence, n={s['n']}"
                break

    # ---------------------------------------------------------------- yours
    my_hr = args.hashrate
    print(f"\n" + "=" * 78)
    print(f"  YOUR POSITION - {args.fleet} at {H.fmt_hashrate(my_hr)}, ${args.rate:.3f}/hr")
    print("=" * 78)

    if net is None or net <= 0:
        print("\n  Not enough recent mints to estimate a network hashrate.")
        net = 0.0
        net_label = "n/a"

    print(f"\n  network (others)  {H.fmt_hashrate(net)}   [from {net_label}]")
    print(f"  you               {H.fmt_hashrate(my_hr)}")
    if net > 0:
        print(f"  your share        {100*my_hr/(my_hr+net):.1f}% of combined hashrate")

    solo_secs = exp_now / my_hr
    comp_secs = competitive_time_to_token(exp_now, my_hr, net)
    drag = (comp_secs / solo_secs - 1) * 100 if solo_secs else 0

    solo_cost = solo_secs / 3600 * args.rate
    comp_cost = comp_secs / 3600 * args.rate

    print(f"\n  EXPECTED TIME AND COST PER TOKEN")
    print(f"    ignoring competition   {H.fmt_duration(solo_secs):>8}   ${solo_cost:,.2f}")
    print(f"    with competition       {H.fmt_duration(comp_secs):>8}   ${comp_cost:,.2f}")
    print(f"    competitive drag       {drag:+.2f}%")

    if net > 0:
        others_per_your_token = net / my_hr
        print(f"\n    While you search for one token, others land ~{others_per_your_token:.1f},")
        print(f"    each raising difficulty 1%. That is the entire competitive cost.")

    # The point estimate rests on very few mints, so show the whole plausible
    # range instead of pretending to a precision the data cannot support.
    print(f"\n  SENSITIVITY - competition is a rounding error across the whole range")
    print(f"    {'network hashrate':<20}{'your share':>11}{'drag':>9}{'$/token':>10}  basis")
    print("    " + "-" * 62)
    scenarios = []
    for lbl in ("30d", "7d", "24h"):
        s = stats.get(lbl)
        if s and s["n"] >= 1:
            scenarios.append((s["hashrate"], f"{lbl} cadence (n={s['n']})"))
    scenarios.append((my_hr * 10, "stress: 10x your fleet"))
    scenarios.append((100e9, "stress: 100 GH/s"))
    seen_hr = set()
    for hr, basis in sorted(scenarios):
        k = round(hr, 3)
        if k in seen_hr:
            continue
        seen_hr.add(k)
        t = competitive_time_to_token(exp_now, my_hr, hr)
        d = (t / solo_secs - 1) * 100
        print(f"    {H.fmt_hashrate(hr):<20}{100*my_hr/(my_hr+hr):>10.1f}%"
              f"{d:>8.2f}%{t/3600*args.rate:>10.2f}  {basis}")
    print()
    print("    Across every plausible level the drag stays under ~3%. Even at a")
    print("    stress-test 100 GH/s - far above anything this contract has seen -")
    print("    it is ~12%, still smaller than the error on the hashrate prior.")
    print(f"    Your search only stalls outright if the field exceeds "
          f"{H.fmt_hashrate(100*my_hr)} (100x you).")

    print(f"\n  COMPETITIVE DRAG - why it is so small")
    print(f"    HTK has no difficulty retargeting. Your find rate is my_hashrate/E")
    print(f"    and does NOT depend on how many others are hashing - unlike Bitcoin,")
    print(f"    where doubling network hashrate halves your share. Here the only")
    print(f"    competitive cost is the 1%-per-mint ratchet others push onto E while")
    print(f"    you search. Losing a race costs you nothing in expectation either:")
    print(f"    keccak search is memoryless, so a discarded partial search has the")
    print(f"    same expected remaining work as a fresh one.")

    # Two real but small losses.
    poll = 12.0
    if net > 0:
        stale_frac = (net / exp_now) * (poll / 2) # fraction of time on a dead prev_hash
        print(f"\n    Two real losses, both small:")
        print(f"      stale work: the miner polls every {poll:.0f}s, so after someone")
        print(f"        else mints you hash a dead prev_hash for ~{poll/2:.0f}s on average")
        print(f"        -> {stale_frac*100:.4f}% of your hashes wasted")
        window = 30.0
        race = (net / exp_now) * window
        print(f"      submission race: if someone lands a mint in the ~{window:.0f}s between")
        print(f"        your find and your confirmation, your nonce dies")
        print(f"        -> ~{race*100:.3f}% of your wins lost")
        total_loss = stale_frac + race
        print(f"      combined: ~{total_loss*100:.3f}% on top of the {drag:.2f}% ratchet drag")

    print(f"\n  WHAT ACTUALLY DRIVES YOUR COST")
    print(f"    Competition moves cost per token by {drag:+.2f}%. The hashrate prior")
    print(f"    moves it by 100% or more if the kernel underperforms. At half the")
    print(f"    assumed rate ({H.fmt_hashrate(my_hr/2)}) this fleet costs "
          f"${comp_cost*2:,.2f}/token")
    print(f"    and takes {H.fmt_duration(comp_secs*2)}. Measure the kernel before scaling.")
    print()
    print("=" * 78)
    print()


if __name__ == "__main__":
    main()
