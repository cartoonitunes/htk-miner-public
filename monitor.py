#!/usr/bin/env python3
"""Cost tracking and fleet monitoring for HTK mining.

Pulls three things together:
  * vast.ai   -- which rigs exist, what they cost, how long they have run
  * ntfy      -- heartbeats from each rig (hashrate, uptime, nonces found)
  * minted.jsonl -- what actually landed on-chain, and the gas it burned

and answers the only question that matters: what is a HTK actually costing?

It also calibrates the hashrate table used by deploy.py's ranking. Every rig
heartbeat reports real throughput for a real GPU model, so after a few hours the
$/token estimates stop being guesses. Calibration is written to
.hashrate_observed.json, which deploy.py reads on its next search.

    python3 monitor.py              # one-shot report
    python3 monitor.py --watch      # refresh every 60s
    python3 monitor.py --calibrate  # update hashrate table from heartbeats and exit
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import htk_common as H
from htk_common import log

SCRIPT_DIR = Path(__file__).resolve().parent
MINTED_LOG = SCRIPT_DIR / "minted.jsonl"
FOUND_LOG = SCRIPT_DIR / "found_nonces.jsonl"
OBSERVED = SCRIPT_DIR / ".hashrate_observed.json"
COST_LOG = SCRIPT_DIR / "cost_history.jsonl"
ALERT_STATE = SCRIPT_DIR / ".alert_state.json"

API = "https://console.vast.ai/api/v0"

RIG_RE = re.compile(r"^rig (\S+)(?:\s+\[(.+?)\])?", re.M)
HR_RE = re.compile(r"^hashrate\s+([\d.]+)\s*(TH/s|GH/s|MH/s|kH/s|H/s)", re.M)
UP_RE = re.compile(r"^uptime\s+(\S+)", re.M)
FOUND_RE = re.compile(r"^found\s+(\d+)", re.M)
SPEND_RE = re.compile(r"^spend\s+\$([\d.]+)", re.M)
GPUCOUNT_RE = re.compile(r"^(\d+)x\s+(.+)$")

UNIT = {"TH/s": 1e12, "GH/s": 1e9, "MH/s": 1e6, "kH/s": 1e3, "H/s": 1.0}


# --------------------------------------------------------------------------
def vast_instances():
    H.load_env()
    key = os.environ.get("VASTAI_API_KEY", "").strip()
    if not key:
        return None
    import requests
    try:
        r = requests.get(f"{API}/instances/",
                         headers={"Authorization": f"Bearer {key}",
                                  "Accept": "application/json"}, timeout=30)
        r.raise_for_status()
        return r.json().get("instances", [])
    except Exception as e:  # noqa: BLE001
        log(f"vast.ai query failed: {e}")
        return None


def ntfy_recent(topic: str, since: str = "12h"):
    """Recent cached messages from an ntfy topic (poll mode, non-streaming)."""
    import requests
    try:
        r = requests.get(f"{H.NTFY_BASE}/{topic}/json",
                         params={"poll": "1", "since": since}, timeout=30)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        log(f"ntfy fetch failed: {e}")
        return []
    out = []
    for line in r.text.splitlines():
        if not line.strip():
            continue
        try:
            m = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if m.get("event") == "message":
            out.append(m)
    return out


def parse_heartbeats(messages):
    """Latest heartbeat per rig."""
    rigs = {}
    for m in messages:
        body = m.get("message", "") or ""
        rm = RIG_RE.search(body)
        if not rm:
            continue
        name = rm.group(1)
        desc = rm.group(2) or ""
        hr = 0.0
        hm = HR_RE.search(body)
        if hm:
            hr = float(hm.group(1)) * UNIT.get(hm.group(2), 1.0)
        rec = {
            "rig": name, "desc": desc, "hashrate": hr,
            "ts": m.get("time", 0),
            "uptime": (UP_RE.search(body).group(1) if UP_RE.search(body) else "?"),
            "found": int(FOUND_RE.search(body).group(1)) if FOUND_RE.search(body) else 0,
            "spend": float(SPEND_RE.search(body).group(1)) if SPEND_RE.search(body) else 0.0,
            "title": m.get("title", ""),
        }
        prev = rigs.get(name)
        if not prev or rec["ts"] >= prev["ts"]:
            rigs[name] = rec
    return rigs


def calibrate(rigs):
    """Turn heartbeats into per-GPU hashrates for deploy.py's ranking."""
    obs = {}
    if OBSERVED.exists():
        try:
            obs = json.loads(OBSERVED.read_text())
        except Exception:  # noqa: BLE001
            obs = {}
    updated = []
    for r in rigs.values():
        if r["hashrate"] <= 0 or not r["desc"]:
            continue
        m = GPUCOUNT_RE.match(r["desc"].strip())
        if not m:
            continue
        n, gpu = int(m.group(1)), m.group(2).strip()
        per_gpu = r["hashrate"] / max(n, 1)
        if gpu in obs:
            # Smooth, so one bad sample cannot wreck the table.
            obs[gpu] = obs[gpu] * 0.7 + per_gpu * 0.3
        else:
            obs[gpu] = per_gpu
        updated.append((gpu, per_gpu))
    if updated:
        OBSERVED.write_text(json.dumps(obs, indent=2, sort_keys=True))
    return obs, updated


def _hours(uptime_str: str) -> float:
    """Parse H.fmt_duration output ('45m', '3.2h', '1.5d') back into hours."""
    try:
        s = (uptime_str or "").strip()
        if s.endswith("m"):
            return float(s[:-1]) / 60.0
        if s.endswith("h"):
            return float(s[:-1])
        if s.endswith("d"):
            return float(s[:-1]) * 24.0
    except ValueError:
        pass
    return 0.0


def read_jsonl(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001
            pass
    return out


def eth_price_usd():
    import requests
    for url, key in (
        ("https://api.coinbase.com/v2/prices/ETH-USD/spot", ("data", "amount")),
        ("https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd", None),
    ):
        try:
            d = requests.get(url, timeout=15).json()
            if key:
                return float(d[key[0]][key[1]])
            return float(d["ethereum"]["usd"])
        except Exception:  # noqa: BLE001
            continue
    return None


# --------------------------------------------------------------------------
def report(args):
    H.load_env()
    threshold = float(os.environ.get("COST_PER_TOKEN_ALERT", args.threshold))

    try:
        prev, max_value, _ = H.read_state_failover()
        chain_ok = True
    except Exception as e:  # noqa: BLE001
        log(f"chain read failed: {e}")
        prev, max_value, chain_ok = b"\x00" * 32, H.INITIAL_MAX_VALUE, False

    msgs = ntfy_recent(os.environ.get("NTFY_STATUS_TOPIC", H.DEFAULT_STATUS_TOPIC),
                       since=args.since)
    rigs = parse_heartbeats(msgs)
    obs, updated = calibrate(rigs)

    instances = vast_instances()
    minted = read_jsonl(MINTED_LOG)
    found = read_jsonl(FOUND_LOG)

    now = time.time()
    stale_after = args.heartbeat_timeout
    live = {k: v for k, v in rigs.items() if now - (v["ts"] or 0) < stale_after}
    dead = {k: v for k, v in rigs.items() if k not in live}

    print("\n" + "=" * 78)
    print(f"  HTK MINING STATUS      {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)

    # ---- chain
    exp = H.expected_hashes(max_value)
    print("\n  CHAIN")
    if chain_ok:
        print(f"    prev_hash        0x{prev.hex()[:32]}...")
        print(f"    difficulty       {exp:.3e} hashes/HTK  (~{H.mints_so_far(max_value):.0f} mints so far)")
        print(f"    each mint by anyone makes this 1% harder, permanently")
    else:
        print("    (unreachable)")

    # ---- fleet
    fleet_hr = sum(r["hashrate"] for r in live.values())
    print(f"\n  FLEET   {len(live)} live, {len(dead)} silent")
    if rigs:
        print(f"    {'rig':<18}{'GPU':<16}{'hashrate':>11}{'uptime':>9}{'found':>7}{'spend':>9}  state")
        print("    " + "-" * 72)
        for name, r in sorted(rigs.items()):
            state = "live" if name in live else f"SILENT {H.fmt_duration(now - (r['ts'] or now))}"
            print(f"    {name[:18]:<18}{r['desc'][:16]:<16}"
                  f"{H.fmt_hashrate(r['hashrate']):>11}{r['uptime']:>9}"
                  f"{r['found']:>7}{('$%.2f' % r['spend']):>9}  {state}")
    else:
        print("    no heartbeats seen. Are any rigs running? Is NTFY_STATUS_TOPIC right?")

    # ---- vast.ai truth
    burn = 0.0
    if instances is not None:
        running = [i for i in instances if i.get("actual_status") == "running"]
        burn = sum((i.get("dph_total") or 0) for i in running)
        print(f"\n  VAST.AI  {len(running)} running / {len(instances)} total")
        for i in instances:
            dph = i.get("dph_total") or 0
            up = i.get("duration") or 0
            cost = dph * up / 3600.0
            print(f"    {i.get('id'):>10} {str(i.get('actual_status')):<10}"
                  f"{str(i.get('gpu_name'))[:16]:<16} ${dph:.4f}/hr  "
                  f"up {H.fmt_duration(up):<7} spent ~${cost:.2f}  {i.get('label') or ''}")
    else:
        print("\n  VAST.AI  (no VASTAI_API_KEY - using rig-reported spend only)")
        burn = None

    # ---- economics
    print("\n  ECONOMICS")
    rig_spend = sum(r["spend"] for r in rigs.values())
    if burn is None:
        # No vast.ai key: infer the burn rate from what the rigs report themselves.
        burn = sum(r["spend"] / max(_hours(r["uptime"]), 1e-6) for r in live.values())
    print(f"    fleet hashrate   {H.fmt_hashrate(fleet_hr)}")
    print(f"    burn rate        ${burn:.4f}/hr  =  ${burn*24:.2f}/day")
    print(f"    spend so far     ${rig_spend:.2f}  (as reported by rigs this session)")

    if fleet_hr > 0 and chain_ok:
        eta = exp / fleet_hr
        proj = H.cost_per_token(max_value, fleet_hr, burn) if burn else 0.0
        print(f"    ETA per HTK      {H.fmt_duration(eta)}")
        if burn:
            print(f"    projected cost   ${proj:.2f} per HTK")
    else:
        proj = 0.0
        print("    ETA per HTK      n/a (no live hashrate)")

    # ---- realised
    won = [m for m in minted if m.get("success")]
    lost = [m for m in minted if not m.get("success")]
    gas_eth = sum(m.get("cost_eth", 0) for m in minted)
    px = eth_price_usd() if args.eth_price else None
    gas_usd = gas_eth * px if px else None

    print(f"\n  RESULTS")
    print(f"    nonces found     {len(found)}")
    print(f"    mints won        {len(won)}   ({len(won) * H.REWARD_TOKENS:.1f} HTK)")
    print(f"    mints lost       {len(lost)}  (reverted - burned full gas limit)")
    print(f"    gas spent        {gas_eth:.6f} ETH" + (f"  (${gas_usd:.2f})" if gas_usd else ""))
    if won:
        realised = (rig_spend + (gas_usd or 0)) / len(won)
        print(f"    realised cost    ${realised:.2f} per HTK (rental + gas / tokens won)")
    else:
        realised = None

    # ---- alert
    check = realised if realised is not None else proj
    if check and check > threshold:
        msg = (f"cost per HTK is ${check:.2f}, above your ${threshold:.0f} threshold "
               f"({'realised' if realised is not None else 'projected'})")
        print(f"\n  !! ALERT: {msg}")
        maybe_alert(msg, threshold, check, args)
    elif check:
        print(f"\n  cost per HTK ${check:.2f} is within the ${threshold:.0f} threshold")

    if updated:
        print(f"\n  calibrated hashrates from heartbeats: " +
              ", ".join(f"{g}={H.fmt_hashrate(v)}" for g, v in updated))
        print("  deploy.py will use these for ranking on its next search")

    print("\n" + "=" * 78 + "\n")

    # cost history line, for later charting
    try:
        with open(COST_LOG, "a", buffering=1) as f:
            f.write(json.dumps({
                "ts": now, "fleet_hashrate": fleet_hr, "burn_per_hr": burn,
                "spend": rig_spend, "max_value": hex(max_value),
                "tokens_won": len(won), "gas_eth": gas_eth,
                "projected_cost_per_token": proj, "realised_cost_per_token": realised,
                "live_rigs": len(live),
            }) + "\n")
    except Exception:  # noqa: BLE001
        pass


def maybe_alert(msg, threshold, value, args):
    """Push a threshold alert, but at most once every --alert-cooldown seconds."""
    now = time.time()
    state = {}
    if ALERT_STATE.exists():
        try:
            state = json.loads(ALERT_STATE.read_text())
        except Exception:  # noqa: BLE001
            pass
    if now - state.get("last", 0) < args.alert_cooldown:
        return
    H.load_env()
    topic = os.environ.get("NTFY_STATUS_TOPIC", H.DEFAULT_STATUS_TOPIC)
    H.ntfy_push(topic, msg, title="HTK cost threshold exceeded",
                tags="money_with_wings", priority=4)
    try:
        ALERT_STATE.write_text(json.dumps({"last": now, "value": value}))
    except Exception:  # noqa: BLE001
        pass


def main():
    p = argparse.ArgumentParser(description="HTK fleet monitor and cost tracker")
    p.add_argument("--watch", action="store_true", help="refresh continuously")
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--since", default="12h", help="how far back to read ntfy heartbeats")
    p.add_argument("--threshold", type=float, default=50.0, help="$/token alert threshold")
    p.add_argument("--alert-cooldown", type=int, default=3600)
    p.add_argument("--heartbeat-timeout", type=int, default=2400,
                   help="a rig silent this long counts as dead")
    p.add_argument("--eth-price", action="store_true", default=True)
    p.add_argument("--no-eth-price", dest="eth_price", action="store_false")
    p.add_argument("--calibrate", action="store_true", help="update hashrate table and exit")
    args = p.parse_args()

    if args.calibrate:
        H.load_env()
        msgs = ntfy_recent(os.environ.get("NTFY_STATUS_TOPIC", H.DEFAULT_STATUS_TOPIC), args.since)
        obs, updated = calibrate(parse_heartbeats(msgs))
        print(json.dumps(obs, indent=2, sort_keys=True))
        print(f"\n{len(updated)} rig sample(s) folded in -> {OBSERVED.name}")
        return

    if not args.watch:
        report(args)
        return
    try:
        while True:
            os.system("clear" if os.name != "nt" else "cls")
            report(args)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
