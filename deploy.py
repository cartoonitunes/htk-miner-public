#!/usr/bin/env python3
"""Provision HTK mining rigs on vast.ai.

Three ways to get the miner code onto a box (--mode). vast.ai caps the onstart
script at 16 KB and the four miner files are ~43 KB, so embedding them all is
not possible - the delivery mode is what keeps the script small:

  --mode image  (default)  The code is baked into a Docker image you built from
                           the Dockerfile and pushed. Onstart just runs it.
                           Robust at any code size; needs a Docker Hub account.
  --mode fetch             Onstart downloads a tarball from --code-url (any
                           public host - the miner code holds no secrets).
                           Small onstart, no Docker Hub.
  --mode inline            The code is gzip'd + base64'd into the onstart. Zero
                           external deps, but only if the payload fits in 16 KB;
                           provisioning refuses it with guidance otherwise.

Offers are ranked by estimated cost per HTK, not by $/hr. A cheap slow card can
easily be worse value than a dearer fast one, and $/hr alone hides that.

Renting costs real money, so nothing is created without --yes.

Instance selection scores fleets on two axes: cost efficiency ($/GH, which is
exactly proportional to $/token) and total hashrate (which sets the ETA). A
"speed premium" says how much extra per hash is worth paying for more speed:
0.20 means "+20% per hash is acceptable for 2x the hashrate".

    python3 deploy.py plan                    # pick the best fleet, spend nothing
    python3 deploy.py plan --gpus 16          # size it explicitly
    python3 deploy.py auto --gpus 8 --yes     # pick and provision
    python3 deploy.py search                  # rank individual offers
    python3 deploy.py list
    python3 deploy.py destroy <id> [<id>...]
    python3 deploy.py destroy --all --yes
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import htk_common as H
from htk_common import log

SCRIPT_DIR = Path(__file__).resolve().parent
API = "https://console.vast.ai/api/v0"

# Keccak256 throughput per GPU, in H/s. Matching is longest-key-first, so
# "RTX 3080 Ti" wins over "RTX 3080" and "A100 SXM4" over "A100".
#
# These are PRIORS, not measurements. They are estimated figures, and
# they assume a well-optimised keccak kernel. The upstream kernel here is a
# straightforward keccak-f1600 with the full 25-word state in local memory, so
# real throughput may land well below these. That matters: every $/token figure
# scales inversely with hashrate, so if the kernel does half these numbers, the
# true cost per token is double what the planner shows.
#
# monitor.py overwrites each entry from real heartbeats as rigs report in
# (.hashrate_observed.json), and anything measured is used in preference to
# these. Run one box first and let it calibrate before committing to a fleet.
HASHRATE_ESTIMATES = {
    "RTX 5090": 4.00e9,
    "H100": 4.00e9,
    "RTX 4090": 2.50e9,
    "A100": 2.50e9,
    "RTX 4080": 2.00e9,
    "RTX 3090": 1.50e9,
    "RTX 3080": 1.20e9,
    # Neighbours of the above, interpolated from the same scale.
    "RTX 3090 Ti": 1.70e9,
    "RTX 3080 Ti": 1.35e9,
    "RTX 4070": 1.10e9,
    "L40S": 2.20e9,
    "RTX A5000": 1.10e9,
    "RTX A6000": 1.40e9,
}
DEFAULT_HASHRATE = 1.00e9

FILES_TO_SHIP = ["keccak_miner.cu", "htk_cuda_miner.py", "htk_common.py", "miner_watchdog.py"]
DEFAULT_IMAGE_INLINE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"
MINER_PIP = "'cupy-cuda12x>=13.0' 'numpy>=1.24' 'requests>=2.31'"

# vast.ai create-instance field limits (from error 400/3471). The onstart
# script is sent as `args`, so it is the 16 KB limit that bites.
VAST_MAX_IMAGE = 1024
VAST_MAX_ONSTART = 16384
VAST_MAX_LABEL = 256


def observed_hashrates():
    path = SCRIPT_DIR / ".hashrate_observed.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            pass
    return {}


def gpu_hashrate(gpu_name: str) -> tuple[float, str]:
    """(hashes_per_second, provenance) for one GPU of this model.

    Measured values always beat priors. Model matching is longest-key-first so
    that "RTX 3080 Ti" does not get scored as a plain "RTX 3080".
    """
    obs = observed_hashrates()
    name = (gpu_name or "").strip()
    if name in obs:
        return float(obs[name]), "measured"
    low = name.lower()
    for k in sorted(obs, key=len, reverse=True):
        if k.lower() in low:
            return float(obs[k]), "measured~"
    if name in HASHRATE_ESTIMATES:
        return HASHRATE_ESTIMATES[name], "prior"
    for k in sorted(HASHRATE_ESTIMATES, key=len, reverse=True):
        if k.lower() in low:
            return HASHRATE_ESTIMATES[k], "prior~"
    return DEFAULT_HASHRATE, "guess"


# --------------------------------------------------------------------------
# Fleet planning
#
# The two axes that matter are cost efficiency and wall time:
#
#   $/GH   = total $/hr divided by total GH/s. Lower is better.
#   GH/s   = total fleet hashrate. Higher is better (shorter time to a token).
#
# These are not independent of the headline number: expected cost per token is
#   (expected_hashes / 3600) * ($/hr / hashrate)
# so cost-per-token is exactly proportional to $/GH. Minimising $/GH and
# minimising $/token are the same objective; hashrate alone sets the ETA.
#
# So the trade is: pay a higher $/GH to buy a shorter ETA. The speed premium
# says how much. "Up to 20% more per hash for 2x the hashrate" generalises to
#
#   tolerance(speedup) = 1 + premium * log2(speedup)      capped at premium_cap
#
# which is exactly 1.20 at a 2x speedup, and keeps behaving sensibly either
# side of it instead of being a cliff at exactly 2.0.
# --------------------------------------------------------------------------
class Fleet:
    """A set of whole vast.ai offers rented together."""

    def __init__(self, offers, label=""):
        self.offers = list(offers)
        self.label = label

    @property
    def gpus(self):
        return sum(o["n"] for o in self.offers)

    @property
    def hashrate(self):
        return sum(o["hashrate"] for o in self.offers)

    @property
    def dph(self):
        return sum(o["dph"] for o in self.offers)

    @property
    def dollars_per_ghs(self):
        """$/hr per GH/s. The cost-efficiency number, proportional to $/token."""
        hr = self.hashrate
        return (self.dph / (hr / 1e9)) if hr > 0 else float("inf")

    def cost_per_token(self, max_value):
        return H.cost_per_token(max_value, self.hashrate, self.dph)

    def eta_seconds(self, max_value):
        return H.expected_hashes(max_value) / self.hashrate if self.hashrate else float("inf")

    @property
    def provenance(self):
        """Worst provenance across the fleet - how much to trust the numbers."""
        order = ["measured", "measured~", "prior", "prior~", "guess"]
        worst = 0
        for o in self.offers:
            worst = max(worst, order.index(o["src"]) if o["src"] in order else len(order) - 1)
        return order[worst]

    def composition(self):
        """'8x RTX 3090' or '4x RTX 3090 + 2x RTX 4090'."""
        counts = {}
        for o in self.offers:
            counts[o["gpu"]] = counts.get(o["gpu"], 0) + o["n"]
        return " + ".join(f"{n}x {g}" for g, n in
                          sorted(counts.items(), key=lambda kv: -kv[1]))

    def price_note(self):
        """'@ $0.10/hr each' when uniform, otherwise the total."""
        per = {round(o["dph"] / max(o["n"], 1), 4) for o in self.offers}
        if len(per) == 1:
            return f"@ ${per.pop():.4f}/hr per GPU"
        return f"@ ${self.dph:.4f}/hr total"

    def key(self):
        return tuple(sorted(o["id"] for o in self.offers))


def greedy_fleet(offers, min_gpus, max_gpus, sort_key, label):
    """Take offers in `sort_key` order until the GPU budget is met."""
    picked, gpus = [], 0
    for o in sorted(offers, key=sort_key):
        if gpus >= max_gpus:
            break
        if gpus + o["n"] > max_gpus:
            continue  # this box overshoots the budget; try a smaller one
        picked.append(o)
        gpus += o["n"]
    if gpus < min_gpus or not picked:
        return None
    return Fleet(picked, label)


def candidate_fleets(ranked, min_gpus, max_gpus):
    """Build a spread of plausible fleets: cheapest-per-hash, fastest, and
    single-model fleets at every size in range (which is what makes the
    '4x 3090 vs 2x 4090' comparison possible)."""
    fleets, seen = [], set()

    def add(f):
        if f and f.hashrate > 0 and f.key() not in seen:
            seen.add(f.key())
            fleets.append(f)

    by_value = lambda o: (o["dph_per_ghs"], -o["hashrate"])       # noqa: E731
    by_speed = lambda o: (-o["hashrate"], o["dph_per_ghs"])       # noqa: E731

    # Whole-market fleets at each achievable size.
    for target in range(min_gpus, max_gpus + 1):
        add(greedy_fleet(ranked, target, target, by_value, "best value"))
        add(greedy_fleet(ranked, target, target, by_speed, "fastest"))

    # Single-model fleets, so homogeneous options show up explicitly.
    models = {o["gpu"] for o in ranked}
    for m in models:
        subset = [o for o in ranked if o["gpu"] == m]
        for target in range(min_gpus, max_gpus + 1):
            add(greedy_fleet(subset, target, target, by_value, f"all {m}"))

    return fleets


def speed_tolerance(speedup, premium, cap):
    """How much extra $/GH is acceptable for a given hashrate multiple."""
    import math
    if speedup <= 1.0:
        return 1.0
    return 1.0 + min(premium * math.log2(speedup), cap)


def select_fleet(fleets, premium, cap):
    """Cheapest $/GH sets the baseline; then buy the most hashrate that stays
    inside the premium tolerance. Returns (chosen, baseline)."""
    if not fleets:
        return None, None
    baseline = min(fleets, key=lambda f: f.dollars_per_ghs)
    base = baseline.dollars_per_ghs

    affordable = []
    for f in fleets:
        speedup = f.hashrate / baseline.hashrate if baseline.hashrate else 1.0
        allowed = base * speed_tolerance(speedup, premium, cap)
        if f.dollars_per_ghs <= allowed * (1 + 1e-9):
            affordable.append(f)

    chosen = max(affordable, key=lambda f: (f.hashrate, -f.dollars_per_ghs))
    return chosen, baseline


# --------------------------------------------------------------------------
def api_key():
    H.load_env()
    k = os.environ.get("VASTAI_API_KEY", "").strip()
    if not k:
        raise SystemExit("VASTAI_API_KEY is not set (put it in .env). "
                         "Get one at https://cloud.vast.ai/account/")
    return k


def req(method: str, path: str, key: str | None = None, **kw):
    import requests
    headers = kw.pop("headers", {})
    if key:
        headers["Authorization"] = f"Bearer {key}"
    headers.setdefault("Accept", "application/json")
    r = requests.request(method, f"{API}{path}", headers=headers, timeout=60, **kw)
    if r.status_code >= 400:
        raise SystemExit(f"vast.ai {method} {path} -> {r.status_code}: {r.text[:400]}")
    try:
        return r.json()
    except Exception:  # noqa: BLE001
        return {"raw": r.text}


# --------------------------------------------------------------------------
def search_offers(args, key=None):
    q = {
        "rentable": {"eq": True},
        # A range, not a fixed count: an 8x3090 box is often better value than
        # eight single-GPU boxes, and the planner can use either.
        "num_gpus": {"gte": args.min_gpus_per_box, "lte": args.max_gpus_per_box},
        "gpu_name": {"in": args.gpu},
        "dph_total": {"lte": args.max_price},
        "reliability2": {"gte": args.min_reliability},
        "inet_down": {"gte": 50},
        "cuda_max_good": {"gte": 12.0},
        "type": "bid" if args.interruptible else "on-demand",
        "order": [["dph_total", "asc"]],
        "limit": 200,
    }
    if getattr(args, "verified_only", False):
        q["verified"] = {"eq": True}
    import requests
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    r = requests.get(f"{API}/bundles/", params={"q": json.dumps(q)},
                     headers=headers, timeout=60)
    if r.status_code >= 400:
        raise SystemExit(f"vast.ai search failed {r.status_code}: {r.text[:300]}")
    offers = r.json().get("offers", [])
    # Filter out blacklisted hosts
    skip = set(getattr(args, "skip_hosts", None) or [])
    if skip:
        before = len(offers)
        offers = [o for o in offers if str(o.get("machine_id", "")) not in skip
                  and str(o.get("host_id", "")) not in skip]
        if len(offers) < before:
            log(f"skipped {before - len(offers)} offers from blacklisted hosts")
    return offers


def rank(offers, max_value):
    out = []
    for o in offers:
        name = o.get("gpu_name", "?")
        n = o.get("num_gpus", 1) or 1
        hr_each, src = gpu_hashrate(name)
        hr = hr_each * n
        dph = o.get("dph_total", 0.0) or 0.0
        if dph <= 0 or hr <= 0:
            continue
        cpt = H.cost_per_token(max_value, hr, dph)
        out.append({
            "id": o.get("id"), "gpu": name, "n": n, "dph": dph, "hashrate": hr,
            "src": src, "cost_per_token": cpt,
            "dph_per_ghs": dph / (hr / 1e9),
            "eta": H.expected_hashes(max_value) / hr,
            "rel": o.get("reliability2", 0.0), "geo": o.get("geolocation") or "?",
            "cuda": o.get("cuda_max_good"), "inet": o.get("inet_down", 0),
        })
    out.sort(key=lambda x: x["cost_per_token"])
    return out


def print_offers(ranked, limit, max_value, threshold):
    exp = H.expected_hashes(max_value)
    print()
    print(f"  difficulty now: max_value=0x{max_value:064x}")
    print(f"  expected hashes per HTK: {exp:.3e}   (~{H.mints_so_far(max_value):.0f} mints so far)")
    print()
    hdr = f"  {'ask_id':>9} {'GPU':<16}{'n':>2} {'$/hr':>7} {'hashrate':>11} {'ETA':>7} {'$/HTK':>8}  {'rel':>5} src"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in ranked[:limit]:
        flag = " " if r["cost_per_token"] <= threshold else "!"
        print(f"  {r['id']:>9} {r['gpu'][:16]:<16}{r['n']:>2} {r['dph']:>7.4f} "
              f"{H.fmt_hashrate(r['hashrate']):>11} {H.fmt_duration(r['eta']):>7} "
              f"{r['cost_per_token']:>7.2f}{flag} {r['rel']:>5.3f} {r['src']}")
    print()
    if ranked:
        b = ranked[0]
        print(f"  best value: ask {b['id']} - {b['n']}x {b['gpu']} at ${b['dph']:.4f}/hr")
        print(f"              ~${b['cost_per_token']:.2f} per HTK, ETA {H.fmt_duration(b['eta'])}")
        if b["cost_per_token"] > threshold:
            print(f"  ! even the best offer exceeds your ${threshold:.0f}/token threshold")
    print("  '!' marks offers above the cost threshold. Hashrates marked 'estimate'/'guess'")
    print("  are priors until real rigs report in; monitor.py refines them.")
    print()


# --------------------------------------------------------------------------
def describe_fleet(f, max_value, indent="  "):
    """One-line fleet summary."""
    return (f"{indent}{f.composition()} {f.price_note()}, "
            f"total {H.fmt_hashrate(f.hashrate)}, "
            f"est ${f.cost_per_token(max_value):.2f}/token, "
            f"est {f.eta_seconds(max_value)/3600:.1f} hrs per token")


def speed_frontier(fleets, baseline):
    """Fleets that are faster than the baseline and not dominated: no other
    fleet is both faster and cheaper per hash. This is the real menu of
    speed-for-money trades available right now."""
    faster = [f for f in fleets if f.hashrate > baseline.hashrate * 1.001]
    frontier = []
    for f in faster:
        if not any(g.hashrate >= f.hashrate and g.dollars_per_ghs < f.dollars_per_ghs
                   for g in faster if g is not f):
            frontier.append(f)
    frontier.sort(key=lambda f: f.hashrate)
    # Drop near-duplicates in hashrate, keeping the cheapest.
    out = []
    for f in frontier:
        if out and abs(f.hashrate - out[-1].hashrate) / max(f.hashrate, 1) < 0.02:
            if f.dollars_per_ghs < out[-1].dollars_per_ghs:
                out[-1] = f
            continue
        out.append(f)
    return out


def print_speed_ladder(fleets, chosen, baseline, max_value, premium, cap):
    """What each step up in speed actually costs, and the --speed-premium
    setting that would buy it. Makes the trade explicit instead of leaving the
    selection looking arbitrary."""
    import math
    frontier = speed_frontier(fleets, baseline)
    if not frontier:
        return
    print("  SPEED LADDER  (what more hashrate would cost)")
    print(f"    {'fleet':<26}{'GH/s':>7}{'ETA':>8}{'$/token':>9}"
          f"{'speed':>7}{'premium':>9}{'needs':>8}  ")
    print("    " + "-" * 76)
    base_line = (f"    {baseline.composition()[:26]:<26}{baseline.hashrate/1e9:>7.2f}"
                 f"{H.fmt_duration(baseline.eta_seconds(max_value)):>8}"
                 f"{baseline.cost_per_token(max_value):>9.2f}"
                 f"{'1.00x':>7}{'-':>9}{'-':>8}   cheapest per hash")
    print(base_line)
    for f in frontier:
        speedup = f.hashrate / baseline.hashrate
        required = f.dollars_per_ghs / baseline.dollars_per_ghs - 1.0
        # The --speed-premium value that would just admit this fleet.
        needed = required / math.log2(speedup) if speedup > 1 else 0.0
        ok = f.dollars_per_ghs <= baseline.dollars_per_ghs * speed_tolerance(
            speedup, premium, cap) * (1 + 1e-9)
        mark = " <- selected" if f is chosen else ("  accepted" if ok else "")
        print(f"    {f.composition()[:26]:<26}{f.hashrate/1e9:>7.2f}"
              f"{H.fmt_duration(f.eta_seconds(max_value)):>8}"
              f"{f.cost_per_token(max_value):>9.2f}"
              f"{speedup:>6.2f}x{required*100:>8.0f}%"
              f"{needed:>8.2f}{mark}")
    print()
    print(f"    'premium' is the extra $/GH that fleet costs; 'needs' is the")
    print(f"    --speed-premium value that would select it. Current setting: "
          f"{premium:.2f} (cap {cap:.2f}).")
    print()


def print_plans(fleets, chosen, baseline, max_value, threshold,
                premium=0.20, cap=0.60, limit=12):
    exp = H.expected_hashes(max_value)
    print()
    print(f"  difficulty {exp:.3e} hashes/HTK  (~{H.mints_so_far(max_value):.0f} mints so far)")
    print()

    ranked = sorted(fleets, key=lambda f: (f.dollars_per_ghs, -f.hashrate))
    hdr = (f"  {'':1}{'fleet':<26}{'GPUs':>5}{'$/hr':>8}{'GH/s':>8}"
           f"{'$/GH':>8}{'$/token':>9}{'ETA':>9}  basis")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    shown = 0
    for f in ranked:
        if shown >= limit and f is not chosen:
            continue
        shown += 1
        mark = ">" if f is chosen else ("*" if f is baseline else " ")
        cpt = f.cost_per_token(max_value)
        flag = "" if cpt <= threshold else "!"
        print(f"  {mark}{f.composition()[:26]:<26}{f.gpus:>5}{f.dph:>8.3f}"
              f"{f.hashrate/1e9:>8.2f}{f.dollars_per_ghs:>8.4f}"
              f"{cpt:>8.2f}{flag:<1}{H.fmt_duration(f.eta_seconds(max_value)):>9}"
              f"  {f.provenance}")
    print()
    print("  > selected    * cheapest per hash    ! above cost threshold")
    print()

    if baseline is not None:
        print_speed_ladder(fleets, chosen, baseline, max_value, premium, cap)

    if baseline is not None and chosen is not baseline:
        speedup = chosen.hashrate / baseline.hashrate if baseline.hashrate else 1.0
        extra = (chosen.dollars_per_ghs / baseline.dollars_per_ghs - 1) * 100
        print("  speed premium applied:")
        print(describe_fleet(baseline, max_value, indent="    cheapest: "))
        print(describe_fleet(chosen, max_value, indent="    selected: "))
        print(f"    -> {speedup:.2f}x the hashrate for {extra:+.1f}% per hash "
              f"({H.fmt_duration(baseline.eta_seconds(max_value))} -> "
              f"{H.fmt_duration(chosen.eta_seconds(max_value))} per token)")
        print()
    elif chosen is not None:
        print("  no fleet bought enough extra speed to justify a premium;")
        print("  the cheapest option is also the selection.")
        print()

    if chosen is not None:
        print("  SELECTED")
        print(describe_fleet(chosen, max_value, indent="    "))
        print(f"    {len(chosen.offers)} instance(s), burn ${chosen.dph:.4f}/hr = "
              f"${chosen.dph*24:.2f}/day")
        if chosen.provenance not in ("measured", "measured~"):
            print()
            print("    NOTE: hashrates are priors, not measurements. Every $/token")
            print("    figure above scales inversely with them - if the kernel does")
            print("    half these rates, tokens cost twice this. Run one box and let")
            print("    monitor.py calibrate before scaling up.")
        if chosen.cost_per_token(max_value) > threshold:
            print()
            print(f"    ! ${chosen.cost_per_token(max_value):.2f}/token exceeds your "
                  f"${threshold:.0f} threshold")
    print()


def plan_fleets(args, key=None):
    """Shared by `plan` and `auto`: returns (ranked_offers, fleets, chosen, baseline, max_value)."""
    _prev, max_value, _ = H.read_state_failover()
    offers = search_offers(args, key)
    if not offers:
        raise SystemExit("no offers matched. Loosen --max-price / --gpu / --min-reliability.")
    ranked = rank(offers, max_value)

    if args.max_burn:
        ranked = [o for o in ranked if o["dph"] <= args.max_burn]

    fleets = candidate_fleets(ranked, args.min_gpus, args.max_gpus)
    if args.max_burn:
        fleets = [f for f in fleets if f.dph <= args.max_burn]
    if not fleets:
        raise SystemExit(
            f"could not assemble a fleet of {args.min_gpus}-{args.max_gpus} GPUs "
            f"within the filters. Try --max-price / --max-burn / --gpu.")

    chosen, baseline = select_fleet(fleets, args.speed_premium, args.premium_cap)
    return ranked, fleets, chosen, baseline, max_value


def do_plan(args):
    ranked, fleets, chosen, baseline, max_value = plan_fleets(
        args, os.environ.get("VASTAI_API_KEY") or None)
    print_plans(fleets, chosen, baseline, max_value, args.cost_threshold,
                args.speed_premium, args.premium_cap, args.limit)
    print("  Nothing rented. Run `deploy.py auto --yes` to provision the selection.")
    print()


def do_auto(args):
    key = api_key()
    ranked, fleets, chosen, baseline, max_value = plan_fleets(args, key)
    print_plans(fleets, chosen, baseline, max_value, args.cost_threshold,
                args.speed_premium, args.premium_cap, args.limit)

    if not args.yes:
        print("  Nothing rented. Re-run with --yes to create these instances.")
        print()
        return

    provision(chosen.offers, args, max_value)


def provision(picked, args, max_value):
    """Create one vast.ai instance per chosen offer."""
    key = api_key()
    image = args.image or DEFAULT_IMAGE_INLINE

    # Preflight: validate field sizes BEFORE spending any API calls, so an
    # oversized onstart fails here with guidance rather than as a cryptic 400
    # after we have already churned through offer selection.
    r0 = picked[0]
    args._dph = r0["dph"]
    args._gpu_desc = f"{r0['n']}x {r0['gpu']}"
    sample_label = f"{args.rig_prefix}-{r0['id']}"[:VAST_MAX_LABEL]
    sample_onstart = build_onstart(args, sample_label)
    log(f"onstart is {len(sample_onstart)} bytes (vast.ai limit {VAST_MAX_ONSTART}), "
        f"mode={args.mode}")
    problems = check_vast_limits(image, sample_onstart, sample_label)
    if problems:
        raise SystemExit("cannot provision - vast.ai field limits exceeded:\n  - "
                         + "\n  - ".join(problems))

    created, failed = [], []
    for r in picked:
        rig_name = f"{args.rig_prefix}-{r['id']}"[:VAST_MAX_LABEL]
        args._dph = r["dph"]
        args._gpu_desc = f"{r['n']}x {r['gpu']}"
        body = {
            "client_id": "me",
            "image": image,
            "disk": args.disk,
            "runtype": "ssh",
            "onstart": build_onstart(args, rig_name),
            "label": rig_name,
        }
        if args.interruptible:
            body["price"] = round(r["dph"] * args.bid_multiplier, 4)
        log(f"creating instance from ask {r['id']} ({r['n']}x {r['gpu']}) as {rig_name} ...")
        try:
            res = req("PUT", f"/asks/{r['id']}/", key=key, json=body)
        except SystemExit as e:
            log(f"  -> FAILED: {e}")
            failed.append(r)
            continue
        if res.get("success"):
            log(f"  -> instance {res.get('new_contract')} created")
            created.append(r)
        else:
            log(f"  -> FAILED: {json.dumps(res)[:300]}")
            failed.append(r)

    print()
    if created:
        f = Fleet(created)
        print(f"  provisioned {len(created)}/{len(picked)} instance(s)")
        print(describe_fleet(f, max_value, indent="    live fleet: "))
    if failed:
        print(f"  {len(failed)} instance(s) failed to start "
              f"(offers go stale fast - re-run to top up)")
    print()
    print("  Rigs are booting. First heartbeat lands on ntfy in a few minutes.")
    print("  Watch with:      python3 monitor.py --watch")
    print("  Submitter must be running locally:  python3 submit_nonce.py")
    print()


def miner_tarball_b64():
    """gzip'd tar of the miner files, base64-encoded. Deterministic (fixed
    mtime/uid) so the same code produces the same blob."""
    import gzip
    import io
    import tarfile
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tf:
        for fn in FILES_TO_SHIP:
            data = (SCRIPT_DIR / fn).read_bytes()
            ti = tarfile.TarInfo(fn)
            ti.size = len(data)
            ti.mtime = 0
            tf.addfile(ti, io.BytesIO(data))
    gz = gzip.compress(raw.getvalue(), 9, mtime=0)
    return base64.b64encode(gz).decode()


# --------------------------------------------------------------------------
def build_onstart(args, rig_name):
    """Bootstrap script run on the rented box: get the miner code onto the box,
    install deps, run the watchdog. Never carries a wallet key or any secret.

    Delivery mode decides how the code arrives, which is what keeps this under
    vast.ai's 16 KB onstart limit:

      image   the code is baked into the Docker image; onstart just runs it.
              Nothing embedded. The robust choice for the current code size.
      fetch   onstart downloads a tarball from --code-url (a public URL - the
              miner code holds no secrets). Small onstart, no Docker Hub.
      inline  the code is gzip'd + base64'd into the onstart itself. Zero
              external dependencies, but only fits if the payload is small
              enough; the preflight guard in provision() rejects it otherwise.
    """
    status_topic = os.environ.get("NTFY_STATUS_TOPIC", H.DEFAULT_STATUS_TOPIC)

    lines = [
        "#!/bin/bash",
        "set -x",
        "exec > >(tee -a /var/log/htk_onstart.log) 2>&1",
        "echo '=== HTK bootstrap ==='",
        "export DEBIAN_FRONTEND=noninteractive",
        "nvidia-smi || true",
        "mkdir -p /opt/htk && cd /opt/htk",
    ]

    need_deps = args.mode in ("inline", "fetch")
    if need_deps:
        lines += [
            "apt-get update && apt-get install -y python3-pip git curl",
            f"python3 -m pip install --no-cache-dir {MINER_PIP}",
        ]

    if args.mode == "inline":
        # Single gzip'd blob, decoded and unpacked in one step.
        lines.append(f"echo '{miner_tarball_b64()}' | base64 -d | tar -xz -C /opt/htk")
    elif args.mode == "fetch":
        url = args.code_url
        if url.endswith(".git"):
            lines.append(f"git clone --depth 1 '{url}' /opt/htk/_src && "
                         "cp -r /opt/htk/_src/. /opt/htk/")
        else:
            # Any static host that serves a .tar.gz of the miner files.
            lines.append("curl -fsSL --retry 5 --retry-delay 3 "
                         f"'{url}' -o /tmp/htk.tgz && tar -xzf /tmp/htk.tgz -C /opt/htk")
    # image mode: files are already at args.workdir inside the image.

    env_exports = {
        "NTFY_TOPIC": os.environ.get("NTFY_TOPIC", H.DEFAULT_NONCE_TOPIC),
        "NTFY_STATUS_TOPIC": status_topic,
        "RIG_NAME": rig_name,
        "VAST_HOURLY_RATE": f"{args._dph:.4f}",
        "GPU_DESC": args._gpu_desc,
    }
    for k, v in env_exports.items():
        lines.append(f"export {k}='{v}'")

    workdir = args.workdir if args.mode == "image" else "/opt/htk"
    lines += [
        f"cd {workdir}",
        "python3 -u miner_watchdog.py "
        f"--rig '{rig_name}' --hourly-rate {args._dph:.4f} "
        f"--heartbeat {args.heartbeat} "
        ">> /var/log/htk_miner.log 2>&1 &",
        "echo '=== HTK bootstrap done ==='",
    ]
    return "\n".join(lines) + "\n"


def check_vast_limits(image, onstart, label):
    """Validate field sizes against vast.ai's create-instance limits. Returns a
    list of human-readable problems (empty if all fit)."""
    problems = []
    if len(image) > VAST_MAX_IMAGE:
        problems.append(f"image is {len(image)} chars (limit {VAST_MAX_IMAGE})")
    if len(label) > VAST_MAX_LABEL:
        problems.append(f"label is {len(label)} chars (limit {VAST_MAX_LABEL})")
    if len(onstart) > VAST_MAX_ONSTART:
        problems.append(
            f"onstart/args is {len(onstart)} bytes (limit {VAST_MAX_ONSTART}). "
            f"The miner code is too large to embed with --mode inline. Use a "
            f"delivery mode that does not carry the code in the script:\n"
            f"      --mode image   (build & push the Dockerfile once, then "
            f"pass --image you/htk-miner)\n"
            f"      --mode fetch --code-url https://.../htk-miner.tar.gz   "
            f"(host the tarball anywhere public)")
    return problems


def create(args):
    key = api_key()
    _prev, max_value, _ = H.read_state_failover()
    offers = search_offers(args, key)
    if not offers:
        raise SystemExit("no offers matched. Loosen --max-price / --gpu / --min-reliability.")
    ranked = rank(offers, max_value)
    print_offers(ranked, args.count + 5, max_value, args.cost_threshold)

    chosen = ranked[: args.count]
    if args.ask_id:
        chosen = [r for r in ranked if str(r["id"]) in args.ask_id] or None
        if not chosen:
            raise SystemExit(f"ask id(s) {args.ask_id} not present in current offers")

    print(f"  about to rent {len(chosen)} instance(s):")
    total = 0.0
    for r in chosen:
        total += r["dph"]
        print(f"    ask {r['id']}  {r['n']}x {r['gpu']}  ${r['dph']:.4f}/hr  "
              f"~${r['cost_per_token']:.2f}/HTK")
    print(f"  combined burn rate: ${total:.4f}/hr  =  ${total*24:.2f}/day")
    if any(r["cost_per_token"] > args.cost_threshold for r in chosen):
        print(f"  WARNING: at least one exceeds the ${args.cost_threshold:.0f}/token threshold")
    print()

    if not args.yes:
        print("  Nothing rented. Re-run with --yes to actually create these instances.")
        return

    provision(chosen, args, max_value)


def list_instances(args):
    key = api_key()
    res = req("GET", "/instances/", key=key)
    ins = res.get("instances", [])
    if not ins:
        print("  no instances")
        return
    print()
    print(f"  {'id':>10} {'status':<12}{'GPU':<18}{'$/hr':>7} {'uptime':>9}  label")
    print("  " + "-" * 72)
    total = 0.0
    for i in ins:
        dph = i.get("dph_total", 0) or 0
        total += dph if i.get("actual_status") == "running" else 0
        up = i.get("duration") or 0
        print(f"  {i.get('id'):>10} {str(i.get('actual_status')):<12}"
              f"{str(i.get('gpu_name'))[:18]:<18}{dph:>7.4f} "
              f"{H.fmt_duration(up):>9}  {i.get('label') or ''}")
    print(f"\n  running burn rate: ${total:.4f}/hr = ${total*24:.2f}/day\n")


def destroy(args):
    key = api_key()
    ids = list(args.ids)
    if args.all:
        res = req("GET", "/instances/", key=key)
        ids = [str(i["id"]) for i in res.get("instances", [])]
    if not ids:
        print("  nothing to destroy")
        return
    print(f"  will destroy: {', '.join(ids)}")
    if not args.yes:
        print("  Re-run with --yes to confirm.")
        return
    for i in ids:
        res = req("DELETE", f"/instances/{i}/", key=key)
        log(f"destroy {i}: {json.dumps(res)[:120]}")


def do_search(args):
    _prev, max_value, _ = H.read_state_failover()
    offers = search_offers(args, os.environ.get("VASTAI_API_KEY") or None)
    if not offers:
        raise SystemExit("no offers matched your filters")
    print_offers(rank(offers, max_value), args.limit, max_value, args.cost_threshold)


def main():
    p = argparse.ArgumentParser(description="Provision HTK miners on vast.ai")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--gpu", nargs="+",
                        default=["RTX 5090", "RTX 4090", "RTX 4080", "RTX 3090", "RTX 3080"],
                        help="acceptable GPU names")
        sp.add_argument("--max-price", type=float, default=2.00, help="max $/hr per box")
        sp.add_argument("--min-reliability", type=float, default=0.95)
        sp.add_argument("--min-gpus-per-box", type=int, default=1)
        sp.add_argument("--max-gpus-per-box", type=int, default=8)
        sp.add_argument("--interruptible", action="store_true",
                        help="bid on interruptible capacity (cheaper, can be preempted)")
        sp.add_argument("--verified-only", action="store_true",
                        help="only rent from verified hosts (more reliable, costs more)")
        sp.add_argument("--skip-hosts", nargs="+", default=[],
                        help="host/machine IDs to blacklist (avoid broken hosts)")
        sp.add_argument("--cost-threshold", type=float,
                        default=float(os.environ.get("COST_PER_TOKEN_ALERT", "50")),
                        help="flag offers above this $/token")

    def planning(sp):
        sp.add_argument("--min-gpus", type=int, default=4, help="smallest acceptable fleet")
        sp.add_argument("--max-gpus", type=int, default=8, help="largest fleet to build")
        sp.add_argument("--gpus", type=int,
                        help="shorthand: exactly this many GPUs (sets both bounds)")
        sp.add_argument("--speed-premium", type=float, default=0.20,
                        help="extra $/GH tolerated per doubling of hashrate (0.20 = 20%%)")
        sp.add_argument("--premium-cap", type=float, default=0.60,
                        help="hard ceiling on the speed premium")
        sp.add_argument("--max-burn", type=float, help="cap total fleet $/hr")
        sp.add_argument("--limit", type=int, default=12, help="rows of plan table to show")

    def provisioning(sp):
        sp.add_argument("--mode", choices=["image", "fetch", "inline"], default="image",
                        help="how miner code reaches the box: image (baked into "
                             "--image, robust), fetch (download from --code-url), "
                             "inline (embed in onstart; only fits small payloads)")
        sp.add_argument("--image", help="docker image (required for --mode image)")
        sp.add_argument("--code-url", default=os.environ.get("MINER_CODE_URL", ""),
                        help="tarball or .git URL of the miner code (--mode fetch)")
        sp.add_argument("--workdir", default="/opt/htk", help="miner dir inside a prebuilt image")
        sp.add_argument("--disk", type=int, default=12, help="GB")
        sp.add_argument("--rig-prefix", default="htk")
        sp.add_argument("--heartbeat", type=int, default=900)
        sp.add_argument("--bid-multiplier", type=float, default=1.15,
                        help="bid this multiple of the ask when --interruptible")
        sp.add_argument("--yes", action="store_true", help="actually spend money")

    s = sub.add_parser("search", help="rank individual offers by cost per token (no spend)")
    common(s)
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(func=do_search)

    pl = sub.add_parser("plan", help="pick the best multi-GPU fleet (no spend)")
    common(pl)
    planning(pl)
    pl.set_defaults(func=do_plan)

    au = sub.add_parser("auto", help="pick the best fleet and provision it")
    common(au)
    planning(au)
    provisioning(au)
    au.set_defaults(func=do_auto)

    c = sub.add_parser("create", help="rent instances (manual, single-offer selection)")
    common(c)
    provisioning(c)
    c.add_argument("--count", type=int, default=1)
    c.add_argument("--ask-id", nargs="+", help="rent these specific ask ids")
    c.set_defaults(func=create)

    l = sub.add_parser("list", help="list your instances")
    l.set_defaults(func=list_instances)

    d = sub.add_parser("destroy", help="destroy instances")
    d.add_argument("ids", nargs="*")
    d.add_argument("--all", action="store_true")
    d.add_argument("--yes", action="store_true")
    d.set_defaults(func=destroy)

    args = p.parse_args()
    H.load_env()
    mode = getattr(args, "mode", None)
    if mode == "image" and not args.image:
        raise SystemExit(
            "--mode image needs --image you/htk-miner:tag.\n"
            "  Build and push it once (free Docker Hub account):\n"
            "    docker build -t you/htk-miner .\n"
            "    docker push you/htk-miner\n"
            "  Then: deploy.py auto --gpus 8 --mode image --image you/htk-miner --yes\n"
            "  Or skip Docker with --mode fetch --code-url https://.../htk-miner.tar.gz")
    if mode == "fetch" and not args.code_url:
        raise SystemExit(
            "--mode fetch needs --code-url (or MINER_CODE_URL in .env): a public "
            "tarball or .git URL of the miner files. The miner code holds no "
            "secrets, so any public host works.")
    if getattr(args, "gpus", None):
        args.min_gpus = args.max_gpus = args.gpus
    if getattr(args, "min_gpus", 0) > getattr(args, "max_gpus", 0) > 0:
        raise SystemExit("--min-gpus cannot exceed --max-gpus")
    args.func(args)


if __name__ == "__main__":
    main()
