# HTK miner toolchain

GPU mining for [HashToken (HTK)](https://etherscan.io/address/0xe5544a2a5fa9b175da60d8eec67add5582bb31b0),
a 2016 proof-of-work ERC-20 on Ethereum mainnet.

The CUDA miner (`keccak_miner.cu`, `htk_cuda_miner.py`) is from
[Ifonlywe/HashToken](https://github.com/Ifonlywe/HashToken). Everything else here
is new: on-chain submission, vast.ai fleet provisioning, supervision, cost
tracking, and competition analysis.

---

## Weekend quickstart

Everything below runs on a laptop. The GPUs are rented; your machine only holds
the wallet key and submits transactions. Budget ~20 minutes to set up.

### 0. Prerequisites

- Python 3.9+
- A **burner wallet** with ~0.01 ETH for gas (never your main wallet)
- Free accounts: [Alchemy](https://dashboard.alchemy.com) (RPC),
  [Etherscan](https://etherscan.io/apis) (analysis),
  [vast.ai](https://cloud.vast.ai/account/) (GPUs, needs a card on file)

```bash
git clone git@github.com:cartoonitunes/htk-miner.git
cd htk-miner
pip install -r requirements-local.txt
```

### 1. Configure

```bash
cp .env.example .env
```

Generate your own ntfy topics and paste them into `.env`:

```bash
python3 -c "import secrets;print('NTFY_TOPIC=htk-nonce-'+secrets.token_hex(12));print('NTFY_STATUS_TOPIC=htk-status-'+secrets.token_hex(12))"
```

> **Do not skip this.** The topics hardcoded as defaults come from the public
> upstream repo. ntfy has no authentication — the topic name *is* the password —
> so anyone reading that repo can subscribe, watch your nonces arrive, and mint
> with them before you do. Generating your own takes five seconds.

Then fill in `MAINNET_RPC_URL`, `MINER_PRIVATE_KEY`, `VASTAI_API_KEY`, and
`ETHERSCAN_API_KEY`.

### 2. Verify before spending anything

```bash
python3 test_htk.py --live
```

109 tests. They check the hashing and validation logic against a real winning
mint pulled from mainnet history. If these fail, stop — something is wrong and
submitting would burn gas.

```bash
python3 submit_nonce.py --status
```

Prints your wallet, ETH balance, current difficulty, and the send path. Confirm
the wallet address and that `send via flashbots` appears.

### 3. See what it will cost

```bash
python3 competition.py --days 365    # who else is mining, and what it costs you
python3 deploy.py plan --gpus 8      # rank fleets; spends nothing
```

Read the **speed ladder** in the plan output before renting — it shows what each
step up in hashrate costs and the `--speed-premium` that buys it.

### 4. Calibrate with ONE box first

This is the step that matters most. Every dollar estimate depends on a hashrate
prior that has never been measured on this kernel.

```bash
python3 deploy.py auto --gpus 1 --yes
```

Wait ~15 minutes for the first heartbeat, then:

```bash
python3 monitor.py --calibrate       # records the real hashrate
python3 deploy.py plan --gpus 8      # re-plan using measured numbers
```

If the measured rate is far below the prior, the honest move is to stop here.
One box for an hour costs ~$0.15 and resolves the biggest unknown in the model.

### 5. Start the submitter, THEN scale up

Run the submitter **before** the miners, so an early nonce is not missed. Leave
it running in its own terminal — it is the only thing that can turn a found
nonce into a token.

```bash
python3 submit_nonce.py              # terminal 1, leave running
```

```bash
python3 deploy.py auto --gpus 8 --yes   # terminal 2
python3 monitor.py --watch              # terminal 3
```

### 6. Stop the meter

```bash
python3 deploy.py list
python3 deploy.py destroy --all --yes
```

vast.ai bills by the second while instances exist. **Destroy them when you are
done** — an idle rig you forgot about costs the same as a working one.

---

## Read this before spending anything

I verified the contract against its deployed bytecode and mainnet history. Five
things materially affect how you should run this.

**1. A token costs real money and HTK has no market.** Current difficulty is
`1.035e15` hashes per token (2^49.9). On the hashrate priors in `deploy.py` a
mid-size fleet lands around **$8–20 per HTK**. There is no liquidity for this
token anywhere. Treat the spend as the cost of a collectible, not an investment.

> **The hashrate priors are unverified, and everything scales with them.**
> `HASHRATE_ESTIMATES` uses the supplied figures (3090 = 1.5 GH/s, 4090 = 2.5,
> and so on), which assume a well-optimised keccak kernel. The upstream kernel
> keeps the full 25-word Keccak state in local memory and does one hash per
> thread, so it may run well below that. If it does half, tokens cost twice the
> estimate. **This is the single largest uncertainty in the whole model** — far
> larger than competition. Do step 4 above.

**2. Every mint by anyone makes it 1% harder, forever.** The contract does
`max_value -= max_value / 100` on each mint and never lowers it. ~3,371 mints
have happened since 2016.

**3. A failed mint burns 100% of your gas limit.** The contract is Solidity
0.3.5, where `throw` compiles to the `INVALID` opcode — it consumes the whole
limit, not just gas used. Confirmed on-chain: successful mints use exactly
38,721 gas; reverted ones consume their full limit. Hence the tight default
`GAS_LIMIT=60000` and the layered pre-checks in `submit_nonce.py`.

**4. `mint(bytes32)` does not bind the nonce to `msg.sender`.** Anyone who sees
your transaction in the public mempool can copy the nonce and mint it themselves.
**Submitting through the public mempool risks losing the token you just paid days
of GPU time for.** `submit_nonce.py` defaults to the Flashbots private relay. Do
not set `SUBMIT_RELAY=public` without understanding the trade.

**5. Your ntfy topic is a bearer secret.** See step 1. Nonces travel over it in
plaintext to an unauthenticated public service.

---

## What the competition actually looks like

From `competition.py --days 365`, using real `Mint(address)` events.

**Mining here is bursty, not steady.** Mints per calendar month:

```
2025-07   394  ######################################
2025-08    33  ###
2025-09     2  #
2026-05     8  #
2026-07     3  #
```

Someone points a farm at the contract for days, then it goes quiet for months.
There was a **247-day dormancy** (Sep 2025 → May 2026) and a 71-day one before
July 2026. Historical volume is dominated by four addresses — `0x0a374b22` (177
mints, 40%), `0x0a7da295` (97), `0x606ae615` (86), `0x5203d2a7` (69) — none of
which have mined recently.

**A small burst is running now.** Three mints in three days (22–24 July 2026),
each from a *different* address. Implied network hashrate by window: 24h →
23.6 GH/s (n=2), 7d → 5.0 GH/s (n=3), 30d → 1.2 GH/s (n=3). Those are Poisson
estimates on 2–3 samples, so ±60–70%.

### Competition barely affects your cost, and here is why

This is counterintuitive but it holds:

1. **No difficulty retargeting.** Your find rate is `your_hashrate / E` — it does
   not depend on how many others are hashing. Bitcoin works the opposite way,
   where doubling network hashrate halves your share. Not here.
2. **Keccak search is memoryless.** When a competitor mints and `prev_hash`
   moves, your discarded partial search has exactly the same expected remaining
   work as a fresh one. Losing a race costs nothing in expectation.

The only real competitive cost is the 1%-per-mint ratchet others push onto the
target while you search. `E` grows linearly at `0.01 × net_hashrate`, which gives
a clean closed form for your mean time to win one token:

```
E[T] = E₀ / (your_hashrate − 0.01 × net_hashrate)
```

The field simply subtracts 1% of its hashrate from yours. (Verified against
numerical integration of the survival function to 6 significant figures; see
`test_htk.py`.) Your search only stalls outright if the field exceeds **100× your
fleet**, which has never happened on this contract.

For an 8× RTX 3080 fleet (9.6 GH/s, $0.378/hr):

| Network hashrate | Your share | Drag | $/token |
|---|---|---|---|
| 1.2 GH/s (30d) | 89% | 0.12% | $11.33 |
| 5.0 GH/s (7d) | 66% | 0.53% | $11.38 |
| 23.6 GH/s (24h) | 29% | 2.52% | $11.60 |
| 100 GH/s (stress) | 9% | 11.6% | $12.63 |

**Competition moves cost per token by under 3% at observed levels. The hashrate
prior moves it by 100%+.** Worry about the kernel, not the competitors.

---

## Architecture: the key never leaves your machine

```
   vast.ai GPU instance              ntfy.sh                 your laptop
   ────────────────────              ───────                 ───────────
   htk_cuda_miner.py    ── nonce ──▶  topic   ── nonce ──▶  submit_nonce.py
   miner_watchdog.py    ── beat ───▶  status                  │  holds the key
     (no wallet key)                                          ▼
                                                        Flashbots relay ─▶ mint()
```

The rented box only ever publishes 32-byte nonces. It has no key, no RPC
credentials, and no ability to move funds. If the instance is compromised, the
worst case is a stolen nonce — not a stolen wallet. `deploy.py` never puts a
secret in the instance bootstrap script; there is a test asserting this.

---

## Files

| File | Where it runs | What it does |
|---|---|---|
| `keccak_miner.cu` | GPU | CUDA keccak256 brute force (upstream) |
| `htk_cuda_miner.py` | GPU | Multi-GPU orchestrator, autotune, chain polling (upstream) |
| `miner_watchdog.py` | GPU | Restarts the miner, detects stalls, heartbeats to ntfy |
| `htk_common.py` | both | Contract constants, keccak, validation, difficulty maths |
| `submit_nonce.py` | **laptop** | Verifies nonces and sends `mint()` via a private relay |
| `deploy.py` | laptop | vast.ai fleet planning and provisioning |
| `monitor.py` | laptop | Cost tracking, fleet dashboard, hashrate calibration |
| `competition.py` | laptop | On-chain mining activity and competitive cost analysis |
| `test_htk.py` | laptop | 109 tests, incl. vectors from real mainnet mints |
| `Dockerfile` | — | Optional prebuilt image for vast.ai |

---

## Fleet selection

`deploy.py plan` scores fleets on two axes: cost efficiency (`$/GH` — dollars per
hour per GH/s) and total hashrate. These are not independent — expected cost per
token is exactly proportional to `$/GH` — so the real trade is *paying a higher
$/GH to shorten the ETA*.

The **speed premium** governs that trade. `--speed-premium 0.20` means "20% more
per hash is acceptable for 2x the hashrate", generalised as
`tolerance = 1 + premium × log2(speedup)`, capped by `--premium-cap`. Among every
fleet inside that tolerance it takes the fastest.

The speed ladder shows what each step costs and what setting would buy it:

```
    fleet                        GH/s     ETA  $/token  speed  premium   needs
    4x RTX 3080                  4.80    2.5d     8.14  1.00x        -       -
    8x RTX 3080                  9.60    1.2d    11.64  2.00x      33%    0.33
    8x RTX 4090                 20.00   14.4h    15.93  4.17x      81%    0.40
```

At the default `0.20` nothing beats the cheapest fleet; `0.33` buys 2x the speed,
`0.40` buys 4.2x. Pick from the ladder rather than guessing.

Useful flags: `--max-burn 2.00` caps total fleet $/hr, `--gpu "RTX 3090"`
restricts models, `--interruptible` bids on preemptible capacity.

---

## How submission stays safe

`submit_nonce.py` runs four checks before broadcasting, in increasing cost order:

1. **Dedup** — a nonce already handled is skipped.
2. **Fresh chain read** — `prev_hash` and `max_value` are re-read at submit time.
   The miner's snapshot is never trusted; it may be seconds stale.
3. **Local verification** — `keccak256(nonce ‖ prev_hash) <= max_value`, computed
   locally. Free, and it catches lost races before they cost gas.
4. **`eth_estimateGas`** — a final simulation against latest state. If it
   reverts, nothing is sent.

Only then does it sign and send, through a private relay, with a tight gas limit.

Because `prev_hash` advances deterministically (below), a rejected nonce is
classified rather than discarded:

- **stale** — mined against a state already gone; another miner won. Dropped.
- **premature** — valid for a state that has not arrived yet. **Held in memory
  and retried** as the chain advances.

---

## The contract's deterministic hash chain

```solidity
prev_hash = sha3(block.blockhash(block.number), prev_hash);
```

`block.blockhash(block.number)` is **always zero** in the EVM — you can only read
hashes of the previous 256 blocks, never the current one. So the update is really
`prev_hash = keccak256(bytes32(0) ‖ prev_hash)`: every future `prev_hash` is
computable right now. Verified against archive state across the mint in block
25,600,440, and covered by `test_htk.py`.

Two consequences this toolchain uses:

- Rejected nonces can be told apart (premature vs stale) with no extra RPC calls.
- **Mining ahead is possible.** You can mine against `prev_hash + N` and bank
  solutions for rounds that have not happened yet. Mine against the *projected*
  `max_value` too (`H.project_state`), since the target tightens 1% per mint. The
  upstream miner does not do this; the helper is there if you want to build it.

---

## Notes on the upstream miner

- Run `python3 htk_cuda_miner.py --self-test` on a new GPU type before trusting
  it. It verifies GPU keccak output against a CPU implementation. Worth doing:
  the kernel casts an `unsigned char[64]` stack buffer to `uint64_t*`, which
  relies on alignment the compiler is not required to guarantee.
- Nonces it produces have the layout `base(8B LE) ‖ thread_id(8B LE) ‖ 16 zero
  bytes`. Large enough to be a non-issue, but note that other miners' winning
  nonces on-chain use all 32 bytes.
- It polls the chain every 12s and restarts the search when `prev_hash` moves.
  Mining is memoryless, so an interrupted search loses nothing — which is also
  why vast.ai preemption is harmless here.

## Safety

- `MINER_PRIVATE_KEY` belongs in `.env` on your laptop and nowhere else.
  `.gitignore` covers `.env`; `deploy.py` never puts it in an onstart script.
- Use a burner wallet holding only gas money.
- `deploy.py auto` / `create` / `destroy` refuse to act without `--yes`.
- `monitor.py` alerts at most once an hour so a cost breach cannot spam you.
- No API keys are committed to this repo. `competition.py` requires
  `ETHERSCAN_API_KEY` from the environment and will not run without it.
