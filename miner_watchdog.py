#!/usr/bin/env python3
"""Supervisor for htk_cuda_miner.py. This is what runs on the rented GPU box.

Responsibilities:
  * keep the miner alive across crashes, driver hiccups and OOM kills
  * parse the miner's stdout for hashrate / found nonces
  * push a periodic heartbeat to ntfy so you can tell a live rig from a dead one
  * track spend at the instance's $/hr and report cost-per-token in every beat

It holds no private key and never touches the wallet: found nonces go out over
ntfy and are submitted from the operator's machine by submit_nonce.py.

A vast.ai instance can be preempted at any moment. Nothing here needs to survive
that: mining is memoryless, so a restart loses no progress. What matters is that
the heartbeat stops, which is the signal the instance is gone.
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import htk_common as H
from htk_common import log

SCRIPT_DIR = Path(__file__).resolve().parent

RATE_RE = re.compile(r"\[rate\] total ([\d.]+) GH/s")
FOUND_RE = re.compile(r"\*\*\* FOUND nonce (0x[0-9a-fA-F]{64})")
READY_RE = re.compile(r"~([\d.]+) MH/s")


class Watchdog:
    def __init__(self, args):
        H.load_env()
        self.args = args
        self.rig = args.rig or os.environ.get("RIG_NAME") or f"rig-{os.urandom(3).hex()}"
        self.status_topic = os.environ.get("NTFY_STATUS_TOPIC", H.DEFAULT_STATUS_TOPIC)
        self.rate_usd = float(os.environ.get("VAST_HOURLY_RATE", args.hourly_rate or 0.0))
        self.gpu_desc = os.environ.get("GPU_DESC", "")

        self.start_ts = time.time()
        self.hashrate_ghs = 0.0
        self.found = 0
        self.restarts = 0
        self.last_line_ts = time.time()
        self.stop = threading.Event()
        self.proc = None
        self.lock = threading.Lock()

    # ------------------------------------------------------------------
    def uptime(self):
        return time.time() - self.start_ts

    def spend(self):
        return self.uptime() / 3600.0 * self.rate_usd

    def status_body(self):
        hr = self.hashrate_ghs * 1e9
        lines = [
            f"rig {self.rig}{(' [' + self.gpu_desc + ']') if self.gpu_desc else ''}",
            f"uptime   {H.fmt_duration(self.uptime())}",
            f"hashrate {H.fmt_hashrate(hr)}",
            f"found    {self.found} nonce(s)",
            f"restarts {self.restarts}",
        ]
        if self.rate_usd > 0:
            lines.append(f"spend    ${self.spend():.2f} @ ${self.rate_usd:.3f}/hr")
        try:
            _prev, mx, _ = H.read_state_failover()
            if hr > 0:
                eta = H.expected_hashes(mx) / hr
                lines.append(f"eta/token {H.fmt_duration(eta)}")
                if self.rate_usd > 0:
                    cpt = H.cost_per_token(mx, hr, self.rate_usd)
                    lines.append(f"cost/token ${cpt:.2f}")
        except Exception:  # noqa: BLE001
            lines.append("chain unreachable for ETA")
        return "\n".join(lines)

    def heartbeat_loop(self):
        # First beat after a short delay so autotune has settled.
        if self.stop.wait(90):
            return
        while not self.stop.is_set():
            body = self.status_body()
            log("[heartbeat]\n" + body)
            H.ntfy_push(self.status_topic, body,
                        title=f"HTK {self.rig} alive", tags="green_circle")
            if self.stop.wait(self.args.heartbeat):
                return

    # ------------------------------------------------------------------
    def _pump(self, stream):
        """Read miner stdout, mirror it, and scrape metrics out of it."""
        for raw in iter(stream.readline, ""):
            if self.stop.is_set():
                break
            line = raw.rstrip("\n")
            self.last_line_ts = time.time()
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

            m = RATE_RE.search(line)
            if m:
                with self.lock:
                    self.hashrate_ghs = float(m.group(1))
                continue
            m = FOUND_RE.search(line)
            if m:
                with self.lock:
                    self.found += 1
                log(f"[watchdog] nonce found -> {m.group(1)}")
                continue
            if self.hashrate_ghs == 0.0:
                m = READY_RE.search(line)
                if m:
                    with self.lock:
                        self.hashrate_ghs = float(m.group(1)) / 1000.0

    def stall_monitor(self):
        """A wedged CUDA context prints nothing. Kill it and let the loop restart."""
        while not self.stop.is_set():
            if self.stop.wait(30):
                return
            quiet = time.time() - self.last_line_ts
            if quiet > self.args.stall_timeout and self.proc and self.proc.poll() is None:
                log(f"[watchdog] no output for {quiet:.0f}s - miner appears wedged, killing it")
                H.ntfy_push(self.status_topic,
                            f"rig {self.rig}: miner stalled {quiet:.0f}s, restarting",
                            title=f"HTK {self.rig} stalled", tags="warning")
                try:
                    self.proc.kill()
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------------------------------------------
    def run(self):
        cmd = [sys.executable, "-u", str(SCRIPT_DIR / "htk_cuda_miner.py")]
        if self.args.miner_args:
            cmd += self.args.miner_args.split()

        log(f"[watchdog] rig={self.rig} rate=${self.rate_usd:.3f}/hr cmd={' '.join(cmd)}")
        H.ntfy_push(self.status_topic,
                    f"rig {self.rig} starting\n{self.gpu_desc or 'unknown GPU'}\n"
                    f"${self.rate_usd:.3f}/hr",
                    title=f"HTK {self.rig} boot", tags="rocket")

        def _sig(_s, _f):
            log("[watchdog] signal received, shutting down")
            self.stop.set()
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
        signal.signal(signal.SIGINT, _sig)
        signal.signal(signal.SIGTERM, _sig)

        threading.Thread(target=self.heartbeat_loop, daemon=True).start()
        threading.Thread(target=self.stall_monitor, daemon=True).start()

        backoff = 5.0
        while not self.stop.is_set():
            run_start = time.time()
            self.last_line_ts = time.time()
            try:
                self.proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, cwd=str(SCRIPT_DIR),
                )
            except Exception as e:  # noqa: BLE001
                log(f"[watchdog] failed to spawn miner: {e}")
                if self.stop.wait(backoff):
                    break
                backoff = min(backoff * 2, 300)
                continue

            self._pump(self.proc.stdout)
            code = self.proc.wait()
            ran = time.time() - run_start

            if self.stop.is_set():
                break

            self.restarts += 1
            log(f"[watchdog] miner exited code={code} after {H.fmt_duration(ran)} "
                f"(restart #{self.restarts})")

            # A miner that dies instantly is misconfigured, not unlucky. Back off
            # hard so we do not spin, and shout about it.
            if ran < 60:
                backoff = min(backoff * 2, 300)
                H.ntfy_push(self.status_topic,
                            f"rig {self.rig}: miner died after {ran:.0f}s (code {code}), "
                            f"retry in {backoff:.0f}s - check the image/driver",
                            title=f"HTK {self.rig} crashloop", tags="rotating_light",
                            priority=4)
            else:
                backoff = 5.0
                H.ntfy_push(self.status_topic,
                            f"rig {self.rig}: miner restarted after "
                            f"{H.fmt_duration(ran)} (code {code})",
                            title=f"HTK {self.rig} restart", tags="warning")

            if self.stop.wait(backoff):
                break

        log("[watchdog] stopped")
        H.ntfy_push(self.status_topic,
                    f"rig {self.rig} stopped\nuptime {H.fmt_duration(self.uptime())}\n"
                    f"spend ${self.spend():.2f}\nfound {self.found}",
                    title=f"HTK {self.rig} down", tags="octagonal_sign", priority=4)


def main():
    p = argparse.ArgumentParser(description="Keep the HTK miner alive and reporting")
    p.add_argument("--rig", help="name for this rig (default: $RIG_NAME or random)")
    p.add_argument("--hourly-rate", type=float, help="instance $/hr, for cost tracking")
    p.add_argument("--heartbeat", type=int, default=900, help="seconds between ntfy heartbeats")
    p.add_argument("--stall-timeout", type=int, default=300,
                   help="restart the miner if it prints nothing for this long")
    p.add_argument("--miner-args", default="", help="extra args passed to htk_cuda_miner.py")
    Watchdog(p.parse_args()).run()


if __name__ == "__main__":
    main()
