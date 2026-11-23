"""Sample process RSS without inspecting process arguments or contacting an API."""

import asyncio
import os
import subprocess


def process_rss(pids):
    try:
        result = subprocess.run(["ps", "-o", "pid=,rss=", "-p", ",".join(map(str, pids))],
                                capture_output=True, text=True, timeout=2, check=False)
        return {int(pid): int(rss) / 1024 for pid, rss in
                (line.split() for line in result.stdout.splitlines() if len(line.split()) == 2)}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


class ResourceSampler:
    def __init__(self, server_pid=None, interval=.5):
        self.pids = {"app": os.getpid(), "server": server_pid}
        self.interval = interval
        self.observed = {key: [] for key in self.pids}
        self.stop = asyncio.Event()

    async def sample(self):
        measurements = await asyncio.to_thread(process_rss, [pid for pid in self.pids.values() if pid])
        for key, pid in self.pids.items():
            if pid in measurements:
                self.observed[key].append(measurements[pid])

    async def loop(self):
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(self.stop.wait(), self.interval)
            except TimeoutError:
                await self.sample()

    async def __aenter__(self):
        await self.sample()
        self.task = asyncio.create_task(self.loop())
        return self

    async def __aexit__(self, *exc):
        self.stop.set()
        await self.task
        await self.sample()

    def result(self):
        return {"method": "sampled process RSS in MiB; not GPU/Metal memory or exact allocation peak",
                "interval_seconds": self.interval,
                **{key: {"pid": self.pids[key], "samples": len(values),
                         "baseline_mib": values[0] if values else None,
                         "peak_sampled_mib": max(values) if values else None,
                         "status": "measured" if values else "not_requested" if not self.pids[key] else "unavailable"}
                   for key, values in self.observed.items()}}
