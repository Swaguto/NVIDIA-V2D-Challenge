"""Optional device-level GPU sampling; shared-device usage is not process usage."""
import subprocess
import threading
import time


class DeviceSampler:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.stop_event = threading.Event()
        self.samples = []
        self.errors = []
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                p = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5, check=True)
                for line in p.stdout.strip().splitlines():
                    uuid, memory = line.split(",")
                    self.samples.append({"time": time.time(), "uuid": uuid.strip(), "memory_mib": float(memory)})
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self.errors.append(str(exc))
                break
            self.stop_event.wait(self.interval)

    def finish(self):
        self.stop_event.set()
        self.thread.join(timeout=6)
        peaks = {}
        for sample in self.samples:
            peaks[sample["uuid"]] = max(peaks.get(sample["uuid"], 0), sample["memory_mib"])
        return {"scope": "whole_device_including_other_processes", "interval_seconds": self.interval,
                "sampled_peak_memory_mib": peaks, "samples": self.samples, "errors": self.errors,
                "note": "Sampled peaks may miss transients; do not attribute shared GPU memory to this run."}
