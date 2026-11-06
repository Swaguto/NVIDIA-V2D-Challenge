import subprocess
from types import SimpleNamespace
from v2d_reliability.telemetry import DeviceSampler


def test_sampling_records_device_scope(monkeypatch):
    sampler = DeviceSampler()
    def run(*args, **kwargs):
        sampler.stop_event.set()
        return SimpleNamespace(stdout="GPU-fixture, 100\n")
    monkeypatch.setattr(subprocess, "run", run)
    sampler.start(); sampler.thread.join(); report = sampler.finish()
    assert report["scope"] == "whole_device_including_other_processes"
    assert report["sampled_peak_memory_mib"] == {"GPU-fixture": 100}


def test_missing_gpu_tool_is_reported(monkeypatch):
    sampler = DeviceSampler()
    def run(*args, **kwargs): raise FileNotFoundError("fixture no GPU tool")
    monkeypatch.setattr(subprocess, "run", run)
    sampler.start(); sampler.thread.join()
    report = sampler.finish()
    assert report["errors"] and not report["sampled_peak_memory_mib"]
