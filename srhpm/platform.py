"""Host detection for the SR-HPM launcher: OS, VRAM, RAM, cores, large pages, CPU dot-product extensions, disk.

Each probe degrades to a conservative answer instead of failing, so the launcher always produces a plan.
"""
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

GIB = 2**30


@dataclass
class Host:
    os: str                     # "windows" | "linux" | "darwin"
    vram_total_gib: float
    vram_free_gib: float        # usable for weights after the driver reserve
    ram_gib: float
    cores: int
    large_pages: bool           # 2 MB pages usable for the weights (privilege on Windows, THP on Linux)
    large_pages_note: str
    cpu_name: str
    vnni: bool                  # AVX-VNNI or AVX512-VNNI: one instruction less per token in the SR-HPM kernels
    vnni_note: str
    notes: list = field(default_factory=list)


def _run(cmd: list[str], timeout: int = 10) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def _vram() -> tuple[float, float]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return 0.0, 0.0
    out = _run([exe, "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"]).split("\n")[0]
    try:
        total, used = (float(x) for x in out.split(","))
        return total / 1024, (total - used) / 1024
    except ValueError:
        return 0.0, 0.0


def _ram_cores() -> tuple[float, int]:
    try:
        import psutil
        return psutil.virtual_memory().total / GIB, psutil.cpu_count(logical=False) or os.cpu_count() or 4
    except Exception:
        return 16.0, max(1, (os.cpu_count() or 8) // 2)


def _large_pages(osname: str) -> tuple[bool, str]:
    if osname == "windows":
        out = _run(["whoami", "/priv"])
        if "SeLockMemoryPrivilege" in out:
            return True, "'Lock pages in memory' right granted"
        return False, ("the 'Lock pages in memory' right is missing (see scripts/grant_large_pages.ps1); "
                       "4 KB pages are used")
    if osname == "linux":
        p = Path("/sys/kernel/mm/transparent_hugepage/enabled")
        mode = p.read_text().strip() if p.exists() else ""
        if "[always]" in mode or "[madvise]" in mode:
            return True, f"transparent huge pages: {mode}"
        return False, "transparent huge pages disabled"
    return False, "not supported on this OS"


# CPU families with AVX-VNNI / AVX512-VNNI when /proc/cpuinfo is not available (Windows): Intel Alder Lake and
# later client parts, Ice Lake and later Xeons, AMD Zen 4 and Zen 5.
_VNNI_NAME = re.compile(r"(i[3579]-1[2-9]\d{3}|Core\(TM\) Ultra|Core Ultra|Xeon.*(Silver|Gold|Platinum) [3-9]\d{3}|"
                        r"Ryzen (\d )?(7|8|9)\d{3}|Ryzen AI|EPYC [9]\d{3}|EPYC 4\d{3})", re.I)


def _cpu(osname: str) -> tuple[str, bool, str]:
    name, flags = platform.processor() or "", ""
    if osname == "linux" and Path("/proc/cpuinfo").exists():
        txt = Path("/proc/cpuinfo").read_text(errors="ignore")
        m = re.search(r"model name\s*:\s*(.*)", txt)
        name = m.group(1).strip() if m else name
        m = re.search(r"flags\s*:\s*(.*)", txt)
        flags = m.group(1) if m else ""
        vnni = "avx_vnni" in flags or "avx512_vnni" in flags
        return name, vnni, "from /proc/cpuinfo flags"
    if osname == "windows":
        out = _run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"])
        name = out.strip() or name
    vnni = bool(_VNNI_NAME.search(name))
    return name, vnni, "guessed from the CPU model name"


def detect(model_dir: str | None = None) -> Host:
    osname = {"Windows": "windows", "Linux": "linux", "Darwin": "darwin"}.get(platform.system(), "linux")
    total, free = _vram()
    # WDDM keeps ~1 GiB of the GPU for itself on Windows (6.0 -> 5.1 GiB observed); Linux leaves almost all
    reserve = 1.0 if osname == "windows" else 0.3
    ram, cores = _ram_cores()
    lp, lp_note = _large_pages(osname)
    cpu_name, vnni, vnni_how = _cpu(osname)
    h = Host(osname, total, max(0.0, free - reserve), ram, cores, lp, lp_note, cpu_name, vnni,
             f"{'with' if vnni else 'without'} VNNI ({vnni_how})")
    if osname == "windows":
        h.notes.append("Windows/WDDM keeps ~1 GiB of VRAM and makes CPU-GPU synchronization dearer; Linux leaves that "
                       "VRAM to the model (more GPU layers) and has transparent huge pages")
    return h


def disk_free_gib(path: str) -> float:
    try:
        return shutil.disk_usage(path).free / GIB
    except Exception:
        return 0.0
