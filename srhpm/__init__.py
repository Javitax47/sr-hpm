"""SR-HPM (Speculative verification on RAM-resident weights for Heterogeneous PC Memory): fast decoding of dense LLMs
larger than the GPU's memory (llama.cpp patch + launcher)."""
import os
import sys
from pathlib import Path


def _import_gguf():
    """The gguf package from pip, or the gguf-py copy inside the SR-HPM llama.cpp checkout."""
    try:
        import gguf  # noqa: F401
        return gguf
    except ImportError:
        for d in (os.environ.get("SRHPM_LLAMA_DIR"), Path(__file__).resolve().parents[1] / "llama.cpp-srhpm",
                  Path.home() / "src/llama.cpp"):
            if d and (Path(d) / "gguf-py").exists():
                sys.path.insert(0, str(Path(d) / "gguf-py"))
                import gguf
                return gguf
        raise


def gguf_reader(path: str):
    return _import_gguf().GGUFReader(path)
