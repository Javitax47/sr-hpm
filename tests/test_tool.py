from pathlib import Path

from srhpm import platform as plat
from types import SimpleNamespace

from srhpm.requant import mix, plan


def test_detect_returns_a_usable_host():
    h = plat.detect()
    assert h.os in ("windows", "linux", "darwin")
    assert h.ram_gib > 0 and h.cores >= 1
    assert h.vram_free_gib >= 0
    assert isinstance(h.large_pages, bool) and h.large_pages_note
    assert isinstance(h.vnni, bool)


def test_vnni_name_heuristic():
    match = plat._VNNI_NAME.search
    assert match("AMD Ryzen 9 7950X 16-Core Processor")
    assert match("13th Gen Intel(R) Core(TM) i7-13700H")
    assert not match("AMD Ryzen 7 5800H with Radeon Graphics")
    assert not match("Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz")


def test_requant_plan_has_all_steps():
    a = SimpleNamespace(out="out", llama_dir=str(Path.home() / "src/llama.cpp"), source="src_dir", imatrix="im.gguf",
                        recipe="speed", calib="calib.txt")
    steps = plan(a, Path("out/types-speed.txt"))
    names = [Path(s[0]).name if not s[0].endswith("python.exe") else Path(s[1]).name for s in steps]
    assert names[0] == "convert_hf_to_gguf.py"  # safetensors source: converted first
    assert any("quantize" in n for n in names)
    assert sum("--kl-divergence" in s for s in steps) == 1
    a.source = "model-Q8_0.gguf"
    assert Path(plan(a, Path("t.txt"))[0][0]).name.startswith("llama-quantize")  # GGUF source: no conversion


def test_kernel_aware_mix():
    tt = {"blk.3.ffn_up.weight": "IQ3_S", "blk.30.ffn_up.weight": "IQ3_S", "blk.30.attn_qkv.weight": "IQ3_XXS",
          "blk.30.ffn_down.weight": "IQ4_XS", "output.weight": "Q5_K"}
    q = mix(tt, "quality", 20)
    assert q["blk.3.ffn_up.weight"] == "IQ3_S"  # GPU layer: unchanged
    assert q["blk.30.ffn_up.weight"] == "IQ4_XS" and q["blk.30.attn_qkv.weight"] == "IQ4_XS"
    s = mix(tt, "speed", 20)
    assert s["blk.30.ffn_up.weight"] == "Q3_K" and s["blk.30.attn_qkv.weight"] == "IQ4_XS"
    assert s["blk.30.ffn_down.weight"] == "IQ4_XS" and s["output.weight"] == "Q5_K"
