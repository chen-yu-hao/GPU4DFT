import pytest

from gpu4dft_codegen import (
    ERI_CLASSES,
    CudaTuning,
    EriClass,
    candidate_tunings,
    canonical_classes_up_to,
    generate,
    list_outputs,
)
from gpu4dft_codegen.__main__ import main
from gpu4dft_codegen.eri import emit_class_header


def test_outputs_per_backend():
    host = set(list_outputs(["host"]))
    both = set(list_outputs(["host", "cuda"]))
    all_three = set(list_outputs(["host", "cuda", "hip"]))

    assert "include/gpu4dft/generated/eri_ssss.hpp" in host
    assert "include/gpu4dft/generated/eri_dddd.hpp" in host
    assert "include/gpu4dft/generated/eri_general.hpp" in host
    assert "include/gpu4dft/generated/eri_deriv1.hpp" in host
    assert "src/eri_ssss_host.cpp" in host
    assert "src/eri_registry.cpp" in host
    assert "src/eri_ssss_cuda.cu" not in host

    cuda_only = both - host
    assert "src/eri_ssss_cuda.cu" in cuda_only
    assert "src/eri_dddd_cuda.cu" in cuda_only
    assert "src/eri_general_cuda.cu" in cuda_only
    assert all(path.endswith(".cu") for path in cuda_only)

    hip_only = all_three - both
    assert "src/eri_ssss_hip.hip" in hip_only
    assert "src/eri_dddd_hip.hip" in hip_only
    assert "src/eri_general_hip.hip" in hip_only
    assert all(path.endswith(".hip") for path in hip_only)


def test_canonical_spd_classes_and_algorithm_split():
    assert len(ERI_CLASSES) == len(canonical_classes_up_to(2)) == 21
    psss_hdr = emit_class_header(EriClass(1, 0, 0, 0))
    dddd_hdr = emit_class_header(EriClass(2, 2, 2, 2))
    ffff_hdr = emit_class_header(EriClass(3, 3, 3, 3))
    assert "md_hermite_1d" in psss_hdr
    assert "rys_roots_weights" in dddd_hdr
    assert "rys_roots_weights" in ffff_hdr


def test_generation_is_deterministic():
    assert generate(["host", "cuda", "hip"]) == generate(["host", "cuda", "hip"])


def test_registry_lists_every_launcher():
    registry = generate(["host", "cuda", "hip"])["src/eri_registry.cpp"]
    for name in (
        "eri_ssss_host_fp64",
        "eri_ssss_host_fp32",
        "eri_ssss_cuda_fp64",
        "eri_ssss_cuda_fp32",
        "eri_ssss_hip_fp64",
        "eri_dddd_cuda_fp64",
        "eri_dddd_hip_fp64",
    ):
        assert f"&generated::{name}," in registry


def test_cuda_and_hip_tuning_knobs_applied():
    source = generate(["cuda"])["src/eri_ssss_cuda.cu"]
    assert "constexpr int kBlockSize = 128;" in source
    assert "__launch_bounds__(kBlockSize)" in source

    warp_tuning = {
        EriClass(1, 1, 1, 1): CudaTuning(
            block_size=256, thread_mapping="warp", prim_unroll=1, algorithm="rys"
        )
    }
    files = generate(["cuda", "hip"], classes=warp_tuning)
    cuda_warp = files["src/eri_pppp_cuda.cu"]
    hip_warp = files["src/eri_pppp_hip.hip"]
    assert "constexpr int kBlockSize = 256;" in cuda_warp
    assert "__shfl_down_sync" in cuda_warp
    assert "<hip/hip_runtime.h>" in hip_warp
    assert "__shfl_down(val, offset, 32)" in hip_warp

    cands = candidate_tunings(EriClass(1, 0, 1, 0))
    assert any(c.thread_mapping == "warp" and c.algorithm == "md" for c in cands)
    assert any(c.thread_mapping == "thread" and c.algorithm == "rys" for c in cands)


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        generate(["opencl"])


def test_unimplemented_class_rejected():
    with pytest.raises(NotImplementedError):
        emit_class_header(EriClass(6, 0, 0, 0))


def test_cli_writes_listed_files(tmp_path, capsys):
    assert main(["--backends", "host,cuda,hip", "--list-outputs"]) == 0
    listed = capsys.readouterr().out.split(";")
    assert main(["--backends", "host,cuda,hip", "--out-dir", str(tmp_path)]) == 0
    written = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert written == sorted(listed)
