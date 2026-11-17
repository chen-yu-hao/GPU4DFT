import pytest

from gpu4dft_codegen import EriClass, generate, list_outputs
from gpu4dft_codegen.__main__ import main
from gpu4dft_codegen.eri import emit_class_header


def test_outputs_per_backend():
    host = set(list_outputs(["host"]))
    both = set(list_outputs(["host", "cuda"]))
    assert "include/gpu4dft/generated/eri_ssss.hpp" in host
    assert "src/eri_ssss_host.cpp" in host
    assert "src/eri_registry.cpp" in host
    assert "src/eri_ssss_cuda.cu" not in host
    assert both - host == {"src/eri_ssss_cuda.cu"}


def test_generation_is_deterministic():
    assert generate(["host", "cuda"]) == generate(["host", "cuda"])


def test_registry_lists_every_launcher():
    registry = generate(["host", "cuda"])["src/eri_registry.cpp"]
    for name in (
        "eri_ssss_host_fp64",
        "eri_ssss_host_fp32",
        "eri_ssss_cuda_fp64",
        "eri_ssss_cuda_fp32",
    ):
        assert f"&generated::{name}," in registry


def test_cuda_tuning_is_applied():
    source = generate(["cuda"])["src/eri_ssss_cuda.cu"]
    assert "constexpr int kBlockSize = 128;" in source
    assert "__launch_bounds__(kBlockSize)" in source


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        generate(["opencl"])


def test_unimplemented_class_rejected():
    with pytest.raises(NotImplementedError):
        emit_class_header(EriClass(1, 0, 0, 0))


def test_cli_writes_listed_files(tmp_path, capsys):
    assert main(["--backends", "host,cuda", "--list-outputs"]) == 0
    listed = capsys.readouterr().out.split(";")
    assert main(["--backends", "host,cuda", "--out-dir", str(tmp_path)]) == 0
    written = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert written == sorted(listed)
