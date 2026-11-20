# GPU4DFT Stage 3 开发进度与验证报告

本文档记录 GPU4DFT 第三阶段（Stage 3：科学公平性能基准、大体系占优轨道 `RI-K`/`RI-JK` 解析核梯度、`libxc` 泛函集成、开壳层 `UHF/UKS`、范围分离杂化泛函 `CAM-B3LYP`/`wB97X`、`DFT-D3(BJ)` 色散校正及 S22/BH76 几何构型优化评测）的当前开发进度、已完成实卡验证数据及后续断点续跑指令。

---

## 1. 里程碑完成矩阵

| 里程碑 | 核心模块与功能范围 | 状态 | 实卡构建与验证摘要 |
| :--- | :--- | :---: | :--- |
| **`E2E`** | 黑盒端到端验收测试集（`tests/e2e/`，Tiers 1–4，含 S22 全部 22 复合物与 BH76 开壳层反应集加载器） | ✅ **已完成** | **98 项** 端到端测试脚本与参考真值预言机（`oracles.py`）就绪 |
| **`M1`** | CMake 链接 `Libxc::xc`（5.2.3 兼容）、自旋极化 `XcEngine`、求积格点权重核导数（`grid_response`）、Grimme `DFT-D3(BJ)` 能量与解析核梯度 | ✅ **已验收** | Colab `A100` 通过 **`415/415`** 项 CTest/`pytest`；平移不变性 $|\sum_A \mathbf{g}_A| \le 10^{-6}\ E_\text{h}/a_0$，`D3(BJ)` 能量误差 $\le 10^{-8}\ E_\text{h}$、梯度误差 $\le 10^{-7}\ E_\text{h}/a_0$ |
| **`M2`** | 开壳层 `UHF/UKS` 自洽场、联合 $\alpha/\beta$ GPU CDIIS、SP2 轨道对角化、`TPSS` $\tau \ge \tau_W$ 正定保护、$\langle S^2\rangle$ 与直接四中心解析核梯度 | ✅ **已验收** | Colab `T4`/`L4` 通过 `5/5` 项 C++ 测试、`test_uhf_uks.py` **`103` 项全过**、`411/411` 项回归测试与 `5/5` 项 E2E 开壳层验收 |
| **`M3`** | 范围分离杂化泛函 `CAM-B3LYP` / `wB97X`（$\omega$ 衰减 $\text{erf}(\omega r_{12})/r_{12}$ Boys/Rys 积分核生成、Direct-JK、RI-JK 2c/3c 度量及闭/开壳层解析核梯度） | ✅ **已验收** | Colab 通过 `6/6` 项 C++ 测试、**`528` 项 Python 测试**与 288+ 项边界探测（相对 `PySCF` $|\Delta E| \le 10^{-6}\ E_\text{h}$、$\max|\Delta\mathbf{g}| \le 10^{-5}\ E_\text{h}/a_0$） |
| **`M4`** | 显存受控占优轨道空间 $O(N_{\text{aux}} N_{\text{AO}}^2 N_{\text{occ}})$ `RI-K` 收缩、`RI-JK` 解析核梯度（`auxbasis_response=True`，含不定号密度矩阵修正）、`DeviceMemoryPool::trim()` 显存回收与 `Valinomycin`/`(H2O)_80` 低内存分块 | ✅ **已验收** | CUDA 零警告构建（`104/104` targets）、**`7/7` 项 C++ 测试**与 **`557` 项 Python 测试全过**；15 GiB `T4` 上 `(H2O)_80/def2-svp`（$N_{\text{AO}}=1920$）梯度主机 RSS 仅 **`1.265 GiB`**，`Valinomycin/6-31g*` `OzakiInt8` 稳定通过 |
| **`M5`** | 科学公平 5 次重复基准脚本（彻底移除 `shared_grids`、三方对称预热、峰值显存监控）与 `Taxol` / `Valinomycin` / `C60` / `(H2O)_32` 大体系评测 | 🔄 **代码就绪 / 跑数待续** | 代码与 `test_benchmarks.py`（`5/5`）已完成；已在 `T4`/`L4` 完成 `12/13` 组 `RI-K` 能量对标与 Suite 4 前 **21 组** 5 次重复统计数据（至 `(H2O)_32`），待恢复 GPU 配额续跑 `C60`/`Taxol`/`Valinomycin` 与 Suite 5 |
| **`M6`** | PySCF 兼容几何优化扫描器（`SCFScanner` / `GradScanner`）、ASE `GPU4DFTCalculator` 热启动显存复用及 S22 / BH76 几何优化评测 | 🔄 **`M6.1` 已完成 / 跑数待续** | `M6.1` 优化器接口全部完成并通过 `test_scanner_opt.py` **`38/38` 项测试**（CUDA `104/104` + CPU `71/71` 构建）；待恢复 GPU 配额执行 `M6.2–M6.4`（BH76 与 S22 批量单点及几何优化） |
| **`M7`** | 全量 E2E Tiers 1–4 验收、Tier-5 对抗强化与最终全套 CTest 门禁 | ⏳ **待启动** | 依赖 `M5` 与 `M6` GPU 跑数收尾后执行 |

---

## 2. 已完成的实卡测试与阶段性基准数据

### 2.1 大体系显存控制与 `RI-JK` 解析核梯度实测（`M4`，Tesla T4 15 GiB VRAM / 12.7 GiB RAM）

- **`DeviceMemoryPool::trim()` 显存即时回收**（`(H2O)_32 / def2-svp`，$N_{\text{AO}}=768$）：
  - 初始空闲显存 `13.750 GiB` $\to$ `RI-JK` 峰值工作区后剩余 `3.582 GiB` $\to$ 调用 `release_workspaces()` / `trim()` 后立即恢复至 **`13.582 GiB`**，`del` 引擎后 100% 恢复至 **`13.750 GiB`**。
- **`Valinomycin / 6-31g*`（168 原子，$N_{\text{AO}}=1272$，`def2-universal-jkfit`）**：
  - 纯库仑 `RI-J` 工作区仅占 **`1.152 GiB`**（跳过冗余交换缓冲）；
  - `RHF` + `OzakiInt8` 密度拟合 SCF 在 15 GiB `T4` 上零稳态分配（`steady_state_allocations == 0`）稳定运行，主机峰值 RSS 仅 **`0.760 GiB`**。
- **`(H2O)_80 / def2-svp` 超大水簇 `RI-JK` 解析核梯度（240 原子，$N_{\text{AO}}=1920, N_{\text{aux}}=9040, N_{\text{occ}}=400$）**：
  - 采用基于可用显存自适应的占优轨道 `i` 分批收缩替代单体主机缓冲，在 15 GiB `T4` 上完整算完解析核梯度：
  - 峰值显存 **`11.411 GiB`**，主机峰值 RSS 仅 **`1.265 GiB`**，净平动合力误差（平移不变性）**$2.283 \times 10^{-11}\ E_\text{h}/a_0$**。
- **不定号密度矩阵（Indefinite DM，如增量 Fock / 响应差分密度）`RI-JK` 梯度校验**：
  - 对照中心有限差分，Host 与 CUDA 后端最大绝对梯度偏差为 **`8.49e-9`** $E_\text{h}/a_0$（HF，$\omega=0$）与 **`2.32e-7`** $E_\text{h}/a_0$（RSH，$\omega=0.33$）。

### 2.2 占优轨道空间 `RI-K` 匹配格点能量精度对标（`M5`，对照 PySCF DF `FP64`）

| 分子体系 | 方法 / 泛函 | $N_{\text{AO}}$ | $|\Delta E|$ ($E_\text{h}$) | $|\Delta\langle S^2\rangle|$ | 门禁阈值 |
| :--- | :--- | ---: | :--- | :--- | :---: |
| `H2O` | `RHF / HF` | 24 | `2.88e-12` | — | $\le 10^{-6}$ ✅ |
| `C6H6` | `RHF / HF` | 114 | `1.54e-11` | — | $\le 10^{-6}$ ✅ |
| `H2O` | `RKS / B3LYP` | 24 | `1.28e-13` | — | $\le 10^{-6}$ ✅ |
| `H2O` | `RKS / PBE0` | 24 | `2.84e-13` | — | $\le 10^{-6}$ ✅ |
| `CH2O` | `RKS / CAM-B3LYP` | 38 | `4.93e-09` | — | $\le 10^{-6}$ ✅ |
| `CH2O` | `RKS / wB97X` | 38 | `8.73e-09` | — | $\le 10^{-6}$ ✅ |
| `HO2` | `UHF / HF` | 33 | `7.53e-12` | `1.35e-07` | $\le 10^{-6}$ ✅ |
| `O2` | `UHF / HF` | 28 | `4.15e-12` | `8.29e-08` | $\le 10^{-6}$ ✅ |
| `HO2` | `UKS / B3LYP` | 33 | `7.73e-12` | `8.06e-08` | $\le 10^{-6}$ ✅ |
| `HO2` | `UKS / CAM-B3LYP` | 33 | `6.49e-09` | `1.14e-07` | $\le 10^{-6}$ ✅ |
| `HO2` | `UKS / wB97X` | 33 | `1.64e-08` | `4.24e-10` | $\le 10^{-6}$ ✅ |
| `(H2O)_32` | `RKS / B3LYP` | 768 | `8.28e-11` | — | $\le 10^{-6}$ ✅ |

### 2.3 科学公平协议下 `NVIDIA L4` 阶段性 5 次重复统计数据（`M5` Suite 4，独立构网格 + 三方对称预热）

> 注：`NVIDIA L4`（Ada `sm_89`）原生硬件双精度吞吐仅约 `0.47 TFLOPS`（1:64 FP64:FP32 比），因此以下 `L4` 阶段性数据主要用于验证零 `shared_grids` 公平评测流水线与极低的 5 次重复标准差（`0.003–0.094s`）；面向大体系的正式双精度吞吐对标建议在恢复 `A100` 或 Blackwell `G4` 配额后完整采集。

| 体系 / 基组 (`B3LYP`, `grids_level=3`) | $N_{\text{AO}}$ | CPU PySCF Direct (s) | CPU PySCF DF (s) | GPU4PySCF Direct (s) | GPU4PySCF DF (s) | GPU4DFT `Direct_FP64` (s) | GPU4DFT `RI_JK_FP64` (s) | $|\Delta E|$ ($E_\text{h}$, 独立网格) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `(H2O)_2 / 6-31g*` | 36 | `1.805 ± 0.018` | `2.160 ± 0.430` | `3.138 ± 0.063` | `0.425 ± 0.006` | **`0.505 ± 0.005`** (`6.22x` vs G4P Direct) | `2.338 ± 0.013` | `1.35e-06` |
| `C6H6 / def2-svp` | 114 | `8.099 ± 0.332` | `9.183 ± 0.265` | `5.706 ± 0.078` | `0.888 ± 0.005` | **`2.564 ± 0.026`** (`2.23x` vs G4P Direct) | `7.464 ± 0.014` | `3.14e-06` |
| `(H2O)_8 / def2-svp` | 192 | `18.541 ± 0.454` | `19.038 ± 0.465` | `8.647 ± 0.013` | `1.872 ± 0.005` | **`5.940 ± 0.024`** (`1.46x` vs G4P Direct) | `10.598 ± 0.009` | `7.01e-06` |
| `aspirin / 6-31g*` | 198 | `27.018 ± 0.264` | `28.000 ± 0.214` | `19.398 ± 0.007` | `2.820 ± 0.006` | `21.434 ± 0.094` (`Mixed`: `21.115 ± 0.019`) | `28.087 ± 0.007` (`Ozaki`: `26.656 ± 0.005`) | `2.84e-05` |
| `(H2O)_16 / def2-svp` | 384 | — | `79.942 ± 1.193` | `22.048 ± 0.007` | `9.527 ± 0.006` | `29.117 ± 0.028` (`Mixed`: `27.710 ± 0.013`) | `30.151 ± 0.010` (`2.65x` vs CPU DF) | `8.05e-05` |
| `(H2O)_32 / def2-svp` | 768 | — | — | — | `85.885 ± 0.057` | — | `172.17` (5-repeat mean) | — |

---

## 3. 新增核心文件清单（Stage 3）

- **泛函、`DFT-D3(BJ)`、开壳层与范围分离杂化泛函（`M1`–`M3`）**：
  - `CMakeLists.txt`（链接 `Libxc::xc` 并注册 `cpp.xc_libxc`、`cpp.dftd3`、`cpp.uhf_uks`、`cpp.rsh`、`cpp.ri_jk_grad`）
  - `include/gpu4dft/dftd3.hpp`、`src/dftd3.cpp`、`src/dftd3_tables.{hpp,cpp}`、`python/gpu4dft/dftd3.py`
  - `include/gpu4dft/xc.hpp`、`src/xc_engine.cpp`、`src/xc_grid.cpp`、`src/cuda/xc_cuda.cu`
  - `codegen/gpu4dft_codegen/{eri,deriv,backends}.py`、`include/gpu4dft/{boys,rys,eri,shell_pairs}.hpp`、`src/{eri,shell_pairs}.cpp`、`src/cuda/eri_cuda.{hpp,cu}`
  - `include/gpu4dft/{diis,linalg,scf,workflow}.hpp`、`src/{diis,linalg,scf,workflow}.cpp`、`src/cuda/linalg_cuda.cu`
- **占优轨道空间 `RI-K`、`RI-JK` 解析核梯度与显存池回收（`M4`）**：
  - `include/gpu4dft/jk_engine.hpp`、`src/jk_engine.cpp`、`src/cuda/jk_cuda.cu`、`src/cuda/device_pool.cu`
- **公平性能基准、几何构型优化器与测试套件（`M5`、`M6.1`、`E2E`）**：
  - `python/bindings.cpp`、`python/gpu4dft/{__init__,scf,ase_calculator,pyscf_interface}.py`
  - `benchmarks/run_benchmarks.py`、`benchmarks/verify_rik_scaling_and_grad.py`、`benchmarks/geometries/{taxol,valinomycin}.xyz`
  - C++ 单元测试：`tests/cpp/test_{xc_libxc,dftd3,uhf_uks,rsh,ri_jk_grad}.cpp`
  - Python 单元与集成测试：`tests/python/test_{xc_libxc_uks,dftd3,uhf_uks,rsh,ri_jk_grad,benchmarks,scanner_opt}.py`
  - E2E 验收测试与 S22/BH76 数据加载器：`tests/e2e/`（18 个测试/支撑模块 + `tests/e2e/data/`）

---

## 4. 恢复 GPU 配额后的断点续跑步骤

当配备 `A100` / Blackwell `G4` / `L4` 的 GPU 环境就绪并安装 `libxc-dev`、`nanobind`、`pyscf`、`gpu4pyscf`、`ase`、`geometric` 后，按顺序执行：

```bash
# 1. 编译全部 104 个 CUDA + C++20 + nanobind 目标并运行 C++ 单元测试
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native -DGPU4DFT_REQUIRE_CUDA=ON
cmake --build build -j
ctest --test-dir build --output-on-failure -R "^cpp\."

# 2. 完成 M5：占优轨道 RI-K 标度/梯度验证与大体系 5 次重复统计对标
PYTHONPATH=$PWD/build/python:$PWD/python python3 -u benchmarks/verify_rik_scaling_and_grad.py
PYTHONPATH=$PWD/build/python:$PWD/python python3 -u benchmarks/run_benchmarks.py --preset standard --suites 4,5 --repeats 5

# 3. 完成 M6.2–M6.4（S22 22 复合物与 BH76 开壳层反应集单点及几何优化）与 M7 全量 E2E 终验
PYTHONPATH=$PWD/build/python:$PWD/python pytest -v -c tests/e2e/pytest.ini tests/e2e/
```
