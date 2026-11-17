# GPU4DFT

以 GPU 为先的密度泛函理论程序包。目前是骨架阶段，已完成：

- 用 Python 写的内核生成器，产出 (ss|ss) 双电子积分内核，提供 CPU 参考后端和 CUDA 后端，每个后端都有 fp64 和 fp32 两种精度；
- CMake 构建，包括构建期代码生成和 nanobind Python 扩展；
- 与 PySCF/libcint 的对照测试。

## 目录结构

```
codegen/gpu4dft_codegen/   内核生成器
  config.py                要生成哪些积分类、精度，以及 CUDA 调优参数
  eri.py                   每个积分类的数学，写成后端无关的 GPU4DFT_HD 函数
  backends.py              各后端驱动（host 循环、CUDA kernel + launcher）和内核注册表
include/gpu4dft/           公共头文件：Basis、壳对 SoA、Boys 函数、内核注册表
src/                       C++ 实现；src/cuda/ 是 CUDA 运行时胶水代码
python/                    nanobind 绑定和 Python 包（含 PySCF 接口）
tests/cpp/                 C++ 单元测试（Boys 函数）
tests/python/              生成器测试，以及与 PySCF 的对照测试
```

## 构建与测试

依赖：CMake ≥ 3.24、C++20 编译器、Python ≥ 3.10、nanobind；测试还需要 pytest 和 PySCF。CUDA 是可选的。

```bash
pip install nanobind numpy pytest pyscf
```

```bash
cmake -S . -B build -G Ninja
```

```bash
cmake --build build
```

```bash
ctest --test-dir build --output-on-failure
```

也可以直接运行 pytest：

```bash
PYTHONPATH=build/python python -m pytest tests/python
```

或者以 Python 包的方式安装（通过 scikit-build-core）：

```bash
pip install .
```

| CMake 选项 | 默认值 | 说明 |
|---|---|---|
| `GPU4DFT_ENABLE_CUDA` | ON | 找到 CUDA 编译器时构建 CUDA 后端 |
| `GPU4DFT_REQUIRE_CUDA` | OFF | 找不到 CUDA 时直接报错，不退回纯 CPU 构建 |
| `GPU4DFT_BUILD_PYTHON` | ON | 构建 Python 扩展 |
| `GPU4DFT_BUILD_TESTS` | ON | 注册 CTest 测试 |
| `CMAKE_CUDA_ARCHITECTURES` | `80;90` | 目标 GPU 架构（A100、H100），可以改成 `native` 等 |

## 用法

```python
from pyscf import gto
import gpu4dft

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g")
basis = gpu4dft.basis_from_pyscf(mol)
eri = gpu4dft.eri_tensor(basis, backend="cuda", precision="fp64")  # (nao,)*4
```

`eri_tensor` 会存下完整的 nao⁴ 张量，只用于验证。生产路径将在积分计算过程中直接收缩，不落盘整个张量。

## 设计要点

1. **以生成器为中心。** 每个积分类生成三样东西：
   - 一份共享的数学头文件，host 和 device 用的是同一份代码，所以 CPU 后端验证的就是 GPU 实际执行的算术；
   - 每个后端一个驱动；
   - 一条注册表条目。

   CMake 在配置期向生成器查询输出文件列表，在构建期才真正生成。生成器的源码一改，就会自动重新配置。
2. **壳对 SoA。** 每个原始高斯对预先算好 p、P，以及前因子 c_a c_b Y₀₀² exp(−ab/p·|AB|²)(π/p)^{3/2}，这样内核里每个原始四元组只剩一次 F₀ 和少量乘法。
3. **精度是内核参数。** 每个类都生成 fp64 和 fp32 两种变体。原始积分按所选精度计算，累加始终用 double。这是后续做动态混合精度的基础。
4. **注册表分派。** 运行时按 (la, lb, lc, ld, backend, precision) 查表找内核。没有生成的积分类会抛出 `NotImplementedError`。
5. **基组约定。** 输入系数遵循基组文件的约定，即乘在归一化原始高斯上，与 PySCF 的 `bas_ctr_coeff` 相同。`Basis` 内部会转换成乘在原始 r^l e^{−ar²} 上的系数，并把收缩函数归一化。AO 顺序与 PySCF 的球谐顺序一致。

## 验证状态

- **host 后端：** 在 5 个体系上与 PySCF 对照（H₂/STO-3G、H₄/6-31G、HeH⁺/3-21G、一般收缩基组、随机 H₈ 团簇加 even-tempered 基组）。最大误差相对最大积分约为 1e-14（fp64）和 1e-7（fp32）。
- **CUDA 后端：** 已在没有 GPU 的环境中确认能编译：用 clang 编译到 sm_80/sm_90 的 SASS，没有寄存器溢出（fp64 用 70 个寄存器）。**还没有在真实 GPU 上运行过。** 第一次在 GPU 机器上请先跑 `ctest`，CUDA 测试会自动启用。

## 路线图

1. McMurchie–Davidson / Rys 生成器，支持更高角动量；同时实现 Boys F_m。
2. 直接构建 J/K（不存储 ERI），加上 Schwarz 筛选和密度加权筛选。
3. 自动调优器：搜索 block size、线程映射方式、原始高斯循环展开方式。
4. HIP 后端（AMD MI300 系列）。
5. XC 数值积分和 SCF 驱动；CI。
