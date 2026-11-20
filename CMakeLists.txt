cmake_minimum_required(VERSION 3.24)
project(gpu4dft VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(GPU4DFT_ENABLE_CUDA "Build the CUDA backend when a CUDA compiler is found" ON)
option(GPU4DFT_REQUIRE_CUDA "Fail the configuration if the CUDA backend cannot be built" OFF)
option(GPU4DFT_ENABLE_HIP "Build the ROCm HIP backend when hipcc is found" OFF)
option(GPU4DFT_BUILD_PYTHON "Build the Python extension module" ON)
option(GPU4DFT_BUILD_TESTS "Build and register tests" ON)

# The interpreter runs the kernel generators; the development module is only
# needed for the Python extension.
if(GPU4DFT_BUILD_PYTHON)
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
else()
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter)
endif()
find_package(OpenMP COMPONENTS CXX)

# ---------------------------------------------------------------------------
# Libxc (5.2.3+ C API: supports Config static Libxc::xc, PkgConfig, or manual)
# ---------------------------------------------------------------------------
find_package(Libxc QUIET)
if(TARGET Libxc::xc)
  set(_gpu4dft_libxc_target Libxc::xc)
  message(STATUS "Found Libxc via CMake Config: Libxc::xc")
else()
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(LIBXC QUIET IMPORTED_TARGET libxc)
  endif()
  if(TARGET PkgConfig::LIBXC)
    set(_gpu4dft_libxc_target PkgConfig::LIBXC)
    message(STATUS "Found Libxc via PkgConfig: PkgConfig::LIBXC (${LIBXC_VERSION})")
  else()
    find_path(LIBXC_INCLUDE_DIR NAMES xc.h)
    find_library(LIBXC_LIBRARY NAMES xc)
    if(LIBXC_INCLUDE_DIR AND LIBXC_LIBRARY)
      add_library(Libxc::xc UNKNOWN IMPORTED)
      set_target_properties(Libxc::xc PROPERTIES
        IMPORTED_LOCATION "${LIBXC_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${LIBXC_INCLUDE_DIR}")
      set(_gpu4dft_libxc_target Libxc::xc)
      message(STATUS "Found Libxc via find_path/find_library: ${LIBXC_LIBRARY}")
    else()
      message(FATAL_ERROR
        "Libxc not found. Install libxc-dev (e.g. `apt-get install -y libxc-dev`) "
        "or set Libxc_DIR / PKG_CONFIG_PATH.")
    endif()
  endif()
endif()

# ---------------------------------------------------------------------------
# CUDA & Optional ROCm HIP backends
# ---------------------------------------------------------------------------
set(GPU4DFT_HAVE_CUDA OFF)
if(GPU4DFT_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      # A100 and H100 by default. Override with -DCMAKE_CUDA_ARCHITECTURES=native (e.g. Blackwell sm_120).
      set(CMAKE_CUDA_ARCHITECTURES 80 90)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    set(GPU4DFT_HAVE_CUDA ON)
  elseif(GPU4DFT_REQUIRE_CUDA)
    message(FATAL_ERROR "GPU4DFT_REQUIRE_CUDA is ON but no CUDA compiler was found")
  endif()
endif()
message(STATUS "GPU4DFT CUDA backend: ${GPU4DFT_HAVE_CUDA}")

set(GPU4DFT_HAVE_HIP OFF)
if(GPU4DFT_ENABLE_HIP)
  include(CheckLanguage)
  check_language(HIP)
  if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    find_package(hip QUIET)
    set(GPU4DFT_HAVE_HIP ON)
  endif()
endif()
message(STATUS "GPU4DFT HIP backend: ${GPU4DFT_HAVE_HIP}")

# ---------------------------------------------------------------------------
# Kernel generation (Integral Codegen + Symbolic XC Codegen)
# ---------------------------------------------------------------------------
set(GPU4DFT_GEN_DIR ${CMAKE_BINARY_DIR}/generated)
set(GPU4DFT_GEN_BACKENDS host)
if(GPU4DFT_HAVE_CUDA)
  list(APPEND GPU4DFT_GEN_BACKENDS cuda)
endif()
if(GPU4DFT_HAVE_HIP)
  list(APPEND GPU4DFT_GEN_BACKENDS hip)
endif()
list(JOIN GPU4DFT_GEN_BACKENDS "," _gen_backends)

# 1. Integral & Derivative Code Generator (`gpu4dft_codegen`)
set(_eri_codegen_cmd
    ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_SOURCE_DIR}/codegen
    ${Python_EXECUTABLE} -m gpu4dft_codegen --backends ${_gen_backends})

file(GLOB_RECURSE _eri_codegen_sources CONFIGURE_DEPENDS
     ${PROJECT_SOURCE_DIR}/codegen/gpu4dft_codegen/*.py)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_eri_codegen_sources})

execute_process(
  COMMAND ${_eri_codegen_cmd} --list-outputs
  OUTPUT_VARIABLE _eri_gen_relpaths
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
list(TRANSFORM _eri_gen_relpaths PREPEND ${GPU4DFT_GEN_DIR}/ OUTPUT_VARIABLE GPU4DFT_ERI_GEN_FILES)

add_custom_command(
  OUTPUT ${GPU4DFT_ERI_GEN_FILES}
  COMMAND ${_eri_codegen_cmd} --out-dir ${GPU4DFT_GEN_DIR}
  DEPENDS ${_eri_codegen_sources}
  COMMENT "Generating integral & derivative kernels (${_gen_backends})"
  VERBATIM)

if(GPU4DFT_HAVE_HIP)
  foreach(_gen_file IN LISTS GPU4DFT_ERI_GEN_FILES)
    if(_gen_file MATCHES "\\.hip$")
      set_source_files_properties(${_gen_file} PROPERTIES LANGUAGE HIP)
    endif()
  endforeach()
endif()

# 2. Symbolic Autodiff XC Functional Code Generator (`xc_codegen`)
set(_xc_codegen_cmd
    ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_SOURCE_DIR}/codegen
    ${Python_EXECUTABLE} -m xc_codegen --backends ${_gen_backends})

file(GLOB_RECURSE _xc_codegen_sources CONFIGURE_DEPENDS
     ${PROJECT_SOURCE_DIR}/codegen/xc_codegen/*.py)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_xc_codegen_sources})

execute_process(
  COMMAND ${_xc_codegen_cmd} --list-outputs
  OUTPUT_VARIABLE _xc_gen_relpaths
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
list(TRANSFORM _xc_gen_relpaths PREPEND ${GPU4DFT_GEN_DIR}/ OUTPUT_VARIABLE GPU4DFT_XC_GEN_FILES)

add_custom_command(
  OUTPUT ${GPU4DFT_XC_GEN_FILES}
  COMMAND ${_xc_codegen_cmd} --out-dir ${GPU4DFT_GEN_DIR}
  DEPENDS ${_xc_codegen_sources}
  COMMENT "Generating symbolic XC functional kernels (${_gen_backends})"
  VERBATIM)

# ---------------------------------------------------------------------------
# Core shared library (`libgpu4dft_core.so`)
# ---------------------------------------------------------------------------
add_library(gpu4dft_core SHARED
  src/basis.cpp
  src/shell_pairs.cpp
  src/eri.cpp
  src/one_electron.cpp
  src/jk_engine.cpp
  src/xc_grid.cpp
  src/xc_engine.cpp
  src/dftd3.cpp
  src/dftd3_tables.cpp
  src/snk.cpp
  src/linalg.cpp
  src/diis.cpp
  src/multigpu.cpp
  src/scf.cpp
  src/workflow.cpp
  src/response.cpp
  src/pw.cpp
  ${GPU4DFT_ERI_GEN_FILES}
  ${GPU4DFT_XC_GEN_FILES})

if(GPU4DFT_HAVE_CUDA)
  target_sources(gpu4dft_core PRIVATE
    src/cuda/eri_cuda.cu
    src/cuda/one_electron_cuda.cu
    src/cuda/jk_cuda.cu
    src/cuda/xc_cuda.cu
    src/cuda/snk_cuda.cu
    src/cuda/device_pool.cu
    src/cuda/linalg_cuda.cu
    src/cuda/diis_cuda.cu
    src/cuda/multigpu_cuda.cu
    src/cuda/response_cuda.cu
    src/cuda/pw_cuda.cu)
  target_link_libraries(gpu4dft_core PUBLIC CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::cufft)
  target_compile_definitions(gpu4dft_core PUBLIC GPU4DFT_HAVE_CUDA=1)
endif()

if(GPU4DFT_HAVE_HIP)
  if(TARGET hip::host)
    target_link_libraries(gpu4dft_core PUBLIC hip::host)
  endif()
  target_compile_definitions(gpu4dft_core PUBLIC GPU4DFT_HAVE_HIP=1)
endif()

target_include_directories(gpu4dft_core
  PUBLIC ${PROJECT_SOURCE_DIR}/include ${GPU4DFT_GEN_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_options(gpu4dft_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
if(OpenMP_CXX_FOUND)
  target_link_libraries(gpu4dft_core PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(gpu4dft_core PRIVATE ${_gpu4dft_libxc_target})

if(APPLE)
  set(_gpu4dft_rpath_origin "@loader_path")
else()
  set(_gpu4dft_rpath_origin "$ORIGIN")
endif()
set_target_properties(gpu4dft_core PROPERTIES
  INSTALL_RPATH "${_gpu4dft_rpath_origin}")

# ---------------------------------------------------------------------------
# Python extension (`gpu4dft._core`)
# ---------------------------------------------------------------------------
if(GPU4DFT_BUILD_PYTHON)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(_core NB_STATIC python/bindings.cpp)
  target_link_libraries(_core PRIVATE gpu4dft_core)

  # Lay out an importable package in the build tree: build/python/gpu4dft.
  set(_pkg_dir ${CMAKE_BINARY_DIR}/python/gpu4dft)
  set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${_pkg_dir}
    BUILD_RPATH "$<TARGET_FILE_DIR:gpu4dft_core>;${_gpu4dft_rpath_origin}"
    INSTALL_RPATH "${_gpu4dft_rpath_origin}")

  file(GLOB _py_sources CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/python/gpu4dft/*.py)
  foreach(_src IN LISTS _py_sources)
    cmake_path(GET _src FILENAME _name)
    configure_file(${_src} ${_pkg_dir}/${_name} COPYONLY)
  endforeach()

  if(SKBUILD)
    install(TARGETS gpu4dft_core _core
            LIBRARY DESTINATION gpu4dft
            RUNTIME DESTINATION gpu4dft)
  endif()
endif()

# ---------------------------------------------------------------------------
# C++ & Python Tests (`ctest`)
# ---------------------------------------------------------------------------
if(GPU4DFT_BUILD_TESTS)
  enable_testing()

  add_executable(test_boys tests/cpp/test_boys.cpp)
  target_link_libraries(test_boys PRIVATE gpu4dft_core)
  set_target_properties(test_boys PROPERTIES BUILD_RPATH "$<TARGET_FILE_DIR:gpu4dft_core>")
  add_test(NAME cpp.boys COMMAND test_boys)

  add_executable(test_rys tests/cpp/test_rys.cpp)
  target_link_libraries(test_rys PRIVATE gpu4dft_core)
  set_target_properties(test_rys PROPERTIES BUILD_RPATH "$<TARGET_FILE_DIR:gpu4dft_core>")
  add_test(NAME cpp.rys COMMAND test_rys)

  add_executable(test_xc_libxc tests/cpp/test_xc_libxc.cpp)
  target_link_libraries(test_xc_libxc PRIVATE gpu4dft_core)
  set_target_properties(test_xc_libxc PROPERTIES BUILD_RPATH "$<TARGET_FILE_DIR:gpu4dft_core>")
  add_test(NAME cpp.xc_libxc COMMAND test_xc_libxc)

  add_executable(test_dftd3 tests/cpp/test_dftd3.cpp)
  target_link_libraries(test_dftd3 PRIVATE gpu4dft_core)
  set_target_properties(test_dftd3 PROPERTIES BUILD_RPATH "$<TARGET_FILE_DIR:gpu4dft_core>")
  add_test(NAME cpp.dftd3 COMMAND test_dftd3)

  add_executable(test_uhf_uks tests/cpp/test_uhf_uks.cpp)
  target_link_libraries(test_uhf_uks PRIVATE gpu4dft_core)
  set_target_properties(test_uhf_uks PROPERTIES BUILD_RPATH "$<TARGET_FILE_DIR:gpu4dft_core>")
  add_test(NAME cpp.uhf_uks COMMAND test_uhf_uks)

  add_executable(test_rsh tests/cpp/test_rsh.cpp)
  target_link_libraries(test_rsh PRIVATE gpu4dft_core)
  set_target_properties(test_rsh PROPERTIES BUILD_RPATH "$<TARGET_FILE_DIR:gpu4dft_core>")
  add_test(NAME cpp.rsh COMMAND test_rsh)

  add_executable(test_ri_jk_grad tests/cpp/test_ri_jk_grad.cpp)
  target_link_libraries(test_ri_jk_grad PRIVATE gpu4dft_core)
  set_target_properties(test_ri_jk_grad PROPERTIES BUILD_RPATH "$<TARGET_FILE_DIR:gpu4dft_core>")
  add_test(NAME cpp.ri_jk_grad COMMAND test_ri_jk_grad)

  if(GPU4DFT_BUILD_PYTHON)
    add_test(NAME python
             COMMAND ${Python_EXECUTABLE} -m pytest -q ${PROJECT_SOURCE_DIR}/tests/python)
    set_tests_properties(python PROPERTIES
      ENVIRONMENT
        "PYTHONPATH=${CMAKE_BINARY_DIR}/python;GPU4DFT_TEST_LIB=$<TARGET_FILE:gpu4dft_core>;GPU4DFT_DRIVER_LIB=$<TARGET_FILE:gpu4dft_core>;GPU4DFT_XC_LIB=$<TARGET_FILE:gpu4dft_core>;GPU4DFT_RESP_LIB=$<TARGET_FILE:gpu4dft_core>;GPU4DFT_PW_LIB=$<TARGET_FILE:gpu4dft_core>")
  endif()
endif()
