cmake_minimum_required(VERSION 3.24)
project(gpu4dft VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(GPU4DFT_ENABLE_CUDA "Build the CUDA backend when a CUDA compiler is found" ON)
option(GPU4DFT_REQUIRE_CUDA "Fail the configuration if the CUDA backend cannot be built" OFF)
option(GPU4DFT_BUILD_PYTHON "Build the Python extension module" ON)
option(GPU4DFT_BUILD_TESTS "Build and register tests" ON)

# The interpreter runs the kernel generator; the development module is only
# needed for the Python extension.
if(GPU4DFT_BUILD_PYTHON)
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
else()
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter)
endif()
find_package(OpenMP COMPONENTS CXX)

# ---------------------------------------------------------------------------
# CUDA backend
# ---------------------------------------------------------------------------
set(GPU4DFT_HAVE_CUDA OFF)
if(GPU4DFT_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      # A100 and H100. Override with -DCMAKE_CUDA_ARCHITECTURES=native etc.
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

# ---------------------------------------------------------------------------
# Kernel generation
# ---------------------------------------------------------------------------
set(GPU4DFT_GEN_DIR ${CMAKE_BINARY_DIR}/generated)
set(GPU4DFT_GEN_BACKENDS host)
if(GPU4DFT_HAVE_CUDA)
  list(APPEND GPU4DFT_GEN_BACKENDS cuda)
endif()
list(JOIN GPU4DFT_GEN_BACKENDS "," _gen_backends)

set(_codegen_cmd
    ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_SOURCE_DIR}/codegen
    ${Python_EXECUTABLE} -m gpu4dft_codegen --backends ${_gen_backends})

# Re-run the configuration whenever the generator changes, so the list of
# generated files below stays in sync with it.
file(GLOB_RECURSE _codegen_sources CONFIGURE_DEPENDS
     ${PROJECT_SOURCE_DIR}/codegen/gpu4dft_codegen/*.py)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_codegen_sources})

execute_process(
  COMMAND ${_codegen_cmd} --list-outputs
  OUTPUT_VARIABLE _gen_relpaths
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
list(TRANSFORM _gen_relpaths PREPEND ${GPU4DFT_GEN_DIR}/ OUTPUT_VARIABLE GPU4DFT_GEN_FILES)

add_custom_command(
  OUTPUT ${GPU4DFT_GEN_FILES}
  COMMAND ${_codegen_cmd} --out-dir ${GPU4DFT_GEN_DIR}
  DEPENDS ${_codegen_sources}
  COMMENT "Generating integral kernels (${_gen_backends})"
  VERBATIM)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(gpu4dft_core STATIC
  src/basis.cpp
  src/shell_pairs.cpp
  src/eri.cpp
  ${GPU4DFT_GEN_FILES})
if(GPU4DFT_HAVE_CUDA)
  target_sources(gpu4dft_core PRIVATE src/cuda/eri_cuda.cu)
  target_link_libraries(gpu4dft_core PUBLIC CUDA::cudart)
  target_compile_definitions(gpu4dft_core PUBLIC GPU4DFT_HAVE_CUDA=1)
endif()
target_include_directories(gpu4dft_core
  PUBLIC ${PROJECT_SOURCE_DIR}/include ${GPU4DFT_GEN_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_options(gpu4dft_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
if(OpenMP_CXX_FOUND)
  target_link_libraries(gpu4dft_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# Python extension
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
  set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${_pkg_dir})
  file(GLOB _py_sources CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/python/gpu4dft/*.py)
  foreach(_src IN LISTS _py_sources)
    cmake_path(GET _src FILENAME _name)
    configure_file(${_src} ${_pkg_dir}/${_name} COPYONLY)
  endforeach()

  if(SKBUILD)
    install(TARGETS _core LIBRARY DESTINATION gpu4dft)
  endif()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(GPU4DFT_BUILD_TESTS)
  enable_testing()

  add_executable(test_boys tests/cpp/test_boys.cpp)
  target_link_libraries(test_boys PRIVATE gpu4dft_core)
  add_test(NAME cpp.boys COMMAND test_boys)

  if(GPU4DFT_BUILD_PYTHON)
    add_test(NAME python
             COMMAND ${Python_EXECUTABLE} -m pytest -q ${PROJECT_SOURCE_DIR}/tests/python)
    set_tests_properties(python PROPERTIES
                         ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python")
  endif()
endif()
