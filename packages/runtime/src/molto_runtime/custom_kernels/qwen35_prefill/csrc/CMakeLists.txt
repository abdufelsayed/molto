cmake_minimum_required(VERSION 3.27)

project(molto_qwen35_prefill_kernels LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

find_package(
  Python 3.8
  COMPONENTS Interpreter Development.Module
  REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)
list(GET MLX_INCLUDE_DIRS 0 MOLTO_QWEN35_PREFILL_MLX_INCLUDE_DIR)

set(MOLTO_CUSTOM_KERNEL_COMMON_DIR
    ${CMAKE_CURRENT_LIST_DIR}/../../common/csrc)

add_library(molto_qwen35_prefill_kernel_ops)
target_sources(
  molto_qwen35_prefill_kernel_ops
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/qwen35_prefill.cpp
  ${CMAKE_CURRENT_LIST_DIR}/qwen35_oq_a8.cpp
  ${CMAKE_CURRENT_LIST_DIR}/qwen35_ane.mm)
target_include_directories(
  molto_qwen35_prefill_kernel_ops
  PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${MOLTO_CUSTOM_KERNEL_COMMON_DIR})
target_compile_definitions(
  molto_qwen35_prefill_kernel_ops
  PUBLIC MOLTO_QWEN35_PREFILL_KERNEL_SOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}"
         MOLTO_QWEN35_PREFILL_MLX_INCLUDE_DIR="${MOLTO_QWEN35_PREFILL_MLX_INCLUDE_DIR}")
target_link_libraries(
  molto_qwen35_prefill_kernel_ops
  PUBLIC mlx
  PRIVATE "-framework Accelerate" "-framework Foundation"
          "-framework IOSurface" "-framework Metal")

if(MLX_BUILD_METAL)
  set(MOLTO_QWEN35_PREFILL_METAL_SOURCES
      ${CMAKE_CURRENT_LIST_DIR}/qwen35_attention.metal
      ${CMAKE_CURRENT_LIST_DIR}/qwen35_qmm.metal
      ${CMAKE_CURRENT_LIST_DIR}/qwen35_oq_a8.metal
      ${CMAKE_CURRENT_LIST_DIR}/qwen35_ane.metal)
  file(GLOB_RECURSE MOLTO_QWEN35_PREFILL_METAL_HEADERS CONFIGURE_DEPENDS
       ${CMAKE_CURRENT_LIST_DIR}/*.h
       ${MOLTO_CUSTOM_KERNEL_COMMON_DIR}/kernels/*.h
       ${MOLTO_CUSTOM_KERNEL_COMMON_DIR}/mlx/**/*.h)
  set(MOLTO_QWEN35_PREFILL_METAL_FLAGS
      -x
      metal
      -Wall
      -Wextra
      -fno-fast-math
      -Wno-c++17-extensions
      -Wno-c++20-extensions)
  if(MLX_METAL_DEBUG)
    set(MOLTO_QWEN35_PREFILL_METAL_FLAGS
        ${MOLTO_QWEN35_PREFILL_METAL_FLAGS} -gline-tables-only -frecord-sources)
  endif()
  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
    set(MOLTO_QWEN35_PREFILL_METAL_FLAGS
        ${MOLTO_QWEN35_PREFILL_METAL_FLAGS}
        "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
  endif()
  set(MOLTO_QWEN35_PREFILL_METAL_INCLUDE_DIRS
      ${CMAKE_CURRENT_LIST_DIR}
      ${MOLTO_CUSTOM_KERNEL_COMMON_DIR}
      ${MLX_INCLUDE_DIRS})
  set(MOLTO_QWEN35_PREFILL_METAL_INCLUDE_ARGS)
  foreach(INCLUDE_DIR IN LISTS MOLTO_QWEN35_PREFILL_METAL_INCLUDE_DIRS)
    list(APPEND MOLTO_QWEN35_PREFILL_METAL_INCLUDE_ARGS -I${INCLUDE_DIR})
  endforeach()
  set(MOLTO_QWEN35_PREFILL_METAL_AIR)
  foreach(SRCFILE IN LISTS MOLTO_QWEN35_PREFILL_METAL_SOURCES)
    get_filename_component(TARGET_NAME ${SRCFILE} NAME_WE)
    set(AIR_FILE ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.air)
    add_custom_command(
      OUTPUT ${AIR_FILE}
      COMMAND xcrun -sdk macosx metal ${MOLTO_QWEN35_PREFILL_METAL_FLAGS} -c
              ${SRCFILE} ${MOLTO_QWEN35_PREFILL_METAL_INCLUDE_ARGS} -o
              ${AIR_FILE}
      DEPENDS ${SRCFILE} ${MOLTO_QWEN35_PREFILL_METAL_HEADERS}
      COMMENT "Building ${TARGET_NAME}.air"
      VERBATIM)
    list(APPEND MOLTO_QWEN35_PREFILL_METAL_AIR ${AIR_FILE})
  endforeach()

  set(MOLTO_QWEN35_PREFILL_METALLIB
      ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/molto_qwen35_prefill_kernels.metallib)
  add_custom_command(
    OUTPUT ${MOLTO_QWEN35_PREFILL_METALLIB}
    COMMAND xcrun -sdk macosx metal ${MOLTO_QWEN35_PREFILL_METAL_AIR} -o
            ${MOLTO_QWEN35_PREFILL_METALLIB}
    DEPENDS ${MOLTO_QWEN35_PREFILL_METAL_AIR}
    COMMENT "Building molto_qwen35_prefill_kernels.metallib"
    VERBATIM)
  add_custom_target(molto_qwen35_prefill_kernels_metallib
                    DEPENDS ${MOLTO_QWEN35_PREFILL_METALLIB})

  add_dependencies(molto_qwen35_prefill_kernel_ops
                   molto_qwen35_prefill_kernels_metallib)

  # NAX (M5 tensor-unit) kernels live in a separate metallib compiled with
  # -mmacosx-version-min=26.2: the MetalPerformancePrimitives tensor-ops API
  # needs macOS SDK 26.2+, and keeping them out of the classic metallib
  # preserves its deployment target. The C++ op only loads this library when
  # the runtime reports NAX support and degrades to the classic kernels when
  # the file is absent.
  execute_process(
    COMMAND xcrun -sdk macosx --show-sdk-version
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE MOLTO_MACOS_SDK_VERSION
    ERROR_QUIET)
  if(MOLTO_MACOS_SDK_VERSION VERSION_GREATER_EQUAL 26.2)
    set(MOLTO_QWEN35_PREFILL_NAX_METAL_FLAGS
        -x
        metal
        -Wall
        -Wextra
        -fno-fast-math
        -Wno-c++17-extensions
        -Wno-c++20-extensions
        -mmacosx-version-min=26.2)
    if(MLX_METAL_DEBUG)
      set(MOLTO_QWEN35_PREFILL_NAX_METAL_FLAGS
          ${MOLTO_QWEN35_PREFILL_NAX_METAL_FLAGS} -gline-tables-only
          -frecord-sources)
    endif()
    set(MOLTO_QWEN35_PREFILL_NAX_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/qwen35_qmm_nax.metal
        ${CMAKE_CURRENT_LIST_DIR}/qwen35_moe_gather_nax.metal
        ${CMAKE_CURRENT_LIST_DIR}/qwen35_oq_a8_nax.metal)
    set(MOLTO_QWEN35_PREFILL_NAX_AIR)
    foreach(NAX_SRCFILE IN LISTS MOLTO_QWEN35_PREFILL_NAX_SOURCES)
      get_filename_component(NAX_TARGET_NAME ${NAX_SRCFILE} NAME_WE)
      set(NAX_AIR_FILE ${CMAKE_CURRENT_BINARY_DIR}/${NAX_TARGET_NAME}.air)
      add_custom_command(
        OUTPUT ${NAX_AIR_FILE}
        COMMAND
          xcrun -sdk macosx metal ${MOLTO_QWEN35_PREFILL_NAX_METAL_FLAGS} -c
          ${NAX_SRCFILE} ${MOLTO_QWEN35_PREFILL_METAL_INCLUDE_ARGS} -o
          ${NAX_AIR_FILE}
        DEPENDS ${NAX_SRCFILE} ${MOLTO_QWEN35_PREFILL_METAL_HEADERS}
        COMMENT "Building ${NAX_TARGET_NAME}.air"
        VERBATIM)
      list(APPEND MOLTO_QWEN35_PREFILL_NAX_AIR ${NAX_AIR_FILE})
    endforeach()
    set(MOLTO_QWEN35_PREFILL_NAX_METALLIB
        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/molto_qwen35_prefill_kernels_nax.metallib)
    add_custom_command(
      OUTPUT ${MOLTO_QWEN35_PREFILL_NAX_METALLIB}
      # The link step also needs the 26.2 floor: MACOSX_DEPLOYMENT_TARGET in
      # the build environment otherwise pins the link to an older AIR version
      # and air-lld silently drops the newer NAX object (empty metallib).
      COMMAND xcrun -sdk macosx metal -mmacosx-version-min=26.2
              ${MOLTO_QWEN35_PREFILL_NAX_AIR} -o
              ${MOLTO_QWEN35_PREFILL_NAX_METALLIB}
      DEPENDS ${MOLTO_QWEN35_PREFILL_NAX_AIR}
      COMMENT "Building molto_qwen35_prefill_kernels_nax.metallib"
      VERBATIM)
    add_custom_target(molto_qwen35_prefill_kernels_nax_metallib
                      DEPENDS ${MOLTO_QWEN35_PREFILL_NAX_METALLIB})
    add_dependencies(molto_qwen35_prefill_kernel_ops
                     molto_qwen35_prefill_kernels_nax_metallib)
  else()
    message(
      STATUS
        "molto: macOS SDK ${MOLTO_MACOS_SDK_VERSION} < 26.2, skipping qwen35 NAX kernels"
    )
  endif()
endif()

nanobind_add_module(
  _ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE molto_qwen35_prefill_kernel_ops)

if(BUILD_SHARED_LIBS)
  # @loader_path resolves the sibling kernel_ops dylib. The second entry
  # resolves the mlx wheel's libmlx.dylib from the installed site-packages
  # layout (molto/custom_kernels/<pkg> -> site-packages -> mlx/lib); without
  # it the only libmlx rpath is the link-time absolute path into the build
  # env, which is dead after install (issue #2233).
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path
                      -Wl,-rpath,@loader_path/../../../mlx/lib)
endif()
