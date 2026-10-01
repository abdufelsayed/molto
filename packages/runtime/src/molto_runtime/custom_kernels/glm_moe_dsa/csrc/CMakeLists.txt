cmake_minimum_required(VERSION 3.27)

project(molto_glm_kernels LANGUAGES CXX)

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
list(GET MLX_INCLUDE_DIRS 0 MOLTO_GLM_MLX_INCLUDE_DIR)

set(MOLTO_CUSTOM_KERNEL_COMMON_DIR
    ${CMAKE_CURRENT_LIST_DIR}/../../common/csrc)

add_library(molto_glm_kernel_ops)
target_sources(
  molto_glm_kernel_ops
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/dsa_indexer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/deepseek_v4_sparse_attention.cpp
  ${CMAKE_CURRENT_LIST_DIR}/deepseek_v41_packed_attention.cpp
  ${CMAKE_CURRENT_LIST_DIR}/deepseek_v41_grouped_expert.cpp
  ${CMAKE_CURRENT_LIST_DIR}/dspark_gemm.cpp
  ${CMAKE_CURRENT_LIST_DIR}/dspark_qmv.cpp
  ${CMAKE_CURRENT_LIST_DIR}/exact_block_attention.cpp
  ${CMAKE_CURRENT_LIST_DIR}/fused_moe.cpp
  ${CMAKE_CURRENT_LIST_DIR}/qwen4_qsa_sparse_gqa.cpp
  ${CMAKE_CURRENT_LIST_DIR}/sparse_mla.cpp)
target_include_directories(
  molto_glm_kernel_ops
  PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${MOLTO_CUSTOM_KERNEL_COMMON_DIR})
target_compile_definitions(
  molto_glm_kernel_ops
  PUBLIC MOLTO_GLM_KERNEL_SOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}"
         MOLTO_GLM_MLX_INCLUDE_DIR="${MOLTO_GLM_MLX_INCLUDE_DIR}")
target_link_libraries(molto_glm_kernel_ops PUBLIC mlx)

if(MLX_BUILD_METAL)
  set(MOLTO_GLM_METAL_SOURCES
      ${CMAKE_CURRENT_LIST_DIR}/deepseek_moe.metal
      ${CMAKE_CURRENT_LIST_DIR}/deepseek_v4_sparse_attention.metal
      ${CMAKE_CURRENT_LIST_DIR}/deepseek_v41_packed_attention.metal
      ${CMAKE_CURRENT_LIST_DIR}/dspark_gemm.metal
      ${CMAKE_CURRENT_LIST_DIR}/dspark_qmv.metal
      ${CMAKE_CURRENT_LIST_DIR}/dsa_indexer.metal
      ${CMAKE_CURRENT_LIST_DIR}/exact_block_attention.metal
      ${CMAKE_CURRENT_LIST_DIR}/fused_moe.metal
      ${CMAKE_CURRENT_LIST_DIR}/qwen4_qsa_sparse_gqa.metal
      ${CMAKE_CURRENT_LIST_DIR}/sparse_mla.metal)
  file(GLOB_RECURSE MOLTO_GLM_METAL_HEADERS CONFIGURE_DEPENDS
       ${CMAKE_CURRENT_LIST_DIR}/*.h
       ${MOLTO_CUSTOM_KERNEL_COMMON_DIR}/kernels/*.h
       ${MOLTO_CUSTOM_KERNEL_COMMON_DIR}/mlx/**/*.h)
  set(MOLTO_GLM_METAL_FLAGS
      -x
      metal
      -Wall
      -Wextra
      -fno-fast-math
      -Wno-c++17-extensions
      -Wno-c++20-extensions)
  if(MLX_METAL_DEBUG)
    set(MOLTO_GLM_METAL_FLAGS ${MOLTO_GLM_METAL_FLAGS} -gline-tables-only
                             -frecord-sources)
  endif()
  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
    set(MOLTO_GLM_METAL_FLAGS
        ${MOLTO_GLM_METAL_FLAGS}
        "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
  endif()
  set(MOLTO_GLM_METAL_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}
                                  ${MOLTO_CUSTOM_KERNEL_COMMON_DIR}
                                  ${MLX_INCLUDE_DIRS})
  set(MOLTO_GLM_METAL_INCLUDE_ARGS)
  foreach(INCLUDE_DIR IN LISTS MOLTO_GLM_METAL_INCLUDE_DIRS)
    list(APPEND MOLTO_GLM_METAL_INCLUDE_ARGS -I${INCLUDE_DIR})
  endforeach()
  set(MOLTO_GLM_METAL_AIR)
  foreach(SRCFILE IN LISTS MOLTO_GLM_METAL_SOURCES)
    get_filename_component(TARGET_NAME ${SRCFILE} NAME_WE)
    set(AIR_FILE ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.air)
    add_custom_command(
      OUTPUT ${AIR_FILE}
      COMMAND xcrun -sdk macosx metal ${MOLTO_GLM_METAL_FLAGS} -c ${SRCFILE}
              ${MOLTO_GLM_METAL_INCLUDE_ARGS} -o ${AIR_FILE}
      DEPENDS ${SRCFILE} ${MOLTO_GLM_METAL_HEADERS}
      COMMENT "Building ${TARGET_NAME}.air"
      VERBATIM)
    list(APPEND MOLTO_GLM_METAL_AIR ${AIR_FILE})
  endforeach()

  set(MOLTO_GLM_METALLIB
      ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/molto_glm_kernels.metallib)
  add_custom_command(
    OUTPUT ${MOLTO_GLM_METALLIB}
    COMMAND xcrun -sdk macosx metal ${MOLTO_GLM_METAL_AIR} -o
            ${MOLTO_GLM_METALLIB}
    DEPENDS ${MOLTO_GLM_METAL_AIR}
    COMMENT "Building molto_glm_kernels.metallib"
    VERBATIM)
  add_custom_target(molto_glm_kernels_metallib DEPENDS ${MOLTO_GLM_METALLIB})

  add_dependencies(molto_glm_kernel_ops molto_glm_kernels_metallib)

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
target_link_libraries(_ext PRIVATE molto_glm_kernel_ops)

if(BUILD_SHARED_LIBS)
  # @loader_path resolves the sibling kernel_ops dylib. The second entry
  # resolves the mlx wheel's libmlx.dylib from the installed site-packages
  # layout (molto/custom_kernels/<pkg> -> site-packages -> mlx/lib); without
  # it the only libmlx rpath is the link-time absolute path into the build
  # env, which is dead after install (issue #2233).
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path
                      -Wl,-rpath,@loader_path/../../../mlx/lib)
endif()
