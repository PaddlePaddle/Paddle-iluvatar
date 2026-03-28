# =============================================================================
# CINN Plugin for Iluvatar (IXUCA) Backend
# =============================================================================

set(IXUCA_PATH $ENV{IXUCA_PATH})
if(NOT IXUCA_PATH)
  set(IXUCA_PATH "/usr/local/corex")
  message(
    STATUS "[Iluvatar CINN] IXUCA_PATH not set, using default: ${IXUCA_PATH}")
else()
  message(STATUS "[Iluvatar CINN] Found IXUCA_PATH: ${IXUCA_PATH}")
endif()

set(CINN_SRCS cinn_interface.cc compiler/compiler.cc runtime/cinn_runtime.cc
              passes/pass_manager.cc)

add_library(iluvatar_cinn_obj OBJECT ${CINN_SRCS})

target_include_directories(
  iluvatar_cinn_obj
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../
          ${IXUCA_PATH}/include ${PADDLE_SOURCE_DIR})

set_property(TARGET iluvatar_cinn_obj PROPERTY CXX_STANDARD 17)

set_property(TARGET iluvatar_cinn_obj PROPERTY POSITION_INDEPENDENT_CODE ON)
