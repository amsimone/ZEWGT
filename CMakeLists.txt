cmake_minimum_required(VERSION 3.8)
project(zewgt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GRIFFIN_DIR "" CACHE PATH "Path to GRIFFIN root")
if(NOT GRIFFIN_DIR)
  message(FATAL_ERROR "GRIFFIN_DIR is required. Configure with -DGRIFFIN_DIR=/path/to/GRIFFIN")
endif()

if(NOT EXISTS "${GRIFFIN_DIR}")
  message(FATAL_ERROR "GRIFFIN_DIR does not exist: ${GRIFFIN_DIR}")
endif()

add_executable(zewgt lhe_add_griffin_ewvirt.cc)
target_include_directories(zewgt PRIVATE "${GRIFFIN_DIR}")
target_link_directories(zewgt PRIVATE "${GRIFFIN_DIR}")
target_link_libraries(zewgt PRIVATE griffin)
