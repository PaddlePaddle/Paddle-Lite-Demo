cmake_minimum_required(VERSION 3.10)
set(CMAKE_SYSTEM_NAME Linux)
# set(CMAKE_SYSTEM_NAME Draw) mac build
if(ARM_TARGET_LANG STREQUAL "gcc")
    set(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION ${ARM_TARGET_LANG})
    set(ANDROID_TOOLCHAIN gcc)
    set(CMAKE_TOOLCHAIN_FILE ${ANDROID_NDK}/build/cmake/android.toolchain.cmake)
    set(CMAKE_C_FLAGS gcc)
    set(CMAKE_CXX_FLAGS g++)
endif()

if(ARM_TARGET_LANG STREQUAL "clang")
    set(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION ${ARM_TARGET_LANG})
    set(ANDROID_TOOLCHAIN clang)
    set(CMAKE_TOOLCHAIN_FILE ${ANDROID_NDK}/build/cmake/android.toolchain.cmake)
    set(CMAKE_C_FLAGS clang)
    set(CMAKE_CXX_FLAGS clang++)
endif()

project(ocr_demo)
message(STATUS "TARGET ARM LANG: ${ARM_TARGET_LANG}")
message(STATUS "TARGET ARCH ABI: ${ARM_ABI}")
message(STATUS "PADDLE LITE DIR: ${PADDLE_LITE_DIR}")
message(STATUS "PADDLE LITE DIR: ${OPENCV_LITE_DIR}")

include_directories(${PADDLE_LITE_DIR}/include)
include_directories(${OPENCV_LITE_DIR}/${ARM_ABI}/include)
link_directories(${PADDLE_LITE_DIR}/libs/${ARM_ABI})
link_directories(${OPENCV_LITE_DIR}/${ARM_ABI}/libs)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -ffast-math -Ofast -Os -DNDEBUG -fno-exceptions -fomit-frame-pointer -fno-asynchronous-unwind-tables -fno-unwind-tables"
)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden -fdata-sections -ffunction-sections"
)
set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -Wl,-z,nocopyreloc")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(ocr_db_crnn_demo Pipeline.cc Uitls.cc cls.cc ret.cc det.cc clipper.cpp db_post_process.cc)
target_link_libraries(ocr_demo paddle_light_api_shared ${OPENCV_LITE_DIR}/${ARM_ABI}/libs)
