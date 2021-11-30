cmake_minimum_required(VERSION 3.10)
set(CMAKE_SYSTEM_NAME Linux)
# set(CMAKE_SYSTEM_NAME Draw) mac build
if(ARM_TARGET_LANG STREQUAL "gcc")
    if(ARM_ABI STREQUAL "arm64-v8a")
        set(CMAKE_SYSTEM_PROCESSOR aarch64)
        set(CMAKE_C_FLAGS "aarch64-linux-gnu-gcc")
        set(CMAKE_CXX_FLAGS "aarch64-linux-gnu-g++")
    elseif(ARM_TARGET_ARCH_ABI STREQUAL "armeabi-v7a")
        set(CMAKE_SYSTEM_PROCESSOR armv7-a)
        set(CMAKE_C_FLAGS "arm-linux-gnueabi-gcc")
        set(CMAKE_CXX_FLAGS "arm-linux-gnueabi-g++")
    else()
        message(FATAL_ERROR "INVALID ARM TARGET ARCH ABI: ${ARM_ABI}")
    endif()
endif()
if(ARM_TARGET_LANG STREQUAL "clang")
    set(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION ${ARM_TARGET_LANG})
    set(ANDROID_TOOLCHAIN clang)
    set(CMAKE_TOOLCHAIN_FILE ${ANDROID_NDK}/build/cmake/android.toolchain.cmake)
    set(CMAKE_C_FLAGS clang)
    set(CMAKE_CXX_FLAGS clang++)
endif()

project(ocr_db_crnn_demo)
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

add_executable(ocr_db_crnn_demo ocr_db_crnn.cc)
target_link_libraries(ocr_db_crnn_demo paddle_light_api_shared ${OPENCV_LITE_DIR}/${ARM_ABI}/libs)
