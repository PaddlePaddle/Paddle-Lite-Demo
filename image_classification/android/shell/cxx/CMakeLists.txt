cmake_minimum_required(VERSION 3.10)
set(CMAKE_SYSTEM_NAME Linux)
# set(CMAKE_SYSTEM_NAME Draw) mac build
if(ARM_TARGET_LANG STREQUAL "gcc")
    set(ANDROID_ARM_NEON True)
    set(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION ${ARM_TARGET_LANG})
    set(ANDROID_TOOLCHAIN gcc)
    set(CMAKE_TOOLCHAIN_FILE ${NDK_ROOT}/build/cmake/android.toolchain.cmake)
    set(CMAKE_C_FLAGS gcc)
    set(CMAKE_CXX_FLAGS g++)
endif()

if(ARM_TARGET_LANG STREQUAL "clang")
    set(ANDROID_ARM_NEON True)
    set(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION ${ARM_TARGET_LANG})
    set(ANDROID_TOOLCHAIN clang)
    set(CMAKE_TOOLCHAIN_FILE ${NDK_ROOT}/build/cmake/android.toolchain.cmake)
    set(CMAKE_C_FLAGS clang)
    set(CMAKE_CXX_FLAGS clang++)
endif()

project(image_classification)
message(STATUS "TARGET ARM LANG: ${ARM_TARGET_LANG}")
message(STATUS "TARGET ARCH ABI: ${ARM_ABI}")
message(STATUS "PADDLE LITE DIR: ${PADDLE_LITE_DIR}")
message(STATUS "PADDLE LITE DIR: ${OPENCV_LITE_DIR}")

include_directories(${PADDLE_LITE_DIR}/include)
link_directories(${PADDLE_LITE_DIR}/libs/${ARM_ABI})

set(OPENCV_PATH "${OPENCV_LITE_DIR}/${ARM_ABI}")
message(STATUS "OPENCV_PATH: ${OPENCV_PATH}")
set(OPENCV_INCLUDE_DIRS "${OPENCV_PATH}/include/")
set(OPENCV_LIBS ${OPENCV_PATH}/libs/libopencv_imgcodecs.a
                    ${OPENCV_PATH}/libs/libopencv_imgproc.a
                    ${OPENCV_PATH}/libs/libopencv_core.a
                    ${OPENCV_PATH}/3rdparty/libs/libtegra_hal.a
                    ${OPENCV_PATH}/3rdparty/libs/liblibjpeg-turbo.a
                    ${OPENCV_PATH}/3rdparty/libs/liblibwebp.a
                    ${OPENCV_PATH}/3rdparty/libs/liblibpng.a
                    ${OPENCV_PATH}/3rdparty/libs/liblibjasper.a
                    ${OPENCV_PATH}/3rdparty/libs/liblibtiff.a
                    ${OPENCV_PATH}/3rdparty/libs/libIlmImf.a
                    ${OPENCV_PATH}/3rdparty/libs/libtbb.a
                    ${OPENCV_PATH}/3rdparty/libs/libcpufeatures.a)
 include_directories(${OPENCV_INCLUDE_DIRS})
 link_directories(${OPENCV_LIBS})

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -ffast-math -Ofast -Os -DNDEBUG -fno-exceptions -fomit-frame-pointer -fno-asynchronous-unwind-tables -fno-unwind-tables"
)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fexceptions -fvisibility=hidden -fvisibility-inlines-hidden -fdata-sections -ffunction-sections"
)
set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -Wl,-z,nocopyreloc")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(image_classification image_classification.cc)
target_link_libraries(image_classification paddle_light_api_shared ${OPENCV_LIBS} -lz -llog)
