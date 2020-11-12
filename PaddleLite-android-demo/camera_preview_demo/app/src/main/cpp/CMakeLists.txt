# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

# Creates and names a library, sets it as either STATIC or SHARED, and provides
# the relative paths to its source code. You can define multiple libraries, and
# CMake builds them for you. Gradle automatically packages shared libraries with
# your APK.

set(OpenCV_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../OpenCV/sdk/native/jni")
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -ffast-math -Ofast -Os -DNDEBUG -fno-exceptions -fomit-frame-pointer -fno-asynchronous-unwind-tables -fno-unwind-tables"
)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden -fdata-sections -ffunction-sections"
)
set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -Wl,-z,nocopyreloc")

add_library(
  # Sets the name of the library.
  Native
  # Sets the library as a shared library.
  SHARED
  # Provides a relative path to your source file(s).
  Native.cc Pipeline.cc Utils.cc)

find_library(
  # Sets the name of the path variable.
  log-lib
  # Specifies the name of the NDK library that you want CMake to locate.
  log)

target_link_libraries(# Specifies the target library.
                      Native ${OpenCV_LIBS} GLESv2 EGL ${log-lib})

# Pack c++ runtime library into APK
if(${ANDROID_STL} MATCHES "_shared")
  if("${ANDROID_STL}" STREQUAL "gnustl_shared")
    set(so_dir "gnu-libstdc++/4.9")
    set(so_name "gnustl_shared")
  elseif("${ANDROID_STL}" STREQUAL "c++_shared")
    set(so_dir "llvm-libc++")
    set(so_name "c++_shared")
  else()
    message(FATAL_ERROR "ANDROID_STL=${ANDROID_STL} is not supported")
  endif()
  add_custom_command(
    TARGET Native
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E copy
      ${ANDROID_NDK}/sources/cxx-stl/${so_dir}/libs/${ANDROID_ABI}/lib${so_name}.so
      ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/lib${so_name}.so)
endif()
