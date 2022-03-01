// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Native.h"
#include "Pipeline.h"
#include <android/bitmap.h>

#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_baidu_paddle_lite_demo_ssd_detection_Native
 * Method:    nativeInit
 * Signature:
 * (Ljava/lang/String;Ljava/lang/String;ILjava/lang/String;II[F[FF)J
 */
JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_lite_demo_image_1classification_Native_nativeInit(
    JNIEnv *env, jclass thiz, jstring jModelDir, jstring jLabelPath,
    jint cpuThreadNum, jstring jCPUPowerMode, jlongArray jInputShape,
    jfloatArray jInputMean, jfloatArray jInputStd, jint jTopk) {
  std::string modelDir = jstring_to_cpp_string(env, jModelDir);
  std::string labelPath = jstring_to_cpp_string(env, jLabelPath);
  std::string cpuPowerMode = jstring_to_cpp_string(env, jCPUPowerMode);
  std::vector<int64_t> inputShape =
      jlongarray_to_int64_vector(env, jInputShape);
  std::vector<float> inputMean = jfloatarray_to_float_vector(env, jInputMean);
  std::vector<float> inputStd = jfloatarray_to_float_vector(env, jInputStd);
  return reinterpret_cast<jlong>(new Pipeline(modelDir, labelPath, cpuThreadNum,
                                              cpuPowerMode, inputShape,
                                              inputMean, inputStd, jTopk));
}

/*
 * Class:     com_baidu_paddle_lite_demo_ssd_detection_Native
 * Method:    nativeRelease
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_lite_demo_image_1classification_Native_nativeRelease(
    JNIEnv *env, jclass thiz, jlong ctx) {
  if (ctx == 0) {
    return JNI_FALSE;
  }
  Pipeline *pipeline = reinterpret_cast<Pipeline *>(ctx);
  delete pipeline;
  return JNI_TRUE;
}

/*
 * Class:     com_baidu_paddle_lite_demo_ssd_detection_Native
 * Method:    nativeProcess
 * Signature: (JIIIILjava/lang/String;)Z
 */
JNIEXPORT jstring JNICALL
Java_com_baidu_paddle_lite_demo_image_1classification_Native_nativeProcess(
    JNIEnv *env, jclass thiz, jlong ctx, jobject jARGB8888ImageBitmap) {
  if (ctx == 0) {
    return JNI_FALSE;
  }

  // Convert the android bitmap(ARGB8888) to the OpenCV RGBA image. Actually,
  // the data layout of AGRB8888 is R, G, B, A, it's the same as CV RGBA image,
  // so it is unnecessary to do the conversion of color format, check
  // https://developer.android.com/reference/android/graphics/Bitmap.Config#ARGB_8888
  // to get the more details about Bitmap.Config.ARGB8888
  auto t = GetCurrentTime();
  void *bitmapPixels;
  AndroidBitmapInfo bitmapInfo;
  if (AndroidBitmap_getInfo(env, jARGB8888ImageBitmap, &bitmapInfo) < 0) {
    LOGE("Invoke AndroidBitmap_getInfo() failed!");
    return JNI_FALSE;
  }
  if (bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
    LOGE("Only Bitmap.Config.ARGB8888 color format is supported!");
    return JNI_FALSE;
  }
  if (AndroidBitmap_lockPixels(env, jARGB8888ImageBitmap, &bitmapPixels) < 0) {
    LOGE("Invoke AndroidBitmap_lockPixels() failed!");
    return JNI_FALSE;
  }
  cv::Mat bmpImage(bitmapInfo.height, bitmapInfo.width, CV_8UC4, bitmapPixels);
  cv::Mat rgbaImage;
  bmpImage.copyTo(rgbaImage);
  if (AndroidBitmap_unlockPixels(env, jARGB8888ImageBitmap) < 0) {
    LOGE("Invoke AndroidBitmap_unlockPixels() failed!");
    return JNI_FALSE;
  }
  LOGD("Read from bitmap costs %f ms", GetElapsedTime(t));

  Pipeline *pipeline = reinterpret_cast<Pipeline *>(ctx);

  std::string res_str = pipeline->Process(rgbaImage);
  bool modified = res_str.empty();
  if (!modified) {
    // Convert the OpenCV RGBA image to the android bitmap(ARGB8888)
    if (rgbaImage.type() != CV_8UC4) {
      LOGE("Only CV_8UC4 color format is supported!");
      return JNI_FALSE;
    }
    t = GetCurrentTime();
    if (AndroidBitmap_lockPixels(env, jARGB8888ImageBitmap, &bitmapPixels) <
        0) {
      LOGE("Invoke AndroidBitmap_lockPixels() failed!");
      return JNI_FALSE;
    }
    cv::Mat bmpImage(bitmapInfo.height, bitmapInfo.width, CV_8UC4,
                     bitmapPixels);
    rgbaImage.copyTo(bmpImage);
    if (AndroidBitmap_unlockPixels(env, jARGB8888ImageBitmap) < 0) {
      LOGE("Invoke AndroidBitmap_unlockPixels() failed!");
      return JNI_FALSE;
    }
    LOGD("Write to bitmap costs %f ms", GetElapsedTime(t));
  }
  return cpp_string_to_jstring(env, res_str);
}

#ifdef __cplusplus
}
#endif
