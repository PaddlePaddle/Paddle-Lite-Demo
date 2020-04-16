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

#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_baidu_paddle_lite_demo_face_keypoints_detection_Native
 * Method:    nativeInit
 * Signature:
 * (Ljava/lang/String;ILjava/lang/String;F[F[FFLjava/lang/String;ILjava/lang/String;II[F[F)J
 */
JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_lite_demo_face_1keypoints_1detection_Native_nativeInit(
    JNIEnv *env, jclass thiz, jstring jfdtModelDir, jint fdtCPUThreadNum,
    jstring jfdtCPUPowerMode, jfloat fdtInputScale, jfloatArray jfdtInputMean,
    jfloatArray jfdtInputStd, jfloat fdtScoreThreshold, jstring jfkpModelDir,
    jint fkpCPUThreadNum, jstring jfkpCPUPowerMode, jint fkpInputWidth,
    jint fkpInputHeight, jfloatArray jfkpInputMean, jfloatArray jfkpInputStd) {
  std::string fdtModelDir = jstring_to_cpp_string(env, jfdtModelDir);
  std::string fdtCPUPowerMode = jstring_to_cpp_string(env, jfdtCPUPowerMode);
  std::vector<float> fdtInputMean =
      jfloatarray_to_float_vector(env, jfdtInputMean);
  std::vector<float> fdtInputStd =
      jfloatarray_to_float_vector(env, jfdtInputStd);
  std::string fkpModelDir = jstring_to_cpp_string(env, jfkpModelDir);
  std::string fkpCPUPowerMode = jstring_to_cpp_string(env, jfkpCPUPowerMode);
  std::vector<float> fkpInputMean =
      jfloatarray_to_float_vector(env, jfkpInputMean);
  std::vector<float> fkpInputStd =
      jfloatarray_to_float_vector(env, jfkpInputStd);
  return reinterpret_cast<jlong>(
      new Pipeline(fdtModelDir, fdtCPUThreadNum, fdtCPUPowerMode, fdtInputScale,
                   fdtInputMean, fdtInputStd, fdtScoreThreshold, fkpModelDir,
                   fkpCPUThreadNum, fkpCPUPowerMode, fkpInputWidth,
                   fkpInputHeight, fkpInputMean, fkpInputStd));
}

/*
 * Class:     com_baidu_paddle_lite_demo_face_keypoints_detection_Native
 * Method:    nativeRelease
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_lite_demo_face_1keypoints_1detection_Native_nativeRelease(
    JNIEnv *env, jclass thiz, jlong ctx) {
  if (ctx == 0) {
    return JNI_FALSE;
  }
  Pipeline *pipeline = reinterpret_cast<Pipeline *>(ctx);
  delete pipeline;
  return JNI_TRUE;
}

/*
 * Class:     com_baidu_paddle_lite_demo_face_keypoints_detection_Native
 * Method:    nativeProcess
 * Signature: (JIIIILjava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_lite_demo_face_1keypoints_1detection_Native_nativeProcess(
    JNIEnv *env, jclass thiz, jlong ctx, jint inTextureId, jint outTextureId,
    jint textureWidth, jint textureHeight, jstring jsavedImagePath) {
  if (ctx == 0) {
    return JNI_FALSE;
  }
  std::string savedImagePath = jstring_to_cpp_string(env, jsavedImagePath);
  Pipeline *pipeline = reinterpret_cast<Pipeline *>(ctx);
  return pipeline->Process(inTextureId, outTextureId, textureWidth,
                           textureHeight, savedImagePath);
}

#ifdef __cplusplus
}
#endif
