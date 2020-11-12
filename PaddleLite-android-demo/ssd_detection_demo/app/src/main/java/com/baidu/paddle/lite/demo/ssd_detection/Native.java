package com.baidu.paddle.lite.demo.ssd_detection;

import android.content.Context;
import android.graphics.Bitmap;

public class Native {
    static {
        System.loadLibrary("Native");
    }

    private long ctx = 0;

    public boolean init(String modelDir,
                        String labelPath,
                        int cpuThreadNum,
                        String cpuPowerMode,
                        int inputWidth,
                        int inputHeight,
                        float[] inputMean,
                        float[] inputStd,
                        float scoreThreshold) {
        ctx = nativeInit(
                modelDir,
                labelPath,
                cpuThreadNum,
                cpuPowerMode,
                inputWidth,
                inputHeight,
                inputMean,
                inputStd,
                scoreThreshold);
        return ctx == 0;
    }

    public boolean release() {
        if (ctx == 0) {
            return false;
        }
        return nativeRelease(ctx);
    }

    public boolean process(Bitmap ARGB8888ImageBitmap, String savedImagePath) {
        if (ctx == 0) {
            return false;
        }
        // ARGB8888 bitmap is only supported in native, other color formats can be added by yourself.
        return nativeProcess(ctx, ARGB8888ImageBitmap, savedImagePath);
    }

    public static native long nativeInit(String modelDir,
                                         String labelPath,
                                         int cpuThreadNum,
                                         String cpuPowerMode,
                                         int inputWidth,
                                         int inputHeight,
                                         float[] inputMean,
                                         float[] inputStd,
                                         float scoreThreshold);

    public static native boolean nativeRelease(long ctx);

    public static native boolean nativeProcess(long ctx, Bitmap ARGB888ImageBitmap, String savedImagePath);
}
