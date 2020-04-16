package com.baidu.paddle.lite.demo.face_keypoints_detection;

public class Native {
    static {
        System.loadLibrary("Native");
    }

    private long ctx = 0;

    public boolean init(String fdtModelDir,
                        int fdtCPUThreadNum,
                        String fdtCPUPowerMode,
                        float fdtInputScale,
                        float[] fdtInputMean,
                        float[] fdtInputStd,
                        float fdtScoreThreshold,
                        String fkpModelDir,
                        int fkpCPUThreadNum,
                        String fkpCPUPowerMode,
                        int fkpInputWidth,
                        int fkpInputHeight,
                        float[] fkpInputMean,
                        float[] fkpInputStd) {
        ctx = nativeInit(
                fdtModelDir,
                fdtCPUThreadNum,
                fdtCPUPowerMode,
                fdtInputScale,
                fdtInputMean,
                fdtInputStd,
                fdtScoreThreshold,
                fkpModelDir,
                fkpCPUThreadNum,
                fkpCPUPowerMode,
                fkpInputWidth,
                fkpInputHeight,
                fkpInputMean,
                fkpInputStd);
        return ctx == 0;
    }

    public boolean release() {
        if (ctx == 0) {
            return false;
        }
        return nativeRelease(ctx);
    }

    public boolean process(int inTextureId, int outTextureId, int textureWidth, int textureHeight, String savedImagePath) {
        if (ctx == 0) {
            return false;
        }
        return nativeProcess(ctx, inTextureId, outTextureId, textureWidth, textureHeight, savedImagePath);
    }

    public static native long nativeInit(String fdtModelDir,
                                         int fdtCPUThreadNum,
                                         String fdtCPUPowerMode,
                                         float fdtInputScale,
                                         float[] fdtInputMean,
                                         float[] fdtInputStd,
                                         float fdtScoreThreshold,
                                         String fkpModelDir,
                                         int fkpCPUThreadNum,
                                         String fkpCPUPowerMode,
                                         int fkpInputWidth,
                                         int fkpInputHeight,
                                         float[] fkpInputMean,
                                         float[] fkpInputStd);

    public static native boolean nativeRelease(long ctx);

    public static native boolean nativeProcess(long ctx, int inTextureId, int outTextureId, int textureWidth, int textureHeight, String savedImagePath);
}
