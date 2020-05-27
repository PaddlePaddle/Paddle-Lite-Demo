package com.baidu.paddle.lite.demo.camera_preview;

import android.content.Context;

public class Native {
    static {
        System.loadLibrary("Native");
    }

    private long ctx = 0;

    public boolean init() {
        ctx = nativeInit();
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

    public static native long nativeInit();

    public static native boolean nativeRelease(long ctx);

    public static native boolean nativeProcess(long ctx, int inTextureId, int outTextureId, int textureWidth, int textureHeight, String savedImagePath);
}
