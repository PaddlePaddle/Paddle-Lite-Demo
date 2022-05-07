package com.baidu.paddle.lite.demo.segmentation;

import android.graphics.Bitmap;
import android.util.Log;

public class Native {
    static {
        System.loadLibrary("Native");
    }
    protected Bitmap inputImage = null;
    protected Bitmap outputImage = null;
    protected long[] inputShape = new long[]{1, 3, 513, 513};
    protected float inferenceTime = 0;
    protected String top1Result = "";
    protected String top2Result = "";
    protected String top3Result = "";
    protected int topk = 1;

    private long ctx = 0;

    public boolean init(String modelDir,
                        String labelPath,
                        int cpuThreadNum,
                        String cpuPowerMode,
                        long[] inputShape,
                        float[] inputMean,
                        float[] inputStd,
                        int topk) {
        if (inputShape.length != 4) {
            Log.i("Paddle-lite", "Size of input shape should be: 4");
            return false;
        }
        if (inputMean.length != inputShape[1]) {
            Log.i("Paddle-lite", "Size of input mean should be: " + Long.toString(inputShape[1]));
            return false;
        }
        if (inputStd.length != inputShape[1]) {
            Log.i("Paddle-lite", "Size of input std should be: " + Long.toString(inputShape[1]));
            return false;
        }
        if (inputShape[0] != 1) {
            Log.i("Paddle-lite", "Only one batch is supported in this demo, you can use any batch size in " +
                    "your Apps!");
            return false;
        }
        if (inputShape[1] != 1 && inputShape[1] != 3) {
            Log.i("Paddle-lite", "Only one/three channels are supported in this demo, you can use any " +
                    "channel size in your Apps!");
            return false;
        }
        this.inputShape = inputShape;
        this.topk = topk;
        ctx = nativeInit(
                modelDir,
                labelPath,
                cpuThreadNum,
                cpuPowerMode,
                inputShape,
                inputMean,
                inputStd,
                topk);
        return ctx != 0;
    }

    public boolean release() {
        if (ctx == 0) {
            return false;
        }
        return nativeRelease(ctx);
    }

    public boolean process() {
        if (ctx == 0) {
            return false;
        }
        // ARGB8888 bitmap is only supported in native, other color formats can be added by yourself.
        String[] res = nativeProcess(ctx, this.inputImage, this.outputImage).split("\n");
        if(res.length > 0)
            inferenceTime = Float.parseFloat(res[0]);
        return (res.length > 0);
    }
    public boolean isLoaded() {
        return ctx != 0;
    }
    public void setInputImage(Bitmap image) {
        if (image == null) {
            return;
        }
        // Scale image to the size of input tensor
        Bitmap rgbaImage = image.copy(Bitmap.Config.ARGB_8888, true);
        Bitmap scaleImage = Bitmap.createScaledBitmap(rgbaImage, (int) inputShape[3], (int) inputShape[2], true);
        this.inputImage = scaleImage;
        this.outputImage = rgbaImage;
    }
    public Bitmap getOutputImage() {
        return this.outputImage;
    }
    public float inferenceTime() {
        return inferenceTime;
    }

    public Bitmap inputImage() {
        return inputImage;
    }

    public Bitmap outputImage() {
        return outputImage;
    }

    public String top1Result() {
        return top1Result;
    }

    public String top2Result() {
        return top2Result;
    }

    public String top3Result() {
        return top3Result;
    }

    public static native long nativeInit(String modelDir,
                                         String labelPath,
                                         int cpuThreadNum,
                                         String cpuPowerMode,
                                         long[] inputShape,
                                         float[] inputMean,
                                         float[] inputStd,
                                         int topk);

    public static native boolean nativeRelease(long ctx);

    public static native String nativeProcess(long ctx, Bitmap ARGB888ImageBitmap, Bitmap OutBitmap);

}
