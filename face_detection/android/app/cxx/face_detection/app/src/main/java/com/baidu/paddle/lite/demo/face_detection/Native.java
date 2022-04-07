package com.baidu.paddle.lite.demo.face_detection;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.util.Log;

public class Native {
    static {
        System.loadLibrary("Native");
    }
    protected Bitmap inputImage = null;
    protected Bitmap outputImage = null;
    protected long[] inputShape = new long[]{1, 3, 240, 320};
    protected int height;
    protected int width;
    protected float inferenceTime = 0;

    private long ctx = 0;

    public boolean init(String modelDir,
                        String labelPath,
                        int cpuThreadNum,
                        String cpuPowerMode,
                        long[] inputShape,
                        float[] inputMean,
                        float[] inputStd) {
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
            Log.i("Paddle-lite", "Only one batch is supported in the image classification demo, you can use any batch size in " +
                    "your Apps!");
            return false;
        }
        if (inputShape[1] != 1 && inputShape[1] != 3) {
            Log.i("Paddle-lite", "Only one/three channels are supported in the image classification demo, you can use any " +
                    "channel size in your Apps!");
            return false;
        }
        this.inputShape = inputShape;
        ctx = nativeInit(
                modelDir,
                labelPath,
                cpuThreadNum,
                cpuPowerMode,
                inputShape,
                inputMean,
                inputStd);
        return ctx != 0;
    }

    public boolean release() {
        if (ctx == 0) {
            return false;
        }
        return nativeRelease(ctx);
    }
    public void draw(float[] boxAndScores, Bitmap outputImage){
        Canvas canvas = new Canvas(outputImage);
        Paint rectPaint = new Paint();
        rectPaint.setStyle(Paint.Style.STROKE);
        rectPaint.setStrokeWidth(2);
        Paint txtPaint = new Paint();
        txtPaint.setTextSize(12);
        txtPaint.setAntiAlias(true);
        int color = 0xFFFFFF33;
        rectPaint.setColor(color);
        txtPaint.setColor(color);
        int txtXOffset = 4;
        int txtYOffset = (int) (Math.ceil(-txtPaint.getFontMetrics().ascent));

        for (int i = 0; i + 4 < boxAndScores.length; i+=5)
            canvas.drawRect(boxAndScores[i], boxAndScores[i+1], boxAndScores[i+2], boxAndScores[i+3], rectPaint);
    }

    public boolean process() {
        if (ctx == 0) {
            return false;
        }
        float[] result = nativeProcess(ctx, this.inputImage, this.height, this.width);
        draw(result,outputImage);
        inferenceTime = result[result.length-1];
        return result.length > 0;
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
        this.height = rgbaImage.getHeight();
        this.width = rgbaImage.getWidth();
        this.inputImage = scaleImage;
        this.outputImage = rgbaImage;
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

    public static native long nativeInit(String modelDir,
                                         String labelPath,
                                         int cpuThreadNum,
                                         String cpuPowerMode,
                                         long[] inputShape,
                                         float[] inputMean,
                                         float[] inputStd);

    public static native boolean nativeRelease(long ctx);

    public static native float[] nativeProcess(long ctx, Bitmap ARGB888ImageBitmap, int height, int width);

}
