package com.baidu.paddle.lite.demo;

import android.content.Context;
import android.graphics.*;
import android.util.Log;

import com.baidu.paddle.lite.Tensor;

import java.io.InputStream;
import java.util.Date;
import java.util.Vector;

import static android.graphics.Color.blue;
import static android.graphics.Color.green;
import static android.graphics.Color.red;

public class ObjDetectPredictor extends Predictor {
    private static final String TAG = ObjDetectPredictor.class.getSimpleName();
    protected Vector<String> wordLabels = new Vector<String>();
    protected Boolean enableRGBColorFormat = false;
    protected long[] inputShape = new long[]{1, 3, 300, 300};
    protected float[] inputMean = new float[]{0.5f, 0.5f, 0.5f};
    protected float[] inputStd = new float[]{0.5f, 0.5f, 0.5f};
    protected float scoreThreshold = 0.5f;
    protected Bitmap inputImage = null;
    protected Bitmap outputImage = null;
    protected String outputResult = "";
    protected float preprocessTime = 0;
    protected float postprocessTime = 0;

    public ObjDetectPredictor() {
        super();
    }

    public boolean init(Context appCtx, String modelPath, String labelPath, Boolean enableRGBColorFormat,
                        long[] inputShape, float[] inputMean,
                        float[] inputStd, float scoreThreshold) {
        if (inputShape.length != 4) {
            Log.i(TAG, "size of input shape should be: 4");
            return false;
        }
        if (inputMean.length != inputShape[1]) {
            Log.i(TAG, "size of input mean should be: " + Long.toString(inputShape[1]));
            return false;
        }
        if (inputStd.length != inputShape[1]) {
            Log.i(TAG, "size of input std should be: " + Long.toString(inputShape[1]));
            return false;
        }
        if (inputShape[0] != 1) {
            Log.i(TAG, "only one batch is supported in the image classification demo, you can use any batch size in " +
                    "your Apps!");
            return false;
        }
        if (inputShape[1] != 1 && inputShape[1] != 3) {
            Log.i(TAG, "only one/three channels are supported in the image classification demo, you can use any " +
                    "channel size in your Apps!");
            return false;
        }
        super.init(appCtx, modelPath);
        if (!super.isLoaded()) {
            return false;
        }
        isLoaded &= loadLabel(labelPath);
        this.enableRGBColorFormat = enableRGBColorFormat;
        this.inputShape = inputShape;
        this.inputMean = inputMean;
        this.inputStd = inputStd;
        this.scoreThreshold = scoreThreshold;
        return isLoaded;
    }

    protected boolean loadLabel(String labelPath) {
        wordLabels.clear();
        // load word labels from file
        try {
            InputStream assetsInputStream = appCtx.getAssets().open(labelPath);
            int available = assetsInputStream.available();
            byte[] lines = new byte[available];
            assetsInputStream.read(lines);
            assetsInputStream.close();
            String words = new String(lines);
            String[] contents = words.split("\n");
            for (String content : contents) {
                wordLabels.add(content);
            }
            Log.i(TAG, "word label size: " + wordLabels.size());
        } catch (Exception e) {
            Log.e(TAG, e.getMessage());
            return false;
        }
        return true;
    }

    public Tensor getInput(int idx) {
        return super.getInput(idx);
    }

    public Tensor getOutput(int idx) {
        return super.getOutput(idx);
    }

    public boolean runModel(Bitmap image) {
        setInputImage(image);
        return runModel();
    }

    public boolean runModel() {
        if (inputImage == null) {
            return false;
        }

        // set input shape
        Tensor inputTensor = getInput(0);
        inputTensor.resize(inputShape);

        // pre-process image, and feed input tensor with pre-processed data
        Date start = new Date();
        int channels = (int) inputShape[1];
        int width = (int) inputShape[3];
        int height = (int) inputShape[2];
        float[] inputData = new float[channels * width * height];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int pixel = inputImage.getPixel(j, i);
                float r = (float) red(pixel) / 255.0f;
                float g = (float) green(pixel) / 255.0f;
                float b = (float) blue(pixel) / 255.0f;
                if (channels == 3) {
                    float ch0 = enableRGBColorFormat ? r : b;
                    float ch1 = g;
                    float ch2 = enableRGBColorFormat ? b : r;
                    int idx0 = i * width + j;
                    int idx1 = idx0 + width * height;
                    int idx2 = idx1 + width * height;
                    inputData[idx0] = (ch0 - inputMean[0]) / inputStd[0];
                    inputData[idx1] = (ch1 - inputMean[1]) / inputStd[1];
                    inputData[idx2] = (ch2 - inputMean[2]) / inputStd[2];
                } else { // channels = 1
                    float gray = (b + g + r) / 3.0f;
                    gray = gray - inputMean[0];
                    gray = gray / inputStd[0];
                    inputData[i * width + j] = gray;
                }
            }
        }
        inputTensor.setData(inputData);
        Date end = new Date();
        preprocessTime = (float) (end.getTime() - start.getTime());

        // inference
        super.runModel();

        // fetch output tensor
        Tensor outputTensor = getOutput(0);

        // post-process
        start = new Date();
        long outputShape[] = outputTensor.shape();
        long outputSize = 1;
        for (long s : outputShape) {
            outputSize *= s;
        }
        outputImage = inputImage;
        outputResult = new String();
        Canvas canvas = new Canvas(outputImage);
        Paint rectPaint = new Paint();
        rectPaint.setStyle(Paint.Style.STROKE);
        rectPaint.setStrokeWidth(1);
        Paint txtPaint = new Paint();
        txtPaint.setTextSize(12);
        txtPaint.setAntiAlias(true);
        int txtXOffset = 4;
        int txtYOffset = (int) (Math.ceil(-txtPaint.getFontMetrics().ascent));
        int imgWidth = outputImage.getWidth();
        int imgHeight = outputImage.getHeight();
        int objectIdx = 0;
        final int[] objectColor = {0xFFFF00CC, 0xFFFF0000, 0xFFFFFF33, 0xFF0000FF, 0xFF00FF00,
                0xFF000000, 0xFF339933};
        for (int i = 0; i < outputSize; i += 6) {
            float score = outputTensor.getFloatData()[i + 1];
            if (score < scoreThreshold) {
                continue;
            }
            int categoryIdx = (int) outputTensor.getFloatData()[i];
            String categoryName = "Unknown";
            if (wordLabels.size() > 0 && categoryIdx >= 0 && categoryIdx < wordLabels.size()) {
                categoryName = wordLabels.get(categoryIdx);
            }
            float rawLeft = outputTensor.getFloatData()[i + 2];
            float rawTop = outputTensor.getFloatData()[i + 3];
            float rawRight = outputTensor.getFloatData()[i + 4];
            float rawBottom = outputTensor.getFloatData()[i + 5];
            float clampedLeft = Math.max(Math.min(rawLeft, 1.f), 0.f);
            float clampedTop = Math.max(Math.min(rawTop, 1.f), 0.f);
            float clampedRight = Math.max(Math.min(rawRight, 1.f), 0.f);
            float clampedBottom = Math.max(Math.min(rawBottom, 1.f), 0.f);
            float imgLeft = clampedLeft * imgWidth;
            float imgTop = clampedTop * imgWidth;
            float imgRight = clampedRight * imgHeight;
            float imgBottom = clampedBottom * imgHeight;
            int color = objectColor[objectIdx % objectColor.length];
            rectPaint.setColor(color);
            txtPaint.setColor(color);
            canvas.drawRect(imgLeft, imgTop, imgRight, imgBottom, rectPaint);
            canvas.drawText(objectIdx + "." + categoryName + ":" + String.format("%.3f", score),
                    imgLeft + txtXOffset, imgTop + txtYOffset, txtPaint);
            outputResult += objectIdx + "." + categoryName + " - " + String.format("%.3f", score) +
                    " [" + String.format("%.3f", rawLeft) + "," + String.format("%.3f", rawTop) + "," + String.format("%.3f", rawRight) + "," + String.format("%.3f", rawBottom) + "]\n";
            objectIdx++;
        }
        end = new Date();
        postprocessTime = (float) (end.getTime() - start.getTime());
        return true;
    }

    public Bitmap inputImage() {
        return inputImage;
    }

    public Bitmap outputImage() {
        return outputImage;
    }

    public String outputResult() {
        return outputResult;
    }

    public float preprocessTime() {
        return preprocessTime;
    }

    public float postprocessTime() {
        return postprocessTime;
    }

    public void setInputImage(Bitmap image) {
        if (image == null) {
            return;
        }
        // scale image to the size of input tensor
        Bitmap rgbaImage = image.copy(Bitmap.Config.ARGB_8888, true);
        Bitmap scaleImage = Bitmap.createScaledBitmap(rgbaImage, (int) inputShape[3], (int) inputShape[2], true);
        this.inputImage = scaleImage;
    }
}
