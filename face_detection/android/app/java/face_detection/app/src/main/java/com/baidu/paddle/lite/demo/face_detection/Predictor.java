package com.baidu.paddle.lite.demo.face_detection;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.util.Log;

import com.baidu.paddle.lite.MobileConfig;
import com.baidu.paddle.lite.PaddlePredictor;
import com.baidu.paddle.lite.PowerMode;
import com.baidu.paddle.lite.Tensor;

import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Vector;

import static android.graphics.Color.blue;
import static android.graphics.Color.green;
import static android.graphics.Color.red;

public class Predictor {
    private static final String TAG = Predictor.class.getSimpleName();
    public boolean isLoaded = false;
    public int warmupIterNum = 1;
    public int inferIterNum = 1;
    public int cpuThreadNum = 1;
    public String cpuPowerMode = "LITE_POWER_HIGH";
    public String modelPath = "";
    public String modelName = "";
    protected PaddlePredictor paddlePredictor = null;
    protected float inferenceTime = 0;
    // Only for image classification
    protected Vector<String> wordLabels = new Vector<String>();
    protected String inputColorFormat = "RGB";
    protected long[] inputShape = new long[]{1, 3, 240, 320};
    protected float[] inputMean = new float[]{0.498f,0.498f,0.498f};
    protected float[] inputStd = new float[]{0.502f,0.502f,0.502f};
    protected Bitmap inputImage = null;
    protected Bitmap outputImage = null;
    protected float preprocessTime = 0;
    protected float postprocessTime = 0;
    protected int nmsParamTopK = 100;
    protected float nmsParamNmsThreshold = (float) 0.5;
    protected float confidenceThreshold = (float) 0.5;

    public Predictor() {
    }

    public boolean init(Context appCtx, String modelPath, String labelPath, int cpuThreadNum, String cpuPowerMode,
                        String inputColorFormat,
                        long[] inputShape, float[] inputMean,
                        float[] inputStd) {
        if (inputShape.length != 4) {
            Log.i(TAG, "Size of input shape should be: 4");
            return false;
        }
        if (inputMean.length != inputShape[1]) {
            Log.i(TAG, "Size of input mean should be: " + Long.toString(inputShape[1]));
            return false;
        }
        if (inputStd.length != inputShape[1]) {
            Log.i(TAG, "Size of input std should be: " + Long.toString(inputShape[1]));
            return false;
        }
        if (inputShape[0] != 1) {
            Log.i(TAG, "Only one batch is supported in the image classification demo, you can use any batch size in " +
                    "your Apps!");
            return false;
        }
        if (inputShape[1] != 1 && inputShape[1] != 3) {
            Log.i(TAG, "Only one/three channels are supported in the image classification demo, you can use any " +
                    "channel size in your Apps!");
            return false;
        }
        if (!inputColorFormat.equalsIgnoreCase("RGB") && !inputColorFormat.equalsIgnoreCase("BGR")) {
            Log.i(TAG, "Only RGB and BGR color format is supported.");
            return false;
        }
        isLoaded = loadModel(appCtx, modelPath, cpuThreadNum, cpuPowerMode);
        if (!isLoaded) {
            return false;
        }
        isLoaded = loadLabel(appCtx, labelPath);
        if (!isLoaded) {
            return false;
        }
        this.inputColorFormat = inputColorFormat;
        this.inputShape = inputShape;
        this.inputMean = inputMean;
        this.inputStd = inputStd;
        return true;
    }

    protected boolean loadModel(Context appCtx, String modelPath, int cpuThreadNum, String cpuPowerMode) {
        // Release model if exists
        releaseModel();

        // Load model
        if (modelPath.isEmpty()) {
            return false;
        }
        String realPath = modelPath;
        if (!modelPath.substring(0, 1).equals("/")) {
            // Read model files from custom path if the first character of mode path is '/'
            // otherwise copy model to cache from assets
            realPath = appCtx.getCacheDir() + "/" + modelPath;
            // push model to mobile
            Utils.copyDirectoryFromAssets(appCtx, modelPath, realPath);
        }
        if (realPath.isEmpty()) {
            return false;
        }
        MobileConfig config = new MobileConfig();
        config.setModelFromFile(realPath + File.separator + "model.nb");
        config.setThreads(cpuThreadNum);
        if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_HIGH")) {
            config.setPowerMode(PowerMode.LITE_POWER_HIGH);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_LOW")) {
            config.setPowerMode(PowerMode.LITE_POWER_LOW);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_FULL")) {
            config.setPowerMode(PowerMode.LITE_POWER_FULL);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_NO_BIND")) {
            config.setPowerMode(PowerMode.LITE_POWER_NO_BIND);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_RAND_HIGH")) {
            config.setPowerMode(PowerMode.LITE_POWER_RAND_HIGH);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_RAND_LOW")) {
            config.setPowerMode(PowerMode.LITE_POWER_RAND_LOW);
        } else {
            Log.e(TAG, "Unknown cpu power mode!");
            return false;
        }
        paddlePredictor = PaddlePredictor.createPaddlePredictor(config);

        this.cpuThreadNum = cpuThreadNum;
        this.cpuPowerMode = cpuPowerMode;
        this.modelPath = realPath;
        this.modelName = realPath.substring(realPath.lastIndexOf("/") + 1);
        return true;
    }

    public void releaseModel() {
        paddlePredictor = null;
        isLoaded = false;
        cpuThreadNum = 1;
        cpuPowerMode = "LITE_POWER_HIGH";
        modelPath = "";
        modelName = "";
    }

    protected boolean loadLabel(Context appCtx, String labelPath) {
        wordLabels.clear();
        // Load word labels from file
        try {
            InputStream assetsInputStream = appCtx.getAssets().open(labelPath);
            int available = assetsInputStream.available();
            byte[] lines = new byte[available];
            assetsInputStream.read(lines);
            assetsInputStream.close();
            String words = new String(lines);
            String[] contents = words.split("\n");
            for (String content : contents) {
                int first_space_pos = content.indexOf(" ");
                if (first_space_pos >= 0 && first_space_pos < content.length()) {
                    wordLabels.add(content.substring(first_space_pos));
                }
            }
            Log.i(TAG, "Word label size: " + wordLabels.size());
        } catch (Exception e) {
            Log.e(TAG, e.getMessage());
            return false;
        }
        return true;
    }

    public Tensor getInput(int idx) {
        if (!isLoaded()) {
            return null;
        }
        return paddlePredictor.getInput(idx);
    }

    public Tensor getOutput(int idx) {
        if (!isLoaded()) {
            return null;
        }
        return paddlePredictor.getOutput(idx);
    }

    public boolean preProcess(){
        // Set input shape
        Tensor inputTensor = getInput(0);
        inputTensor.resize(inputShape);
        Date start = new Date();
        int channels = (int) inputShape[1];
        int width = (int) inputShape[3];
        int height = (int) inputShape[2];
        float[] inputData = new float[channels * width * height];
        if (channels == 3) {
            int[] channelIdx = null;
            if (inputColorFormat.equalsIgnoreCase("RGB")) {
                channelIdx = new int[]{0, 1, 2};
            } else if (inputColorFormat.equalsIgnoreCase("BGR")) {
                channelIdx = new int[]{2, 1, 0};
            } else {
                Log.i(TAG, "Unknown color format " + inputColorFormat + ", only RGB and BGR color format is " +
                        "supported!");
                return false;
            }
            int[] channelStride = new int[]{width * height, width * height * 2};
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int color = inputImage.getPixel(x, y);
                    float[] rgb = new float[]{(float) red(color) / 255.0f, (float) green(color) / 255.0f,
                            (float) blue(color) / 255.0f};
                    inputData[y * width + x] = (rgb[channelIdx[0]] - inputMean[0]) / inputStd[0];
                    inputData[y * width + x + channelStride[0]] = (rgb[channelIdx[1]] - inputMean[1]) / inputStd[1];
                    inputData[y * width + x + channelStride[1]] = (rgb[channelIdx[2]] - inputMean[2]) / inputStd[2];
                }
            }
        } else if (channels == 1) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int color = inputImage.getPixel(x, y);
                    float gray = (float) (red(color) + green(color) + blue(color)) / 3.0f / 255.0f;
                    inputData[y * width + x] = (gray - inputMean[0]) / inputStd[0];
                }
            }
        } else {
            Log.i(TAG, "Unsupported channel size " + Integer.toString(channels) + ",  only channel 1 and 3 is " +
                    "supported!");
            return false;
        }
        inputTensor.setData(inputData);
        Date end = new Date();
        preprocessTime = (float) (end.getTime() - start.getTime());
        return true;
    }

    public void postProcess() {
        Date start = new Date();
        List<float[]>  boxAndScores = new ArrayList<>();
        float[][] boxs = new float[8840][4];
        float[] scores = new float[8840];

        Tensor scoresTensor = getOutput(0);
        Tensor boxsTensor = getOutput(1);

        long scoresShape[] = scoresTensor.shape();
        long scoresSize = 1;
        for (long s : scoresShape) {
            scoresSize *= s;
        }

        int imgWidth = outputImage.getWidth();
        int imgHeight = outputImage.getHeight();
        int number_boxs = 0;

        float[] box = new float[4];

        for (int i = 0, j = 0; i < scoresSize; i += 2, j+=4) {

            float rawLeft = boxsTensor.getFloatData()[j];
            float rawTop = boxsTensor.getFloatData()[j + 1];
            float rawRight = boxsTensor.getFloatData()[j + 2];
            float rawBottom = boxsTensor.getFloatData()[j + 3];

            float clampedLeft = Math.max(Math.min(rawLeft, 1.f), 0.f);
            float clampedTop = Math.max(Math.min(rawTop, 1.f), 0.f);

            float clampedRight = Math.max(Math.min(rawRight, 1.f), 0.f);
            float clampedBottom = Math.max(Math.min(rawBottom, 1.f), 0.f);

            boxs[number_boxs][0] = clampedLeft * imgWidth;
            boxs[number_boxs][1] = clampedTop * imgHeight;
            boxs[number_boxs][2] = clampedRight * imgWidth;
            boxs[number_boxs][3] = clampedBottom * imgHeight;

            scores[number_boxs] = scoresTensor.getFloatData()[i+1];

            number_boxs = number_boxs + 1 ;
        }

        NMS.sortScores(boxs, scores);

        int[] index = NMS.nmsScoreFilter(boxs, scores, nmsParamTopK, nmsParamNmsThreshold);

        if(index.length>0){
            for(int id: index){
                if(scores[id] < confidenceThreshold) break;

                if(Float.isNaN(scores[id])){//skip the NaN score, maybe not correct
                    continue;
                }
                float[] boxScore = new float[5];

                for(int k=0;k<4;k++)
                    boxScore[k] = boxs[id][k];//x1,y1,x2,y2
                boxScore[4] = scores[id];  //possibility

                boxAndScores.add(boxScore);
            }
        }

        Date end = new Date();
        postprocessTime = (float) (end.getTime() - start.getTime());

        Canvas canvas = new Canvas(outputImage);
        Paint rectPaint = new Paint();
        rectPaint.setStyle(Paint.Style.STROKE);
        rectPaint.setStrokeWidth(2);
        int color = 0xFFFFFF33;
        rectPaint.setColor(color);

        for(float[] boxAndScore: boxAndScores) {
            canvas.drawRect(boxAndScore[0], boxAndScore[1], boxAndScore[2], boxAndScore[3], rectPaint);
        }


    }
    public boolean runModel() {
        if (inputImage == null || !isLoaded()) {
            return false;
        }

        // Set input shape
        boolean pre_val = preProcess();
        if (!pre_val){
            return false;
        }

        // Warm up
        for (int i = 0; i < warmupIterNum; i++) {
            paddlePredictor.run();
        }
        // Run inference
        Date start = new Date();
        for (int i = 0; i < inferIterNum; i++) {
            paddlePredictor.run();
        }
        Date end = new Date();
        inferenceTime = (end.getTime() - start.getTime()) / (float) inferIterNum;

        // Fetch output tensor
        postProcess();
        return true;
    }


    public boolean isLoaded() {
        return paddlePredictor != null && isLoaded;
    }

    public String modelPath() {
        return modelPath;
    }

    public String modelName() {
        return modelName;
    }

    public int cpuThreadNum() {
        return cpuThreadNum;
    }

    public String cpuPowerMode() {
        return cpuPowerMode;
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
        // Scale image to the size of input tensor
        Bitmap rgbaImage = image.copy(Bitmap.Config.ARGB_8888, true);
        Bitmap scaleImage = Bitmap.createScaledBitmap(rgbaImage, (int) inputShape[3], (int) inputShape[2], true);
        this.inputImage = scaleImage;
        this.outputImage = rgbaImage;
    }
}
