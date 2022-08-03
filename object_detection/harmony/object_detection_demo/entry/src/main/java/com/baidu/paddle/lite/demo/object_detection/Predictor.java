package com.baidu.paddle.lite.demo.object_detection;


import com.baidu.paddle.lite.MobileConfig;
import com.baidu.paddle.lite.PaddlePredictor;
import com.baidu.paddle.lite.PowerMode;
import com.baidu.paddle.lite.Tensor;
import ohos.agp.colors.RgbColor;
import ohos.agp.render.Canvas;
import ohos.agp.render.Paint;
import ohos.agp.render.Texture;
import ohos.agp.utils.Color;
import ohos.agp.utils.RectFloat;
import ohos.app.Context;
import ohos.media.image.PixelMap;
import ohos.media.image.common.PixelFormat;
import ohos.media.image.common.Position;
import ohos.media.image.common.Size;
import ohos.miscservices.timeutility.Time;

import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.Vector;

public class Predictor {
    static {
        System.loadLibrary("paddle_lite_jni");
    }
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
    private static final Logger Log = new Logger();

    private ArrayList<Recognition> recognitionList;
    private Size mOrinSize;
    private Context mContext;

    public Vector<String> getWordLabels() {
        return wordLabels;
    }

    // Only for object detection
    protected Vector<String> wordLabels = new Vector<String>();
    protected String inputColorFormat = "RGB";
    protected long[] inputShape = new long[]{1, 3, 300, 300};
    protected float[] inputMean = new float[]{0.5f, 0.5f, 0.5f};
    protected float[] inputStd = new float[]{0.5f, 0.5f, 0.5f};
    protected float scoreThreshold = 0.5f;
    protected PixelMap inputImage = null;
    protected PixelMap outputImage = null;
    protected String outputResult = "";
    protected float preprocessTime = 0;
    protected float postprocessTime = 0;
    private Tensor outputTensor;

    public Predictor() {
    }

    public boolean init(Context appCtx, String modelPath, String labelPath, int cpuThreadNum, String cpuPowerMode,
                        String inputColorFormat,
                        long[] inputShape, float[] inputMean,
                        float[] inputStd, float scoreThreshold) {
        mContext = appCtx;
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
        this.scoreThreshold = scoreThreshold;
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
            Log.e(TAG, "unknown cpu power mode!");
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
            Log.i(TAG, "start load label");
            InputStream assetsInputStream = appCtx.getResourceManager().getRawFileEntry(labelPath).openRawFile();
            int available = assetsInputStream.available();
            byte[] lines = new byte[available];
            assetsInputStream.read(lines);
            assetsInputStream.close();
            String words = new String(lines);
            String[] contents = words.split("\n");
            for (String content : contents) {
                wordLabels.add(content);
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

    public ArrayList<Recognition> recognize(String className) {
        if (inputImage == null || !isLoaded()) {
            return null;
        }
        long startTime = Time.getRealTime();
        // Set input shape
        Tensor inputTensor = getInput(0);
        inputTensor.resize(inputShape);

        // Pre-process image, and feed input tensor with pre-processed data

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
                return null;
            }
            int[] channelStride = new int[]{width * height, width * height * 2};
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int colorInt = inputImage.readPixel(new Position(x,y));
                    RgbColor color = new RgbColor(colorInt);
                    float[] rgb = new float[]{(float) color.getRed() / 255.0f, (float) color.getGreen() / 255.0f,
                            (float) color.getBlue() / 255.0f};
                    inputData[y * width + x] = (rgb[channelIdx[0]] - inputMean[0]) / inputStd[0];
                    inputData[y * width + x + channelStride[0]] = (rgb[channelIdx[1]] - inputMean[1]) / inputStd[1];
                    inputData[y * width + x + channelStride[1]] = (rgb[channelIdx[2]] - inputMean[2]) / inputStd[2];
                }
            }
        } else if (channels == 1) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int colorInt = inputImage.readPixel(new Position(x,y));
                    RgbColor color = new RgbColor(colorInt);
                    float gray = (float) (color.getRed() + color.getGreen() + color.getBlue()) / 3.0f / 255.0f;
                    inputData[y * width + x] = (gray - inputMean[0]) / inputStd[0];
                }
            }
        } else {
            Log.i(TAG, "Unsupported channel size " + Integer.toString(channels) + ",  only channel 1 and 3 is " +
                    "supported!");
            return null;
        }
        inputTensor.setData(inputData);
        Date end = new Date();
        preprocessTime = (float) (end.getTime() - start.getTime());

        // Warm up
        for (int i = 0; i < warmupIterNum; i++) {
            paddlePredictor.run();
        }
        // Run inference
        start = new Date();

        for (int i = 0; i < inferIterNum; i++) {
            paddlePredictor.run();
        }
        end = new Date();
        inferenceTime = (end.getTime() - start.getTime()) / (float) inferIterNum;

        // Fetch output tensor
        outputTensor = getOutput(0);//结果保存在这里

        // Post-process

        long outputShape[] = outputTensor.shape();
        long outputSize = 1;
        for (long s : outputShape) {
            outputSize *= s;
        }

        ArrayList<Recognition> recognitionList = new ArrayList<Recognition>();
        outputImage = inputImage;
        Canvas canvas = new Canvas();
        Paint rectPaint = new Paint();
        rectPaint.setStyle(Paint.Style.STROKE_STYLE);
        rectPaint.setStrokeWidth(1);
        Paint txtPaint = new Paint();
        txtPaint.setTextSize(12);
        txtPaint.setAntiAlias(true);
        int txtXOffset = 4;
        int txtYOffset = (int) (Math.ceil(-txtPaint.getFontMetrics().ascent));
        int imgWidth = outputImage.getImageInfo().size.width;
        int imgHeight = outputImage.getImageInfo().size.width;
//        int imgWidth = mOrinSize.width;
//        int imgHeight = mOrinSize.height;
        int objectIdx = 0;
        final int[] objectColor = {0xFFFF00CC, 0xFFFF0000, 0xFFFFFF33, 0xFF0000FF, 0xFF00FF00,
                0xFF000000, 0xFF339933};
        for (int i = 0; i < outputSize; i += 6) {
            float score = outputTensor.getFloatData()[i + 1];
            if (score < scoreThreshold) {
                continue;
            }

            int categoryIdx = (int) outputTensor.getFloatData()[i]; // softmax前面是分类id，后面是分数
            String categoryName = null;
            if (wordLabels.size() > 0 && categoryIdx >= 0 && categoryIdx < wordLabels.size() && className.equals(wordLabels.get(categoryIdx))) { //这里可以再简化 todo
                categoryName = wordLabels.get(categoryIdx);
            } else {
                continue;
            }

            float rawLeft = outputTensor.getFloatData()[i + 2];
            float clampedLeft = Math.max(Math.min(rawLeft, 1.f), 0.f);
            float imgLeft = clampedLeft * imgWidth;

            float rawTop = outputTensor.getFloatData()[i + 3];
            float clampedTop =  Math.max(Math.min(rawTop, 1.f), 0.f);
            float imgTop = clampedTop * imgWidth;

            float rawRight = outputTensor.getFloatData()[i + 4];
            float clampedRight = Math.max(Math.min(rawRight, 1.f), 0.f);
            float imgRight = clampedRight * imgHeight;

            float rawBottom = outputTensor.getFloatData()[i + 5];
            float clampedBottom = Math.max(Math.min(rawBottom, 1.f), 0.f);
            float imgBottom = clampedBottom * imgHeight;
            int color = objectColor[objectIdx % objectColor.length];
            RectFloat rectF = new RectFloat(clampedLeft, clampedTop, clampedRight, clampedBottom);
            rectPaint.setColor(new Color(color));
            txtPaint.setColor(new Color(color));
//            canvas.drawPixelMapHolder(new PixelMapHolder(outputImage), 0, 0, rectPaint);//究竟是texture还是paint还没搞清楚
            canvas.setTexture(new Texture(outputImage));
            canvas.drawRect(imgLeft, imgTop, imgRight, imgBottom, rectPaint);
            canvas.drawText(txtPaint, objectIdx + "." + categoryName + ":" + String.format("%.3f", score),
                    imgLeft + txtXOffset, imgTop + txtYOffset);

            Recognition recognition = new Recognition(
                    objectIdx + "", //这个没搞清楚是啥
                    categoryName,
                    score,
                    rectF,
                    categoryIdx
            );
            recognitionList.add(recognition);
            outputResult += objectIdx + "." + categoryName + " - " + String.format("%.3f", score) +
                    " [" + String.format("%.3f", rawLeft) + "," + String.format("%.3f", rawTop) + "," + String.format("%.3f", rawRight) + "," + String.format("%.3f", rawBottom) + "]\n";
            objectIdx++;
        }
        setRecognitionList(recognitionList);
        postprocessTime = Time.getRealTime() - startTime;
        PixelMapUtil.savePixelMap(mContext, inputImage, "IMG_"+new Date()+".jpg");
        return recognitionList;
    }

    public boolean runModel() {
        if (inputImage == null || !isLoaded()) {
            return false;
        }

        // Set input shape
        Tensor inputTensor = getInput(0);
        inputTensor.resize(inputShape);

        // Pre-process image, and feed input tensor with pre-processed data
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
                    int colorInt = inputImage.readPixel(new Position(x,y));
                    RgbColor color = new RgbColor(colorInt);
                    float[] rgb = new float[]{(float) color.getRed() / 255.0f, (float) color.getGreen() / 255.0f,
                            (float) color.getBlue() / 255.0f};
                    inputData[y * width + x] = (rgb[channelIdx[0]] - inputMean[0]) / inputStd[0];
                    inputData[y * width + x + channelStride[0]] = (rgb[channelIdx[1]] - inputMean[1]) / inputStd[1];
                    inputData[y * width + x + channelStride[1]] = (rgb[channelIdx[2]] - inputMean[2]) / inputStd[2];
                }
            }
        } else if (channels == 1) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int colorInt = inputImage.readPixel(new Position(x,y));
                    RgbColor color = new RgbColor(colorInt);
                    float gray = (float) (color.getRed() + color.getGreen() + color.getBlue()) / 3.0f / 255.0f;
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

        // Warm up
        for (int i = 0; i < warmupIterNum; i++) {
            paddlePredictor.run();
        }
        // Run inference
        start = new Date();
        for (int i = 0; i < inferIterNum; i++) {
            paddlePredictor.run();
        }
        end = new Date();
        inferenceTime = (end.getTime() - start.getTime()) / (float) inferIterNum;

        // Fetch output tensor
        Tensor outputTensor = getOutput(0);

        // Post-process
        start = new Date();
        long outputShape[] = outputTensor.shape();
        long outputSize = 1;
        for (long s : outputShape) {
            outputSize *= s;
        }
        outputImage = inputImage;
        outputResult = new String();

        Canvas canvas = new Canvas(new Texture(outputImage));
        Paint rectPaint = new Paint();
        rectPaint.setStyle(Paint.Style.STROKE_STYLE);
        rectPaint.setStrokeWidth(1);
        Paint txtPaint = new Paint();
        txtPaint.setTextSize(12);
        txtPaint.setAntiAlias(true);
        int txtXOffset = 4;
        int txtYOffset = (int) (Math.ceil(-txtPaint.getFontMetrics().ascent));
        int imgWidth = outputImage.getImageInfo().size.width;
        int imgHeight = outputImage.getImageInfo().size.height;
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
            float rawTop = outputTensor.getFloatData()[i + 3];
            float rawLeft = outputTensor.getFloatData()[i + 2];
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
            rectPaint.setColor(new Color(color));
            txtPaint.setColor(new Color(color));
            canvas.drawRect(imgLeft, imgTop, imgRight, imgBottom, rectPaint);
            canvas.drawText(txtPaint, objectIdx + "." + categoryName + ":" + String.format("%.3f", score),
                    imgLeft + txtXOffset, imgTop + txtYOffset);
            outputResult += objectIdx + "." + categoryName + " - " + String.format("%.3f", score) +
                    " [" + String.format("%.3f", rawLeft) + "," + String.format("%.3f", rawTop) + "," + String.format("%.3f", rawRight) + "," + String.format("%.3f", rawBottom) + "]\n";
            objectIdx++;
        }
        end = new Date();
        postprocessTime = (float) (end.getTime() - start.getTime());
        return true;
    }

    public ArrayList<Recognition> getRecognitionList() {
        return recognitionList;
    }

    private void setRecognitionList(ArrayList<Recognition> recognitionList) {
        this.recognitionList = recognitionList;
    }


    public Tensor getOutputTensor() {
        return outputTensor;
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

    public PixelMap inputImage() {
        return inputImage;
    }

    public PixelMap outputImage() {
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


    public void setInputImage(PixelMap image) {
        if (image == null) {
            return;
        }
        // Scale image to the size of input tensor
//        PixelMap rgbaImage = image.copy(PixelMap.Config.ARGB_8888, true);
        PixelMap.InitializationOptions rgbaOpt = new PixelMap.InitializationOptions();
        rgbaOpt.pixelFormat = PixelFormat.ARGB_8888;
        PixelMap rgbaImage = PixelMap.create(image, rgbaOpt);

        PixelMap.InitializationOptions ops = new PixelMap.InitializationOptions();
        ops.size = new Size((int)inputShape[3], (int)inputShape[2]);
        PixelMap scaleImage = PixelMap.create(rgbaImage,ops);
        this.inputImage = scaleImage;
    }
}
