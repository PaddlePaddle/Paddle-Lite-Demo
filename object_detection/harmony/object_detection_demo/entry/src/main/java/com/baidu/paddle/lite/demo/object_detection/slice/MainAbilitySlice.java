package com.baidu.paddle.lite.demo.object_detection.slice;

import com.baidu.paddle.lite.demo.object_detection.ResourceTable;
import com.baidu.paddle.lite.demo.object_detection.PixelMapUtil;
import com.baidu.paddle.lite.demo.object_detection.Utils;
import com.baidu.paddle.lite.demo.object_detection.Predictor;
import ohos.aafwk.ability.AbilitySlice;
import ohos.aafwk.content.Intent;
import ohos.agp.components.Image;
import ohos.agp.components.Text;
import ohos.agp.window.dialog.CommonDialog;
import ohos.agp.window.dialog.ToastDialog;
import ohos.app.Context;
import ohos.data.DatabaseHelper;
import ohos.data.preferences.Preferences;
import ohos.eventhandler.EventHandler;
import ohos.eventhandler.EventRunner;
import ohos.eventhandler.InnerEvent;
import ohos.media.image.PixelMap;

import java.io.File;
import java.io.IOException;

public class MainAbilitySlice extends AbilitySlice {
    private static final String TAG = MainAbilitySlice.class.getSimpleName();
    public static final int OPEN_GALLERY_REQUEST_CODE = 0;
    public static final int TAKE_PHOTO_REQUEST_CODE = 1;

    public static final int REQUEST_LOAD_MODEL = 0;
    public static final int REQUEST_RUN_MODEL = 1;
    public static final int RESPONSE_LOAD_MODEL_SUCCESSED = 0;
    public static final int RESPONSE_LOAD_MODEL_FAILED = 1;
    public static final int RESPONSE_RUN_MODEL_SUCCESSED = 2;
    public static final int RESPONSE_RUN_MODEL_FAILED = 3;
    private static final int TOAST_LENGTH_SHORT = 2000;
    private static final int RESPONSE_LOAD_IMAGE_FAILED = 4;

    protected CommonDialog pbLoadModel = null;
    protected CommonDialog pbRunModel = null;

    protected EventHandler receiver = null; // Receive messages from worker thread
    protected EventHandler sender = null; // Send command to worker thread
    protected EventRunner worker = null; // Worker thread to load&run model

    // UI components of object detection
    protected Text tvInputSetting;
    protected Image ivInputImage;
    protected Text tvOutputResult;
    protected Text tvInferenceTime;

    // Model settings of object detection
    protected String modelPath = "";
    protected String labelPath = "";
    protected String imagePath = "";
    protected int cpuThreadNum = 1;
    protected String cpuPowerMode = "";
    protected String inputColorFormat = "";
    protected long[] inputShape = new long[]{};
    protected float[] inputMean = new float[]{};
    protected float[] inputStd = new float[]{};
    protected float scoreThreshold = 0.5f;

    protected Predictor predictor = new Predictor();

    @Override
    protected void onStart(Intent intent) {
        super.onStart(intent);
        super.setUIContent(ResourceTable.Layout_ability_main);
        // Setup the UI components
        tvInputSetting = findComponentById(ResourceTable.Id_tv_input_setting);
        ivInputImage = findComponentById(ResourceTable.Id_iv_input_image);
        tvInferenceTime = findComponentById(ResourceTable.Id_tv_inference_time);
        tvOutputResult = findComponentById(ResourceTable.Id_tv_output_result);

        receiver = new EventHandler(EventRunner.getMainEventRunner()) {
            @Override
            protected void processEvent(InnerEvent event) {
                switch (event.eventId) {
                    case RESPONSE_LOAD_MODEL_SUCCESSED:
                        pbLoadModel.destroy();//remove 方法会触发某个onRemove方法
                        onLoadModelSucceed();
                        break;
                    case RESPONSE_LOAD_MODEL_FAILED:
                        pbLoadModel.destroy();
                        new ToastDialog(MainAbilitySlice.this).setText("Load model failed!").setDuration(TOAST_LENGTH_SHORT).show();
                        onLoadModelFailed();
                        break;
                    case RESPONSE_RUN_MODEL_SUCCESSED:
                        pbRunModel.destroy();
                        onRunModelSucceed();
                        break;
                    case RESPONSE_RUN_MODEL_FAILED:
                        pbRunModel.destroy();
                        new ToastDialog(MainAbilitySlice.this).setText("Run model failed!").setDuration(TOAST_LENGTH_SHORT).show();
                        onRunModelFailed();
                        break;
                    case RESPONSE_LOAD_IMAGE_FAILED:
                        new ToastDialog(MainAbilitySlice.this).setText("Load image failed!").setDuration(TOAST_LENGTH_SHORT).show();
                        break;
                    default:
                        break;
                }
            }
        };
        worker = EventRunner.create("Predictor Worker");
        sender = new EventHandler(worker) {
            @Override
            protected void processEvent(InnerEvent event) {
                super.processEvent(event);
                switch (event.eventId) {
                    case REQUEST_LOAD_MODEL:
                        // Load model and reload test image
                        if (onLoadModel()) {
                            receiver.sendEvent(RESPONSE_LOAD_MODEL_SUCCESSED);
                        } else {
                            receiver.sendEvent(RESPONSE_LOAD_MODEL_FAILED);
                        }

                        break;
                    case REQUEST_RUN_MODEL:
                        // Run model if model is loaded
                        if (onRunModel()) {
                            receiver.sendEvent(RESPONSE_RUN_MODEL_SUCCESSED);
                        } else {
                            receiver.sendEvent(RESPONSE_RUN_MODEL_FAILED);
                        }
                        break;
                    default:
                        break;
                }
            }
        };
    }

    private boolean onRunModel() {
        return predictor.isLoaded() && predictor.runModel();
    }

    private boolean onLoadModel() {
        return predictor.init(MainAbilitySlice.this, modelPath, labelPath, cpuThreadNum,
                cpuPowerMode,
                inputColorFormat,
                inputShape, inputMean,
                inputStd, scoreThreshold);
    }

    private void onRunModelFailed() {

    }

    private void onRunModelSucceed() {
        // Obtain results and update UI
        tvInferenceTime.setText("Inference time: " + predictor.inferenceTime() + " ms");
        PixelMap outputImage = predictor.outputImage();
        if (outputImage != null) {
            ivInputImage.setPixelMap(outputImage);
        }
        tvOutputResult.setText(predictor.outputResult());
        tvOutputResult.scrollTo(0, 0);
    }

    private void onLoadModelFailed() {
    }

    private void onLoadModelSucceed() {
        // Load test image from path and run model
        if (imagePath.isEmpty()) {
            return;
        }
        PixelMap image = null;
        // Read test image file from custom path if the first character of mode path is '/', otherwise read test
        // image file from assets
        try {
            if (!imagePath.substring(0, 1).equals("/")) {

                image = PixelMapUtil.getPixelMapByResPath(this, imagePath);

            } else {
                if (!new File(imagePath).exists()) {
                    return;
                }

                try {
                    image = PixelMapUtil.getPixelMapByFilePath(this, imagePath);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

            if (image != null && predictor.isLoaded()) {
                predictor.setInputImage(image);
                runModel();
            }
        } catch (IOException e) {
            new ToastDialog(this).setContentText("Load image failed!")
                    .setDuration(TOAST_LENGTH_SHORT).show();
            e.printStackTrace();
        }
    }

    public void runModel() {
        pbRunModel = new CommonDialog(this).setContentText("Running model...");
        pbRunModel.show();

        sender.sendEvent(REQUEST_RUN_MODEL);
    }

    @Override
    protected void onActive() {
        super.onActive();
        Context context = getApplicationContext();
        DatabaseHelper databaseHelper = new DatabaseHelper(context); // context入参类型为ohos.app.Context。
        String fileName = "test_pref"; ///data/data/{PackageName}/preferences。  todo fileName是什么
        Preferences preferences = databaseHelper.getPreferences(fileName);
        boolean settingsChanged = false;
        String model_path = preferences.getString(getString(ResourceTable.String_MODEL_PATH_KEY),
                getString(ResourceTable.String_MODEL_PATH_DEFAULT));
        String label_path = preferences.getString(getString(ResourceTable.String_LABEL_PATH_KEY),
                getString(ResourceTable.String_LABEL_PATH_DEFAULT));
        String image_path = preferences.getString(getString(ResourceTable.String_IMAGE_PATH_KEY),
                getString(ResourceTable.String_IMAGE_PATH_DEFAULT));
        settingsChanged |= !model_path.equalsIgnoreCase(modelPath);
        settingsChanged |= !label_path.equalsIgnoreCase(labelPath);
        settingsChanged |= !image_path.equalsIgnoreCase(imagePath);
        int cpu_thread_num = Integer.parseInt(preferences.getString(getString(ResourceTable.String_CPU_THREAD_NUM_KEY),
                getString(ResourceTable.String_CPU_THREAD_NUM_DEFAULT)));
        settingsChanged |= cpu_thread_num != cpuThreadNum;
        String cpu_power_mode =
                preferences.getString(getString(ResourceTable.String_CPU_POWER_MODE_KEY),
                        getString(ResourceTable.String_CPU_POWER_MODE_DEFAULT));
        settingsChanged |= !cpu_power_mode.equalsIgnoreCase(cpuPowerMode);
        String input_color_format =
                preferences.getString(getString(ResourceTable.String_INPUT_COLOR_FORMAT_KEY),
                        getString(ResourceTable.String_INPUT_COLOR_FORMAT_DEFAULT));
        settingsChanged |= !input_color_format.equalsIgnoreCase(inputColorFormat);
        long[] input_shape =
                Utils.parseLongsFromString(preferences.getString(getString(ResourceTable.String_INPUT_SHAPE_KEY),
                        getString(ResourceTable.String_INPUT_SHAPE_DEFAULT)), ",");
        float[] input_mean =
                Utils.parseFloatsFromString(preferences.getString(getString(ResourceTable.String_INPUT_MEAN_KEY),
                        getString(ResourceTable.String_INPUT_MEAN_DEFAULT)), ",");
        float[] input_std =
                Utils.parseFloatsFromString(preferences.getString(getString(ResourceTable.String_INPUT_STD_KEY)
                        , getString(ResourceTable.String_INPUT_STD_DEFAULT)), ",");
        settingsChanged |= input_shape.length != inputShape.length;
        settingsChanged |= input_mean.length != inputMean.length;
        settingsChanged |= input_std.length != inputStd.length;
        if (!settingsChanged) {
            for (int i = 0; i < input_shape.length; i++) {
                settingsChanged |= input_shape[i] != inputShape[i];
            }
            for (int i = 0; i < input_mean.length; i++) {
                settingsChanged |= input_mean[i] != inputMean[i];
            }
            for (int i = 0; i < input_std.length; i++) {
                settingsChanged |= input_std[i] != inputStd[i];
            }
        }
        float score_threshold =
                Float.parseFloat(preferences.getString(getString(ResourceTable.String_SCORE_THRESHOLD_KEY),
                        getString(ResourceTable.String_SCORE_THRESHOLD_DEFAULT)));
        settingsChanged |= scoreThreshold != score_threshold;
        if (settingsChanged) {
            modelPath = model_path;
            labelPath = label_path;
            imagePath = image_path;
            cpuThreadNum = cpu_thread_num;
            cpuPowerMode = cpu_power_mode;
            inputColorFormat = input_color_format;
            inputShape = input_shape;
            inputMean = input_mean;
            inputStd = input_std;
            scoreThreshold = score_threshold;
            // Update UI
            tvInputSetting.setText("Model: " + modelPath.substring(modelPath.lastIndexOf("/") + 1) + "\n" + "CPU" +
                    " Thread Num: " + Integer.toString(cpuThreadNum) + "\n" + "CPU Power Mode: " + cpuPowerMode);
            tvInputSetting.scrollTo(0, 0);
            // Reload model if configure has been changed
            loadModel();
        }
    }

    private void loadModel() {
        pbLoadModel = new CommonDialog(this).setContentText("Loading model...");
        pbLoadModel.show();
        sender.sendEvent(REQUEST_LOAD_MODEL);
    }
}
