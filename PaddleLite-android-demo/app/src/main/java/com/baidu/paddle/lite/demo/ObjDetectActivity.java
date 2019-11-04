package com.baidu.paddle.lite.demo;

import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.text.method.ScrollingMovementMethod;
import android.view.Menu;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;

public class ObjDetectActivity extends CommonActivity {
    private static final String TAG = ObjDetectActivity.class.getSimpleName();

    protected TextView tvInputSetting;
    protected ImageView ivInputImage;
    protected TextView tvOutputResult;
    protected TextView tvInferenceTime;

    // model config
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

    protected ObjDetectPredictor predictor = new ObjDetectPredictor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_obj_detect);

        tvInputSetting = findViewById(R.id.tv_input_setting);
        ivInputImage = findViewById(R.id.iv_input_image);
        tvInferenceTime = findViewById(R.id.tv_inference_time);
        tvOutputResult = findViewById(R.id.tv_output_result);
        tvInputSetting.setMovementMethod(ScrollingMovementMethod.getInstance());
        tvOutputResult.setMovementMethod(ScrollingMovementMethod.getInstance());
    }

    @Override
    public boolean onLoadModel() {
        return super.onLoadModel() && predictor.init(ObjDetectActivity.this, modelPath, labelPath, cpuThreadNum,
                cpuPowerMode,
                inputColorFormat,
                inputShape, inputMean,
                inputStd, scoreThreshold);
    }

    @Override
    public boolean onRunModel() {
        return super.onRunModel() && predictor.isLoaded() && predictor.runModel();
    }

    @Override
    public void onLoadModelSuccessed() {
        super.onLoadModelSuccessed();
        // load test image from path and run model
        try {
            if (imagePath.isEmpty()) {
                return;
            }
            Bitmap image = null;
            // read test image file from custom path if the first character of mode path is '/', otherwise read test
            // image file from assets
            if (!imagePath.substring(0, 1).equals("/")) {
                InputStream imageStream = getAssets().open(imagePath);
                image = BitmapFactory.decodeStream(imageStream);
            } else {
                if (!new File(imagePath).exists()) {
                    return;
                }
                image = BitmapFactory.decodeFile(imagePath);
            }
            if (image != null && predictor.isLoaded()) {
                predictor.setInputImage(image);
                runModel();
            }
        } catch (IOException e) {
            Toast.makeText(ObjDetectActivity.this, "Load image failed!", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }
    }

    @Override
    public void onLoadModelFailed() {
        super.onLoadModelFailed();
    }


    @Override
    public void onRunModelSuccessed() {
        super.onRunModelSuccessed();
        // obtain results and update UI
        tvInferenceTime.setText("Inference time: " + predictor.inferenceTime() + " ms");
        Bitmap outputImage = predictor.outputImage();
        if (outputImage != null) {
            ivInputImage.setImageBitmap(outputImage);
        }
        tvOutputResult.setText(predictor.outputResult());
        tvOutputResult.scrollTo(0, 0);
    }

    @Override
    public void onRunModelFailed() {
        super.onRunModelFailed();
    }

    @Override
    public void onImageChanged(Bitmap image) {
        super.onImageChanged(image);
        // rerun model if users pick test image from gallery or camera
        if (image != null && predictor.isLoaded()) {
            predictor.setInputImage(image);
            runModel();
        }
    }

    public void onSettingsClicked() {
        super.onSettingsClicked();
        startActivity(new Intent(ObjDetectActivity.this, ObjDetectSettingsActivity.class));
    }

    @Override
    public boolean onPrepareOptionsMenu(Menu menu) {
        boolean isLoaded = predictor.isLoaded();
        menu.findItem(R.id.open_gallery).setEnabled(isLoaded);
        menu.findItem(R.id.take_photo).setEnabled(isLoaded);
        return super.onPrepareOptionsMenu(menu);
    }

    @Override
    protected void onResume() {
        super.onResume();
        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);
        boolean settingsChanged = false;
        String model_path = sharedPreferences.getString(getString(R.string.ODS_MODEL_PATH_KEY),
                getString(R.string.ODS_MODEL_PATH_DEFAULT));
        String label_path = sharedPreferences.getString(getString(R.string.ODS_LABEL_PATH_KEY),
                getString(R.string.ODS_LABEL_PATH_DEFAULT));
        String image_path = sharedPreferences.getString(getString(R.string.ODS_IMAGE_PATH_KEY),
                getString(R.string.ODS_IMAGE_PATH_DEFAULT));
        settingsChanged |= !model_path.equalsIgnoreCase(modelPath);
        settingsChanged |= !label_path.equalsIgnoreCase(labelPath);
        settingsChanged |= !image_path.equalsIgnoreCase(imagePath);
        int cpu_thread_num = Integer.parseInt(sharedPreferences.getString(getString(R.string.ODS_CPU_THREAD_NUM_KEY),
                getString(R.string.ODS_CPU_THREAD_NUM_DEFAULT)));
        settingsChanged |= cpu_thread_num != cpuThreadNum;
        String cpu_power_mode =
                sharedPreferences.getString(getString(R.string.ODS_CPU_POWER_MODE_KEY),
                        getString(R.string.ODS_CPU_POWER_MODE_DEFAULT));
        settingsChanged |= !cpu_power_mode.equalsIgnoreCase(cpuPowerMode);
        String input_color_format =
                sharedPreferences.getString(getString(R.string.ODS_INPUT_COLOR_FORMAT_KEY),
                        getString(R.string.ODS_INPUT_COLOR_FORMAT_DEFAULT));
        settingsChanged |= !input_color_format.equalsIgnoreCase(inputColorFormat);
        long[] input_shape =
                Utils.parseLongsFromString(sharedPreferences.getString(getString(R.string.ODS_INPUT_SHAPE_KEY),
                        getString(R.string.ODS_INPUT_SHAPE_DEFAULT)), ",");
        float[] input_mean =
                Utils.parseFloatsFromString(sharedPreferences.getString(getString(R.string.ODS_INPUT_MEAN_KEY),
                        getString(R.string.ODS_INPUT_MEAN_DEFAULT)), ",");
        float[] input_std =
                Utils.parseFloatsFromString(sharedPreferences.getString(getString(R.string.ODS_INPUT_STD_KEY)
                        , getString(R.string.ODS_INPUT_STD_DEFAULT)), ",");
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
                Float.parseFloat(sharedPreferences.getString(getString(R.string.ODS_SCORE_THRESHOLD_KEY),
                        getString(R.string.ODS_SCORE_THRESHOLD_DEFAULT)));
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
            // update UI
            tvInputSetting.setText("Model: " + modelPath.substring(modelPath.lastIndexOf("/") + 1) + "\n" + "CPU" +
                    " Thread Num: " + Integer.toString(cpuThreadNum) + "\n" + "CPU Power Mode: " + cpuPowerMode);
            tvInputSetting.scrollTo(0, 0);
            // reload model if configure has been changed
            loadModel();
        }
    }


    @Override
    protected void onDestroy() {
        if (predictor != null) {
            predictor.releaseModel();
        }
        super.onDestroy();
    }
}
