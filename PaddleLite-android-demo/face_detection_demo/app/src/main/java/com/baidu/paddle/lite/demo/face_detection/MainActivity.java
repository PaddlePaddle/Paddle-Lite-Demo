package com.baidu.paddle.lite.demo.face_detection;

import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.Menu;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.baidu.paddle.lite.demo.face_detection.config.Config;
import com.baidu.paddle.lite.demo.face_detection.core.CommonActivity;
import com.baidu.paddle.lite.demo.face_detection.preprocess.Preprocess;
import com.baidu.paddle.lite.demo.face_detection.visual.Visualize;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;

public  class MainActivity extends CommonActivity {
    private static final String TAG = MainActivity.class.getSimpleName();

    protected TextView tvInputSetting;
    protected ImageView ivInputImage;
    protected TextView tvOutputResult;
    protected TextView tvInferenceTime;

    // model config
    Config config = new Config();

    protected FaceDetectPredictor predictor = new FaceDetectPredictor();

    Preprocess preprocess = new Preprocess();

    Visualize visualize = new Visualize();

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tvInputSetting = findViewById(R.id.tv_input_setting);
        ivInputImage = findViewById(R.id.iv_input_image);
        tvInferenceTime = findViewById(R.id.tv_inference_time);
        tvOutputResult =  findViewById(R.id.tv_output_result);
        tvInputSetting.setMovementMethod(ScrollingMovementMethod.getInstance());
        tvOutputResult.setMovementMethod(ScrollingMovementMethod.getInstance());
    }


    @Override
    public boolean onLoadModel() {
        return super.onLoadModel() && predictor.init(MainActivity.this, config);
    }

    @Override
    public boolean onRunModel() {
        return super.onRunModel() && predictor.isLoaded() && predictor.runModel(preprocess, visualize);
    }

    @Override
    public void onLoadModelSuccessed() {
        super.onLoadModelSuccessed();
        // load test image from path and run model
        try {
            if (config.imagePath.isEmpty()) {
                return;
            }
            Bitmap image = null;
            // read test image file from custom path if the first character of mode path is '/', otherwise read test
            // image file from assets
            if (!config.imagePath.substring(0, 1).equals("/")) {
                InputStream imageStream = getAssets().open(config.imagePath);
                image = BitmapFactory.decodeStream(imageStream);
                //                this.config.setInputShape(image);
            } else {
                if (!new File(config.imagePath).exists()) {
                    return;
                }
                image = BitmapFactory.decodeFile(config.imagePath);
                //                this.config.setInputShape(image);
            }
            if (image != null && predictor.isLoaded()) {
                //                predictor.setConfig(this.config);
                predictor.setInputImage(image);
                runModel();
            }
        } catch (IOException e) {
            Toast.makeText(MainActivity.this, "Load image failed!", Toast.LENGTH_SHORT).show();
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
            //            predictor.setConfig(config);
            runModel();
        }
    }

    public void onSettingsClicked() {
        super.onSettingsClicked();
        startActivity(new Intent(MainActivity.this, FaceDetectSettingsActivity.class));
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
        Log.i(TAG, "begin onResume");
        super.onResume();

        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);
        boolean settingsChanged = false;
        String model_path = sharedPreferences.getString(getString(R.string.FD_MODEL_PATH_KEY),
                getString(R.string.FD_MODEL_PATH_DEFAULT));
        String label_path = sharedPreferences.getString(getString(R.string.FD_LABEL_PATH_KEY),
                getString(R.string.FD_LABEL_PATH_DEFAULT));
        String image_path = sharedPreferences.getString(getString(R.string.FD_IMAGE_PATH_KEY),
                getString(R.string.FD_IMAGE_PATH_DEFAULT));
        settingsChanged |= !model_path.equalsIgnoreCase(config.modelPath);
        settingsChanged |= !label_path.equalsIgnoreCase(config.labelPath);
        settingsChanged |= !image_path.equalsIgnoreCase(config.imagePath);
        int cpu_thread_num = Integer.parseInt(sharedPreferences.getString(getString(R.string.FD_CPU_THREAD_NUM_KEY),
                getString(R.string.FD_CPU_THREAD_NUM_DEFAULT)));
        settingsChanged |= cpu_thread_num != config.cpuThreadNum;
        String cpu_power_mode =
                sharedPreferences.getString(getString(R.string.FD_CPU_POWER_MODE_KEY),
                        getString(R.string.FD_CPU_POWER_MODE_DEFAULT));
        settingsChanged |= !cpu_power_mode.equalsIgnoreCase(config.cpuPowerMode);
        String input_color_format =
                sharedPreferences.getString(getString(R.string.FD_INPUT_COLOR_FORMAT_KEY),
                        getString(R.string.FD_INPUT_COLOR_FORMAT_DEFAULT));
        settingsChanged |= !input_color_format.equalsIgnoreCase(config.inputColorFormat);
        long[] input_shape =
                Utils.parseLongsFromString(sharedPreferences.getString(getString(R.string.FD_INPUT_SHAPE_KEY),
                        getString(R.string.FD_INPUT_SHAPE_DEFAULT)), ",");
        float[] input_mean =
                Utils.parseFloatsFromString(sharedPreferences.getString(getString(R.string.FD_INPUT_MEAN_KEY),
                        getString(R.string.FD_INPUT_MEAN_DEFAULT)), ",");
        float[] input_std =
                Utils.parseFloatsFromString(sharedPreferences.getString(getString(R.string.FD_INPUT_STD_KEY)
                        , getString(R.string.FD_INPUT_STD_DEFAULT)), ",");
        settingsChanged |= input_shape.length != config.inputShape.length;
        settingsChanged |= input_mean.length != config.inputMean.length;
        settingsChanged |= input_std.length != config.inputStd.length;
        if (!settingsChanged) {
            for (int i = 0; i < input_shape.length; i++) {
                settingsChanged |= input_shape[i] != config.inputShape[i];
            }
            for (int i = 0; i < input_mean.length; i++) {
                settingsChanged |= input_mean[i] != config.inputMean[i];
            }
            for (int i = 0; i < input_std.length; i++) {
                settingsChanged |= input_std[i] != config.inputStd[i];
            }
        }

        if (settingsChanged) {
            config.init(model_path, label_path, image_path, cpu_thread_num, cpu_power_mode,
                    input_color_format, input_shape, input_mean, input_std);
            preprocess.init(config);
            // update UI
            tvInputSetting.setText("Model: " + config.modelPath.substring(config.modelPath.lastIndexOf("/") + 1) + "\n" + "CPU" +
                    " Thread Num: " + Integer.toString(config.cpuThreadNum) + "\n" + "CPU Power Mode: " + config.cpuPowerMode);
            tvInputSetting.scrollTo(0, 0);
            // reload model if configure has been changed
            loadModel();
        }
    }

}
