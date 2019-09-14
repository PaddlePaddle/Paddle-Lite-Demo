package com.baidu.paddle.lite.demo;

import android.app.ProgressDialog;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Message;
import android.preference.PreferenceManager;
import android.support.v7.app.ActionBar;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;

public class ImgClassifyActivity extends CommonActivity {
    private static final String TAG = ImgClassifyActivity.class.getSimpleName();

    public static final int REQUEST_LOAD_MODEL = 0;
    public static final int REQUEST_RUN_MODEL = 1;

    public static final int RESPONSE_LOAD_MODEL_SUCCESS = 0;
    public static final int RESPONSE_LOAD_MODEL_FAILED = 1;
    public static final int RESPONSE_RUN_MODEL_SUCCESS = 2;
    public static final int RESPONSE_RUN_MODEL_FAILED = 3;

    protected ProgressDialog pbLoadModel = null;
    protected ProgressDialog pbRunModel = null;

    protected TextView tvModelName;
    protected TextView tvInferenceTime;
    protected ImageView ivImageData;
    protected TextView tvTop1Result;
    protected TextView tvTop2Result;
    protected TextView tvTop3Result;

    // model config
    protected String modelPath = "";
    protected String labelPath = "";
    protected String imagePath = "";
    protected long[] inputShape = new long[]{};
    protected float[] inputMean = new float[]{};
    protected float[] inputStd = new float[]{};

    protected ImgClassifyPredictor predictor = new ImgClassifyPredictor();

    protected Handler receiver = null; // receive messages from worker thread
    protected Handler sender = null; // send command to worker thread
    protected HandlerThread worker = null; // worker thread to load&run model

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_img_classify);

        tvModelName = findViewById(R.id.tv_model_name);
        tvInferenceTime = findViewById(R.id.tv_inference_time);
        ivImageData = findViewById(R.id.iv_image_data);
        tvTop1Result = findViewById(R.id.tv_top1_result);
        tvTop2Result = findViewById(R.id.tv_top2_result);
        tvTop3Result = findViewById(R.id.tv_top3_result);

        ActionBar supportActionBar = getSupportActionBar();
        if (supportActionBar != null) {
            supportActionBar.setDisplayHomeAsUpEnabled(true);
        }

        receiver = new Handler() {
            @Override
            public void handleMessage(Message msg) {
                switch (msg.what) {
                    case RESPONSE_LOAD_MODEL_SUCCESS:
                        pbLoadModel.dismiss();
                        // reload test image and run model
                        if (loadImage()) {
                            runModel();
                        }
                        break;
                    case RESPONSE_LOAD_MODEL_FAILED:
                        pbLoadModel.dismiss();
                        Toast.makeText(ImgClassifyActivity.this, "Load model failed!", Toast.LENGTH_SHORT).show();
                        break;
                    case RESPONSE_RUN_MODEL_SUCCESS:
                        pbRunModel.dismiss();
                        // obtain results and update UI
                        outputResult();
                        break;
                    case RESPONSE_RUN_MODEL_FAILED:
                        pbRunModel.dismiss();
                        Toast.makeText(ImgClassifyActivity.this, "Run model failed!", Toast.LENGTH_SHORT).show();
                        break;
                    default:
                        break;
                }
            }
        };

        worker = new HandlerThread("Image Classification Worker");
        worker.start();
        sender = new Handler(worker.getLooper()) {
            public void handleMessage(Message msg) {
                switch (msg.what) {
                    case REQUEST_LOAD_MODEL:
                        // load model and reload test image
                        if (predictor.init(ImgClassifyActivity.this, modelPath, labelPath, inputShape, inputMean,
                                inputStd)) {
                            receiver.sendEmptyMessage(RESPONSE_LOAD_MODEL_SUCCESS);
                        } else {
                            receiver.sendEmptyMessage(RESPONSE_LOAD_MODEL_FAILED);
                        }
                        break;
                    case REQUEST_RUN_MODEL:
                        // run model if model is loaded
                        if (predictor.isLoaded() && predictor.runModel()) {
                            receiver.sendEmptyMessage(RESPONSE_RUN_MODEL_SUCCESS);
                        } else {
                            receiver.sendEmptyMessage(RESPONSE_RUN_MODEL_FAILED);
                        }
                        break;
                    default:
                        break;
                }
            }
        };
    }

    public void loadModel() {
        pbLoadModel = ProgressDialog.show(this, "", "Loading model...", false, false);
        sender.sendEmptyMessage(REQUEST_LOAD_MODEL);
    }

    public void runModel() {
        pbRunModel = ProgressDialog.show(this, "", "Running model...", false, false);
        sender.sendEmptyMessage(REQUEST_RUN_MODEL);
    }

    public boolean loadImage() {
        try {
            if (imagePath.isEmpty()) {
                return false;
            }
            Bitmap imageData = null;
            // read test image file from custom path if the first character of mode path is '/', otherwise read test
            // image file from assets
            if (!imagePath.substring(0, 1).equals("/")) {
                InputStream imageStream = getAssets().open(imagePath);
                imageData = BitmapFactory.decodeStream(imageStream);
            } else {
                if (!new File(imagePath).exists()) {
                    return false;
                }
                imageData = BitmapFactory.decodeFile(imagePath);
            }
            if (imageData != null && predictor.isLoaded()) {
                predictor.setImageData(imageData);
                return true;
            }
        } catch (IOException e) {
            Toast.makeText(ImgClassifyActivity.this, "Load image failed!", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }
        return false;
    }

    public void outputResult() {
        tvModelName.setText("Model: " + predictor.modelName());
        tvInferenceTime.setText("Inference time: " + predictor.inferenceTime() + " ms");
        Bitmap imageData = predictor.imageData();
        if (imageData != null) {
            ivImageData.setImageBitmap(imageData);
        }
        tvTop1Result.setText(predictor.top1Result());
        tvTop2Result.setText(predictor.top2Result());
        tvTop3Result.setText(predictor.top3Result());
    }

    @Override
    public void onImageChanged(Bitmap imageData) {
        // rerun model if users pick test image from gallery or camera
        if (imageData != null && predictor.isLoaded()) {
            predictor.setImageData(imageData);
            runModel();
        }
        super.onImageChanged(imageData);
    }

    public void onSettingsClicked() {
        startActivity(new Intent(ImgClassifyActivity.this, ImgClassifySettingsActivity.class));
        super.onSettingsClicked();
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
        String model_path = sharedPreferences.getString(getString(R.string.ICS_MODEL_PATH_KEY),
                getString(R.string.ICS_MODEL_PATH_DEFAULT));
        String label_path = sharedPreferences.getString(getString(R.string.ICS_LABEL_PATH_KEY),
                getString(R.string.ICS_LABEL_PATH_DEFAULT));
        String image_path = sharedPreferences.getString(getString(R.string.ICS_IMAGE_PATH_KEY),
                getString(R.string.ICS_IMAGE_PATH_DEFAULT));
        settingsChanged |= !model_path.equalsIgnoreCase(modelPath);
        settingsChanged |= !label_path.equalsIgnoreCase(labelPath);
        settingsChanged |= !image_path.equalsIgnoreCase(imagePath);
        long[] input_shape =
                Utils.parseLongsFromString(sharedPreferences.getString(getString(R.string.ICS_INPUT_SHAPE_KEY),
                        getString(R.string.ICS_INPUT_SHAPE_DEFAULT)), ",");
        float[] input_mean =
                Utils.parseFloatsFromString(sharedPreferences.getString(getString(R.string.ICS_INPUT_MEAN_KEY),
                        getString(R.string.ICS_INPUT_MEAN_DEFAULT)), ",");
        float[] input_std =
                Utils.parseFloatsFromString(sharedPreferences.getString(getString(R.string.ICS_INPUT_STD_KEY)
                        , getString(R.string.ICS_INPUT_STD_DEFAULT)), ",");
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
        if (settingsChanged) {
            modelPath = model_path;
            labelPath = label_path;
            imagePath = image_path;
            inputShape = input_shape;
            inputMean = input_mean;
            inputStd = input_std;
            // reload model if configure has been changed
            loadModel();
        }
    }


    @Override
    protected void onDestroy() {
        if (predictor != null) {
            predictor.releaseModel();
        }
        worker.quit();
        super.onDestroy();
    }
}
