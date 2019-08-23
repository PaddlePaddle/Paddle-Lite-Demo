package com.baidu.paddle.lite.demo;

import android.Manifest;
import android.content.ContentResolver;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.RadioGroup;
import android.widget.RadioGroup.OnCheckedChangeListener;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.io.InputStream;

public class ImgClassifyActivity extends AppCompatActivity {
    private static final String TAG = ImgClassifyActivity.class.getSimpleName();
    public static final int GALLERY_REQUEST_CODE = 0;
    public static final int IMAGE_CAPTURE_REQUEST_CODE = 1;

    protected TextView tvModelName;
    private RadioGroup rgChooseDevice;
    protected Button btnGallery;
    protected Button btnCamera;
    protected ImageView ivImageData;
    protected TextView tvTop1Result;
    protected TextView tvTop2Result;
    protected TextView tvTop3Result;
    protected TextView tvInferenceTime;

    // model info
    public static final long modelInputWidth = 224;
    public static final long modelInputHeight = 224;
    public static final String modelFilePath = "image_classification/models/mobilenet_v1";
    public static final String modelLabelPath = "image_classification/labels/synset_words.txt";
    public static final String imageFilePath = "image_classification/images/egypt_cat.jpg";

    protected ImgClassifyPredictor predictor = new ImgClassifyPredictor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_img_classify);

        btnGallery = findViewById(R.id.btn_gallery);
        btnCamera = findViewById(R.id.btn_camera);
        tvModelName = findViewById(R.id.tv_model_name);
        rgChooseDevice = findViewById(R.id.rg_choose_device);
        ivImageData = findViewById(R.id.iv_image_data);
        tvTop1Result = findViewById(R.id.tv_top1_result);
        tvTop2Result = findViewById(R.id.tv_top2_result);
        tvTop3Result = findViewById(R.id.tv_top3_result);
        tvInferenceTime = findViewById(R.id.tv_inference_time);

        ActionBar supportActionBar = getSupportActionBar();
        if (supportActionBar != null) {
            supportActionBar.setDisplayHomeAsUpEnabled(true);
        }

        btnGallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                checkStoragePermission();
            }
        });

        btnCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                checkCameraPermission();
            }
        });

        rgChooseDevice.setOnCheckedChangeListener(new OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                if (predictor != null && predictor.isLoaded()) {
                    switch (group.getCheckedRadioButtonId()) {
                        case R.id.rb_on_cpu:
                            predictor.useCPU();
                            break;
                        case R.id.rb_on_npu:
                            predictor.useNPU();
                            break;
                        default:
                            break;
                    }
                    if (predictor.runModel()) {
                        updateUI();
                    }
                }
            }
        });
        rgChooseDevice.getChildAt(1).setEnabled(Utils.isSupportNPU());

        // load model
        predictor.init(this, modelFilePath, modelLabelPath, modelInputWidth, modelInputHeight);

        // load test image and run model
        if (predictor != null && predictor.isLoaded()) {
            try {
                InputStream is = getAssets().open(imageFilePath);
                Bitmap image = BitmapFactory.decodeStream(is);
                if (image != null) {
                    if (predictor.runModel(image)) {
                        updateUI();
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public void updateUI() {
        tvModelName.setText("model: " + predictor.modelName());
        Bitmap imageData = predictor.imageData();
        if (imageData != null) {
            ivImageData.setImageBitmap(imageData);
        }
        tvTop1Result.setText(predictor.top1Result());
        tvTop2Result.setText(predictor.top2Result());
        tvTop3Result.setText(predictor.top3Result());
        tvInferenceTime.setText("inference time: " + predictor.inferenceTime() + " ms");
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case android.R.id.home:
                finish();
                break;
        }
        return super.onOptionsItemSelected(item);
    }

    private void checkStoragePermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED &&
                ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                        != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.CAMERA},
                    GALLERY_REQUEST_CODE);
        } else {
            chooseImageAndRunModel();
        }
    }

    private void checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED &&
                ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                        != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.CAMERA},
                    IMAGE_CAPTURE_REQUEST_CODE);
        } else {
            takePhotoAndRunModel();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == GALLERY_REQUEST_CODE &&
                grantResults[1] == PackageManager.PERMISSION_GRANTED) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                chooseImageAndRunModel();
            } else {
                Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show();
            }
        }

        if (requestCode == IMAGE_CAPTURE_REQUEST_CODE) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED &&
                    grantResults[1] == PackageManager.PERMISSION_GRANTED) {
                takePhotoAndRunModel();
            } else {
                Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void takePhotoAndRunModel() {
        Intent takePhotoIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePhotoIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePhotoIntent, IMAGE_CAPTURE_REQUEST_CODE);
        }
    }

    private void chooseImageAndRunModel() {
        Intent intent = new Intent(Intent.ACTION_PICK, null);
        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(intent, GALLERY_REQUEST_CODE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && data != null) {
            switch (requestCode) {
                case GALLERY_REQUEST_CODE:
                    try {
                        ContentResolver resolver = getContentResolver();
                        Uri originalUri = data.getData();
                        Bitmap image = MediaStore.Images.Media.getBitmap(resolver, originalUri);
                        String[] proj = {MediaStore.Images.Media.DATA};
                        Cursor cursor = managedQuery(originalUri, proj, null, null, null);
                        cursor.moveToFirst();
                        if (predictor != null && predictor.isLoaded() && image != null) {
                            if (predictor.runModel(image)) {
                                updateUI();
                            }
                        }
                    } catch (IOException e) {
                        Log.e(TAG, e.toString());
                    }
                    break;
                case IMAGE_CAPTURE_REQUEST_CODE:
                    Bundle extras = data.getExtras();
                    Bitmap image = (Bitmap) extras.get("data");
                    if (predictor != null && predictor.isLoaded() && image != null) {
                        if (predictor.runModel(image)) {
                            updateUI();
                        }
                    }
                    break;
                default:
                    break;
            }
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

