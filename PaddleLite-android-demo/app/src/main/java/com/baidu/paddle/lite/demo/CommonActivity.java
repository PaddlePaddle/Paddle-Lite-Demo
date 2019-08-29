package com.baidu.paddle.lite.demo;

import android.Manifest;
import android.content.ContentResolver;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.widget.Toast;

import java.io.IOException;

public class CommonActivity extends AppCompatActivity {
    private static final String TAG = CommonActivity.class.getSimpleName();
    public static final int OPEN_GALLERY_REQUEST_CODE = 0;
    public static final int TAKE_PHOTO_REQUEST_CODE = 1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        ActionBar supportActionBar = getSupportActionBar();
        if (supportActionBar != null) {
            supportActionBar.setDisplayHomeAsUpEnabled(true);
        }

    }

    public void onImageChanged(Bitmap image) {
    }

    public void onDeviceChanged(MenuItem item) {
    }

    public void onSettingsClicked() {
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu_action_options, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case android.R.id.home:
                finish();
                break;
            case R.id.open_gallery:
                if (requestAllPermissions()) {
                    openGallery();
                }
                break;
            case R.id.take_photo:
                if (requestAllPermissions()) {
                    takePhoto();
                }
                break;
            case R.id.run_on_cpu:
                if (!item.isChecked()) {
                    onDeviceChanged(item);
                    item.setChecked(true);
                }
                break;
            case R.id.run_on_npu:
                if (!item.isChecked()) {
                    onDeviceChanged(item);
                    item.setChecked(true);
                }
                break;
            case R.id.settings:
                if (requestAllPermissions()) {
                    // make sure we have SDCard r&w permissions to load model from SDCard
                    onSettingsClicked();
                }
                break;
        }
        return super.onOptionsItemSelected(item);
    }

    @Override
    public boolean onPrepareOptionsMenu(Menu menu) {
        menu.findItem(R.id.run_on_npu).setEnabled(Utils.isSupportedNPU());
        if (!Utils.isSupportedNPU()) {
            Toast.makeText(CommonActivity.this, "Kirin 810's NPU is the only supported NPU in Paddle-Lite, but we " +
                    "will add support for more HUAWEI NPUs in the future!", Toast.LENGTH_LONG).show();
        }
        return super.onPrepareOptionsMenu(menu);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED || grantResults[1] != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show();
        }
    }

    private boolean requestAllPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED || ContextCompat.checkSelfPermission(this,
                Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE,
                            Manifest.permission.CAMERA},
                    0);
            return false;
        }
        return true;
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, null);
        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(intent, OPEN_GALLERY_REQUEST_CODE);
    }

    private void takePhoto() {
        Intent takePhotoIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePhotoIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePhotoIntent, TAKE_PHOTO_REQUEST_CODE);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && data != null) {
            switch (requestCode) {
                case OPEN_GALLERY_REQUEST_CODE:
                    try {
                        ContentResolver resolver = getContentResolver();
                        Uri originalUri = data.getData();
                        Bitmap imageData = MediaStore.Images.Media.getBitmap(resolver, originalUri);
                        String[] proj = {MediaStore.Images.Media.DATA};
                        Cursor cursor = managedQuery(originalUri, proj, null, null, null);
                        cursor.moveToFirst();
                        onImageChanged(imageData);
                    } catch (IOException e) {
                        Log.e(TAG, e.toString());
                    }
                    break;
                case TAKE_PHOTO_REQUEST_CODE:
                    Bundle extras = data.getExtras();
                    Bitmap imageData = (Bitmap) extras.get("data");
                    onImageChanged(imageData);
                    break;
                default:
                    break;
            }
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
    }
}
