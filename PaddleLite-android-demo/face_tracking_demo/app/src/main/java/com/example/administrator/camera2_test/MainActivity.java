package com.example.administrator.camera2_test;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;

import com.baidu.paddle.lite.MobileConfig;
import com.baidu.paddle.lite.PaddlePredictor;
import com.baidu.paddle.lite.PowerMode;
import com.baidu.paddle.lite.Tensor;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import static android.graphics.Color.blue;
import static android.graphics.Color.green;
import static android.graphics.Color.red;

class Face{
    public int id;
    public int valid;
    public int trackCount;
    public float xmin;
    public float xmax;
    public float ymin;
    public float ymax;

    public Face(int id, int valid, int trackCount, float xmin, float xmax, float ymin, float ymax) {
        this.id = id;
        this.valid = valid;
        this.xmin = xmin;
        this.xmax = xmax;
        this.ymin = ymin;
        this.ymax = ymax;
    }
}
public class MainActivity extends AppCompatActivity {
    private static final int PERMISSIONS_REQUEST_CODE = 0 ;
    private TextureView mTextureView = null;
    private TextureView mTextureView_result = null;
    private TextureView mTextureView_face[] = new TextureView[3];
    private Size mPreviewSize = null;
    String mCameraId;
    private CameraDevice mCameraDevice = null;
    private CaptureRequest.Builder mCaptureRequestBuilder;
    private ImageReader mImageReader;
    private Rect mSrcRect = new Rect();
    private Rect mDstRect = new Rect();
    private Paint mPaint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.DITHER_FLAG);
    private MobileConfig config;
    private String modelpath;
    private PaddlePredictor predictor;
    private Face face[] = new Face[3];
    private int nowId;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mTextureView = (TextureView) findViewById(R.id.texture_view);
        mTextureView_result = (TextureView) findViewById(R.id.texture_view_output);

        mTextureView_face[0] = (TextureView) findViewById(R.id.texture_view_face1);
        mTextureView_face[1] = (TextureView) findViewById(R.id.texture_view_face2);
        mTextureView_face[2] = (TextureView) findViewById(R.id.texture_view_face3);
        face[0]=new Face(-1,0,0,0,0,0,0);
        face[1]=new Face(-1,0,0,0,0,0,0);
        face[2]=new Face(-1,0,0,0,0,0,0);//暂时设定最多跟踪3个脸
        mTextureView.setSurfaceTextureListener(textureListener);
        nowId=0;

        config = new MobileConfig();
        modelpath = getModelPath(this);
        config.setModelDir(modelpath);
        config.setPowerMode(PowerMode.LITE_POWER_HIGH);
        config.setThreads(1);
        predictor = PaddlePredictor.createPaddlePredictor(config);

    }
    TextureView.SurfaceTextureListener textureListener = new TextureView.SurfaceTextureListener(){
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height){
            setupCamera(width, height);
            openCamera();
        }
        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture texture, int width, int height){
        }
        public boolean onSurfaceTextureDestroyed(SurfaceTexture texture)
        {
            return true;
        }
        public void onSurfaceTextureUpdated(SurfaceTexture texture){
        }
    };
    private void setupCamera(int width, int height){
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try{
            for (String cameraId: manager.getCameraIdList()){
                CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
                if(characteristics.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_FRONT)
                    continue;
                StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                mPreviewSize = getOptimalSize(map.getOutputSizes(SurfaceTexture.class), width, height);
                mCameraId = cameraId;
            }
        }catch (CameraAccessException e)
        {
            e.printStackTrace();
        }
    }
    private Size getOptimalSize(Size[] sizeMap, int width, int height) {
        List<Size> sizeList = new ArrayList<>();
        for (Size option : sizeMap){
            if (width > height){
                if (option.getWidth() > width && option.getHeight() > height){
                    sizeList.add(option);
                }
            }else{
                if (option.getWidth() > height && option.getHeight() > width)
                {
                    sizeList.add(option);
                }
            }
        }
        if (sizeList.size() > 0){
            return Collections.min(sizeList, new Comparator<Size>(){
                @Override
                public int compare(Size lhs, Size rhs){
                    return Long.signum(lhs.getWidth() * lhs.getHeight() - rhs.getWidth() * rhs.getHeight());
                }
            });
        }
        return sizeMap[0];
    }

    private void openCamera(){
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try
        {
            if(ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED)
            {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA},PERMISSIONS_REQUEST_CODE);
            }
            manager.openCamera(mCameraId, stateCallback, null);
        }
        catch (CameraAccessException e)
        {
            e.printStackTrace();
        }
    }

    private final CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback(){
        @Override
        public void onOpened(CameraDevice camera){
            mCameraDevice = camera;
            setupImageReader();
            startPreview();
        }
        @Override
        public void onDisconnected(CameraDevice camera){
            camera.close();
            mCameraDevice = null;
        }
        @Override
        public void onError(CameraDevice camera, int error){
            camera.close();
            mCameraDevice = null;
        }
    };
    private void setupImageReader()
    {
        mImageReader = ImageReader.newInstance(mPreviewSize.getWidth(), mPreviewSize.getHeight(),ImageFormat.JPEG, 2);
        mImageReader.setOnImageAvailableListener(new ImageReader.OnImageAvailableListener()
        {
            @Override
            public void onImageAvailable(ImageReader reader)
            {
                //获得原始图像，并旋转缩放-----------
                Image image = reader.acquireLatestImage();
                if(image == null)
                    return;
                ByteBuffer buffer = image.getPlanes()[0].getBuffer();
                byte[] data = new byte[buffer.remaining()];
                buffer.get(data);
                Bitmap bitmapOrigin = BitmapFactory.decodeByteArray(data, 0, data.length);
                Matrix matrix = new Matrix();
                matrix.setRotate(90);
                Bitmap bitmapRotate = Bitmap.createBitmap(bitmapOrigin,0,0,bitmapOrigin.getWidth(),bitmapOrigin.getHeight(),matrix,false);
                Matrix matrix2 = new Matrix();
                matrix2.postScale((float) 0.1,(float) 0.1);//因为检测太耗时，所以只能resize到0.1倍检测，此处可以再优化
                Bitmap bitmap = Bitmap.createBitmap(bitmapRotate,0,0,bitmapRotate.getWidth(),bitmapRotate.getHeight(),matrix2,false);
                //---------------------------

                //预测得到输出---------------
                Tensor inputTensor = predictor.getInput(0);
                int bitmap_height = bitmap.getHeight();
                int bitmap_width = bitmap.getWidth();
                long[] dims = {1, 3, bitmap_height, bitmap_width};
                inputTensor.resize(dims);
                long starttime = System.currentTimeMillis();
                float[] inputData = new float[3*bitmap_height*bitmap_width];
                for(int y=0;y<bitmap_height;y++)
                {
                    for(int x=0;x<bitmap_width;x++)
                    {
                        int color = bitmap.getPixel(x,y);
                        float[] rgb = new float[]{(float)red(color)-123,(float) green(color)-117, (float) blue(color)-104};
                        inputData[y*bitmap_width + x] = (float) (rgb[2] * 0.007843);
                        inputData[y*bitmap_width + x + bitmap_width*bitmap_height] = (float) (rgb[1]*0.007843);
                        inputData[y*bitmap_width + x + bitmap_width*bitmap_height*2] = (float) (rgb[0]*0.007843);
                    }
                }
                inputTensor.setData(inputData);
                predictor.run();
                Tensor outputTensor = predictor.getOutput(0);
                long outputShape[] = outputTensor.shape();
                long endtime = System.currentTimeMillis();
                Log.e("time:",(endtime-starttime)+"ms");
                //------------------------------

                //取结果------------------------
                float[][] result = new float[3][4];
                long outputSize = 1;
                for (long s : outputShape)
                {
                    outputSize *= s;
                }
                int validObjectNum=0;
                for(int i=1; i<outputSize;i=i+6)
                {
                    float score = outputTensor.getFloatData()[i];
                    if(score>0.5 && validObjectNum<3)
                    {
                        result[validObjectNum][0] = (int) (outputTensor.getFloatData()[i+1] * bitmap_width*10);
                        result[validObjectNum][1] = (int) (outputTensor.getFloatData()[i+2] * bitmap_height*10);
                        result[validObjectNum][2] = (int) (outputTensor.getFloatData()[i+3] * bitmap_width*10);
                        result[validObjectNum][3] = (int) (outputTensor.getFloatData()[i+4] * bitmap_height*10);
                        validObjectNum = validObjectNum+1;
                    }
                    else
                    {
                        break;
                    }
                }
                //------------------------------------

                //绘制人脸框--------------------------
                Canvas canvas = mTextureView_result.lockCanvas();
                canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
                Paint p = new Paint();
                p.setColor(Color.RED);// 设置红色
                p.setStrokeWidth(5);
                p.setTextSize(40);
                for(int i=0;i<validObjectNum;i++)
                {
                    canvas.drawLine(result[i][0],result[i][1],result[i][2],result[i][1],p);
                    canvas.drawLine(result[i][0],result[i][1],result[i][0],result[i][3],p);
                    canvas.drawLine(result[i][2],result[i][3],result[i][2],result[i][1],p);
                    canvas.drawLine(result[i][2],result[i][3],result[i][0],result[i][3],p);
                }
                mTextureView_result.unlockCanvasAndPost(canvas);
                //-------------------------------

                //更新坐标-----------------------
                //将已有的脸的坐标更新上去
                for(int i=0;i<3;i++){
                    int flag=0;
                    for(int j=0;j<validObjectNum;j++){
                        float thisIou = iou(face[i].xmin,face[i].ymin,face[i].xmax,face[i].ymax,result[j][0],result[j][1],result[j][2],result[j][3]);
                        if(thisIou>0.5 && face[i].valid==1){
                            face[i].xmin=result[j][0];
                            face[i].ymin=result[j][1];
                            face[i].xmax=result[j][2];
                            face[i].ymax=result[j][3];
                            face[i].trackCount=face[i].trackCount+1;
                            result[j][0]=0;  //设置未0表示此坐标框无效
                            result[j][1]=0;
                            result[j][2]=0;
                            result[j][3]=0;
                            flag=1;
                            break;
                        }
                    }
                    if(flag==0){
                        face[i].valid=0;
                        face[i].trackCount=0;
                    }
                }
                //将新产生的脸的坐标添加
                for(int i=0;i<3;i++){
                    for(int j=0;j<validObjectNum;j++){
                        if(face[i].valid==0 && result[j][0]>0){
                            nowId = nowId+1;
                            face[i].id = nowId;
                            face[i].xmin = result[j][0];
                            face[i].ymin = result[j][1];
                            face[i].xmax = result[j][2];
                            face[i].ymax = result[j][3];
                            face[i].valid = 1;
                            result[j][0]=0;
                            result[j][1]=0;
                            result[j][2]=0;
                            result[j][3]=0;
                            break;
                        }
                    }
                }
                //-------------------------------

                //该textureview用于显示被框选保存的目标
                for(int i=0;i<3;i++)
                {
                    Canvas canvasFace = mTextureView_face[i].lockCanvas();
                    canvasFace.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
                    if(face[i].valid==1){
                        mSrcRect.set((int)(face[i].xmin),(int)(face[i].ymin),(int)(face[i].xmax), (int)(face[i].ymax));
                        mDstRect.set(0,0,canvasFace.getWidth(), (int)(face[i].ymax-face[i].ymin) * canvasFace.getWidth()/((int)(face[i].xmax-face[i].xmin)));
                        canvasFace.drawBitmap(bitmapRotate, mSrcRect, mDstRect, mPaint);
                        canvasFace.drawText(String.valueOf(face[i].id),0,30,p);
                        //canvasFace.drawText(String.valueOf(face[i].id),(face[i].xmax+face[i].xmin)/2,(face[i].ymax+face[i].ymin)/2,p);
                        if(face[i].trackCount==30){//暂无质量判断，直接指定写出第30帧
                            Bitmap faceTmp = Bitmap.createBitmap(bitmapRotate,(int)face[i].xmin,(int)face[i].ymin,(int)(face[i].xmax-face[i].xmin),(int)(face[i].ymax-face[i].ymin));
                            saveToSystemGallery(faceTmp);
                        }
                    }
                    mTextureView_face[i].unlockCanvasAndPost(canvasFace);
                }

                image.close();
            }
        },null);
    }
    private void startPreview(){
        SurfaceTexture mSurfaceTexture = mTextureView.getSurfaceTexture();
        mSurfaceTexture.setDefaultBufferSize(mPreviewSize.getWidth(), mPreviewSize.getHeight());
        Surface mSurface = new Surface(mSurfaceTexture);
        Surface imageReaderSurface = mImageReader.getSurface();
        try
        {
             mCaptureRequestBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
             mCaptureRequestBuilder.addTarget(mSurface);
             mCaptureRequestBuilder.addTarget(imageReaderSurface);
             mCameraDevice.createCaptureSession(Arrays.asList(mSurface,imageReaderSurface), new CameraCaptureSession.StateCallback() {
                 @Override
                 public void onConfigured(CameraCaptureSession session) {
                     try {
                         CaptureRequest mCaptureRequest = mCaptureRequestBuilder.build();
                         CameraCaptureSession mPreviewSession = session;
                         mPreviewSession.setRepeatingRequest(mCaptureRequest, null, null);
                     } catch (CameraAccessException e) {
                         e.printStackTrace();
                     }
                 }

                 @Override
                 public void onConfigureFailed(CameraCaptureSession cameraCaptureSession) {
                 }
             }, null);
        }
        catch (CameraAccessException e)
        {
            e.printStackTrace();
        }
    }
    public String getModelPath(Context appCtx){
        //String modelPath = "object_detection/models/ssd_mobilenet_v1_pascalvoc_for_cpu";
        String modelPath = "detect_v070";//该人脸检测模型非最优模型，效果普通
        String realPath = modelPath;
        realPath = appCtx.getCacheDir() + "/" + modelPath;
        copyDirectoryFromAssets(appCtx, modelPath, realPath);
        return realPath;
    }
    public void copyDirectoryFromAssets(Context appCtx, String srcDir, String dstDir) {
        if (srcDir.isEmpty() || dstDir.isEmpty()) {
            return;
        }
        try {
            if (!new File(dstDir).exists()) {
                new File(dstDir).mkdirs();
            }
            for (String fileName : appCtx.getAssets().list(srcDir)) {
                String srcSubPath = srcDir + File.separator + fileName;
                String dstSubPath = dstDir + File.separator + fileName;
                if (new File(srcSubPath).isDirectory()) {
                    copyDirectoryFromAssets(appCtx, srcSubPath, dstSubPath);
                } else {
                    copyFileFromAssets(appCtx, srcSubPath, dstSubPath);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    public void copyFileFromAssets(Context appCtx, String srcPath, String dstPath) {
        if (srcPath.isEmpty() || dstPath.isEmpty()) {
            return;
        }
        InputStream is = null;
        OutputStream os = null;
        try {
            is = new BufferedInputStream(appCtx.getAssets().open(srcPath));
            os = new BufferedOutputStream(new FileOutputStream(new File(dstPath)));
            byte[] buffer = new byte[1024];
            int length = 0;
            while ((length = is.read(buffer)) != -1) {
                os.write(buffer, 0, length);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                os.close();
                is.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
    public float iou(float xmin1,float ymin1,float xmax1,float ymax1,float xmin2,float ymin2,float xmax2,float ymax2){
        float xmin = xmin1 > xmin2 ? xmin1:xmin2;
        float ymin = ymin1 > ymin2 ? ymin1:ymin2;
        float xmax = xmax1 > xmax2 ? xmax2:xmax1;
        float ymax = ymax1 > ymax2 ? ymax2:ymax1;
        if(xmax<xmin || ymax<ymin){
            return 0;
        }
        float intersection = (xmax - xmin)*(ymax - ymin);
        float area = (xmax1-xmin1)*(ymax1-ymin1) + (xmax2-xmin2)*(ymax2-ymin2);
        if(area==intersection){
            return 0;
        }
        float iou = intersection/(area-intersection);
        return iou;
    }
    public void saveToSystemGallery(Bitmap bmp) {
        // 首先保存图片
        File appDir = new File(Environment.getExternalStorageDirectory(), "vgmap");
        if (!appDir.exists()) {
            appDir.mkdir();
        }
        //String fileName = System.currentTimeMillis() + ".jpg";
        String fileName = "monitor" + ".jpg";
        File file = new File(appDir, fileName);
        try {
            FileOutputStream fos = new FileOutputStream(file);
            bmp.compress(Bitmap.CompressFormat.JPEG, 100, fos);
            fos.flush();
            fos.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        // 其次把文件插入到系统图库
        Context mContext = getApplicationContext();
        try {
            MediaStore.Images.Media.insertImage(mContext.getContentResolver(),
                    file.getAbsolutePath(), fileName, null);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        // 最后通知图库更新
        mContext.sendBroadcast(new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE, Uri.parse(file.getAbsolutePath())));
    }
}
