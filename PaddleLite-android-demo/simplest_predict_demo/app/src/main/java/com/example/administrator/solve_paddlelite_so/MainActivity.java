package com.example.administrator.solve_paddlelite_so;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import com.baidu.paddle.lite.MobileConfig;
import com.baidu.paddle.lite.PaddlePredictor;
import com.baidu.paddle.lite.PowerMode;
import com.baidu.paddle.lite.Tensor;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;

import static android.graphics.Color.blue;
import static android.graphics.Color.green;
import static android.graphics.Color.red;

public class MainActivity extends AppCompatActivity {
    InputStream imageStream = null;
    Bitmap image;
    String realPath;
    Context appCtx;
    MobileConfig config;
    PaddlePredictor paddlePredictor;
    String[] label;
    int width;
    int height;
    float inferenceTime;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //read image, resize and show. init label
        try{
            imageStream = getAssets().open("object_detection/images/dog.jpg");
        }catch (IOException e) {
            e.printStackTrace();
        }
        Bitmap imageTmp = BitmapFactory.decodeStream(imageStream);
        Bitmap rgbaImageTmp = imageTmp.copy(Bitmap.Config.ARGB_8888, true);
        image = Bitmap.createScaledBitmap(rgbaImageTmp, (int) 300, (int) 300, true);//because of the ssd model, input image size must be 300 pixel
        label = new String[]{"background","aeroplane" ,"bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"};
        ImageView mImageView = findViewById(R.id.image);
        mImageView.setImageBitmap(image);

        //resave model
        String modelPath = "object_detection/models/ssd_mobilenet_v1_pascalvoc_for_cpu";
        realPath = modelPath;
        appCtx = getApplicationContext();
        if (!modelPath.substring(0, 1).equals("/")) {
            realPath = appCtx.getCacheDir() + "/" + modelPath;
            Utils.copyDirectoryFromAssets(appCtx, modelPath, realPath);
        }

        //initial predictor
        config = new MobileConfig();
        config.setModelDir(realPath);
        config.setThreads(1);
        config.setPowerMode(PowerMode.LITE_POWER_HIGH);
        paddlePredictor = PaddlePredictor.createPaddlePredictor(config);

        //preprocessing image and copy`
        height = image.getHeight();
        width = image.getWidth();
        long[] dims={1,3,height,width};
        float[] inputBuffer = new float[3*height*width];
        for(int y=0;y<height;y++){
            for(int x=0;x<width;x++){
                int color = image.getPixel(x,y);
                float[] rgb = new float[]{(float)red(color)/255,(float) green(color)/255, (float) blue(color)/255};
                inputBuffer[y*width + x] = (float) ((rgb[0] -0.5)/0.5);
                inputBuffer[y*width + x + width*height] = (float) ((rgb[1]-0.5)/0.5);
                inputBuffer[y*width + x + width*height*2] = (float) ((rgb[2]-0.5)/0.5);
            }
        }

        //predict
        Tensor input = paddlePredictor.getInput(0);
        input.resize(dims);
        input.setData(inputBuffer);
        Date start = new Date();
        paddlePredictor.run();
        Date end = new Date();
        inferenceTime = (end.getTime() - start.getTime());

        //get output
        Tensor output = paddlePredictor.getOutput(0);

        //print result
        long outputShape[] = output.shape();
        long outputSize = 1;
        for (long s : outputShape) {
            outputSize *= s;
        }
        float[] outputValue = output.getFloatData();
        for(int i=0; i<outputSize;i++){
            Log.e("debug",String.valueOf(outputValue[i]));
        }

        //show result on screen
        List<List<Float>> result = new ArrayList<List<Float>>();
        for(int i=1;i<outputSize;i=i+6){
            if(outputValue[i]>0.5){
                result.add(Arrays.asList(outputValue[i-1], outputValue[i], outputValue[i+1]*width,outputValue[i+2]*height,outputValue[i+3]*width,outputValue[i+4]*height));
            }
        }
        String textString = "\n  inference time:"+String.valueOf(inferenceTime)+"ms\n";
        for(int i=0;i<result.size();i++){
            int index = (int)(result.get(i).get(0).floatValue());
            textString = textString+"  class:"+label[index]+
                    ", score:"+String.valueOf(result.get(i).get(1))+
                    ",\n  xmin:"+String.valueOf(result.get(i).get(2).intValue())+
                    ", ymin:"+String.valueOf(result.get(i).get(3).intValue())+
                    ", xmax:"+String.valueOf(result.get(i).get(4).intValue())+
                    ", xmax:"+String.valueOf(result.get(i).get(5).intValue())+"\n\n";
        }
        TextView mTextureView = findViewById(R.id.text);
        mTextureView.setText(textString);
        Log.e("debug",textString);
    }
}
