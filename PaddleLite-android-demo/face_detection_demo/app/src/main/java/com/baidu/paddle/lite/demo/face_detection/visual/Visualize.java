package com.baidu.paddle.lite.demo.face_detection.visual;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.util.Log;

import com.baidu.paddle.lite.PaddlePredictor;
import com.baidu.paddle.lite.Tensor;

import java.util.ArrayList;
import java.util.List;

public class Visualize {
    private static final String TAG = Visualize.class.getSimpleName();
    protected String outputResult = "";

    protected float[][] boxs;
    protected float[] scores;
    int nmsParamTopK = 100;
    float nmsParamNmsThreshold = (float) 0.5;
    float confidenceThreshold = (float) 0.5;
    public List<float[]> boxAndScores;

    public void nms(Bitmap outputImage, PaddlePredictor Predictor ){

        boxAndScores = new ArrayList<>();
        boxs = new float[8840][4];
        scores = new float[8840];
        Tensor scoresTensor = Predictor.getOutput(0);
        Tensor boxsTensor = Predictor.getOutput(1);

        // post-process
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

        NMS.sortScores(this.boxs, this.scores);

        int[] index = NMS.nmsScoreFilter(this.boxs, this.scores, nmsParamTopK, nmsParamNmsThreshold);

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
        Log.i(TAG, "len of boxAndScores: " +  boxAndScores.size());
    }

    public void draw(List<float[]> boxAndScores, Bitmap outputImage, String outputResult ){
        Canvas canvas = new Canvas(outputImage);
        Paint rectPaint = new Paint();
        rectPaint.setStyle(Paint.Style.STROKE);
        rectPaint.setStrokeWidth(2);
        Paint txtPaint = new Paint();
        txtPaint.setTextSize(12);
        txtPaint.setAntiAlias(true);
        int color = 0xFFFFFF33;
        rectPaint.setColor(color);
        txtPaint.setColor(color);
        int txtXOffset = 4;
        int txtYOffset = (int) (Math.ceil(-txtPaint.getFontMetrics().ascent));

        for(float[] boxAndScore: boxAndScores) {
            Log.i(TAG, "boxAndScore: " + boxAndScore[0] + boxAndScore[1] + boxAndScore[2] + boxAndScore[3] + boxAndScore[4]);
            canvas.drawRect(boxAndScore[0], boxAndScore[1], boxAndScore[2], boxAndScore[3], rectPaint);

        }

    }
}
