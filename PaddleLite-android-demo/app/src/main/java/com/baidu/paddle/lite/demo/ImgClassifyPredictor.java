/*
*@file ClassifyItemModel.java
*
* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

package com.baidu.paddle.lite.demo;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import com.baidu.paddle.lite.Tensor;

import java.io.InputStream;
import java.util.Vector;

import static android.graphics.Color.blue;
import static android.graphics.Color.green;
import static android.graphics.Color.red;

public class ImgClassifyPredictor extends Predictor {
    private static final String TAG = ImgClassifyPredictor.class.getSimpleName();
    protected Vector<String> wordLabels = new Vector<String>();
    protected long imageWidth = 224;
    protected long imageHeight = 224;
    protected Bitmap imageData = null;
    protected String top1Result = "";
    protected String top2Result = "";
    protected String top3Result = "";

    public ImgClassifyPredictor() {
        super();
    }

    public boolean init(Context appCtx, String modelPath, String labelPath, long imageWidth, long imageHeight) {
       super.init(appCtx, modelPath);
       if (!super.isLoaded()) {
           return false;
       }
       this.imageWidth = imageWidth;
       this.imageHeight = imageHeight;
       isLoaded &= loadLabel(labelPath);
       return isLoaded;
    }

    protected boolean loadLabel(String labelPath) {
        wordLabels.clear();
        // load word labels from file
        try {
            InputStream assetsInputStream = appCtx.getAssets().open(labelPath);
            int available = assetsInputStream.available();
            byte[] lines = new byte[available];
            assetsInputStream.read(lines);
            assetsInputStream.close();
            String words = new String(lines);
            String[] contents = words.split("\n");
            for (String content : contents) {
                int first_space_pos = content.indexOf(" ");
                if (first_space_pos >= 0 && first_space_pos < content.length()) {
                    wordLabels.add(content.substring(first_space_pos));
                }
            }
            Log.i(TAG, "word label size: " + wordLabels.size());
        } catch (Exception e) {
            Log.e(TAG, e.getMessage());
            return false;
        }
        return true;
    }

    public Tensor getInput(int idx) {
        return super.getInput(idx);
    }

    public Tensor getOutput(int idx) {
        return super.getOutput(idx);
    }

    public boolean runModel(Bitmap imageData) {
        setImageData(imageData);
        return runModel();
    }

    public boolean runModel() {
        if (imageData == null) {
            return false;
        }

        // set input shape
        Tensor inputTensor = getInput(0);
        long[] inputShape = {1, 3, imageHeight, imageWidth};
        inputTensor.resize(inputShape);

        // scale image, pre-process image, and feed input tensor with pre-processed data
        int channels = (int) inputShape[1];
        int width = (int) inputShape[3];
        int height = (int) inputShape[2];
        Bitmap rgbaData = imageData.copy(Bitmap.Config.ARGB_8888,true);
        Bitmap scaleData = Bitmap.createScaledBitmap(rgbaData, width, height,true);

        float[] inputData = new float[channels * width * height];
        int rIndex, gIndex, bIndex;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                bIndex = i * width + j;
                gIndex = bIndex + width * height;
                rIndex = gIndex + width * height;

                int color = scaleData.getPixel(j, i);

                inputData[bIndex] = (float) blue(color) / 255.0f;
                inputData[gIndex] = (float) green(color) / 255.0f;
                inputData[rIndex] = (float) red(color) / 255.0f;
            }
        }
        inputTensor.setData(inputData);

        // inference
        super.runModel();

        // fetch output tensor
        Tensor outputTensor = getOutput(0);

        // post-process
        long outputShape[] = outputTensor.shape();
        long outputSize = 1;
        for (long s : outputShape) {
            outputSize *= s;
        }
        int[] max_index = new int[3]; // top3 indices
        double[] max_num = new double[3]; // top3 scores
        for (int i = 0; i < outputSize; i++) {
            float tmp = outputTensor.getFloatData()[i];
            int tmp_index = i;
            for (int j = 0; j < 3; j++) {
                if (tmp > max_num[j]) {
                    tmp_index += max_index[j];
                    max_index[j] = tmp_index - max_index[j];
                    tmp_index -= max_index[j];
                    tmp += max_num[j];
                    max_num[j] = tmp - max_num[j];
                    tmp -= max_num[j];
                }
            }
        }

        if (wordLabels.size() > 0) {
            top1Result = "top1: " + wordLabels.get(max_index[0]) + " - " + String.format("%.2f", max_num[0] * 100) + "%";
            top2Result = "top2: " + wordLabels.get(max_index[1]) + " - " + String.format("%.2f", max_num[1] * 100) + "%";
            top3Result = "top3: " + wordLabels.get(max_index[2]) + " - " + String.format("%.2f", max_num[2] * 100) + "%";
        }
        return true;
    }

    public Bitmap imageData() {
        return imageData;
    }

    public void setImageData(Bitmap imageData) { this.imageData = imageData; }

    public String top1Result() {
        return top1Result;
    }

    public String top2Result() {
        return top2Result;
    }

    public String top3Result() {
        return top3Result;
    }
}


