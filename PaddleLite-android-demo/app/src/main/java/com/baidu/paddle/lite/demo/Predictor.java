package com.baidu.paddle.lite.demo;

import android.content.Context;
import com.baidu.paddle.lite.*;

import java.util.ArrayList;
import java.util.Date;

public class Predictor {
    private static final String TAG = Predictor.class.getSimpleName();

    public boolean isLoaded = false;
    public int warmupIterNum = 1;
    public int inferIterNum = 1;
    protected Context appCtx = null;
    public String modelName = "";
    protected PaddlePredictor paddlePredictor = null;
    protected float inferenceTime = 0;

    public Predictor() {
    }

    public boolean init(Context appCtx, String modelPath) {
        this.appCtx = appCtx;
        isLoaded = loadModel(modelPath);
        return isLoaded;
    }

    protected boolean loadModel(String modelPath) {
        // release model if exists
        releaseModel();

        // load model
        if (modelPath.isEmpty()) {
            return false;
        }
        String realPath = modelPath;
        if (!modelPath.substring(0, 1).equals("/")) {
            // read model files from custom path if the first character of mode path is '/'
            // otherwise copy model to cache from assets
            realPath = appCtx.getCacheDir() + "/" + modelPath;
            Utils.copyDirectoryFromAssets(appCtx, modelPath, realPath);
        }
        if (realPath.isEmpty()) {
            return false;
        }
        MobileConfig config = new MobileConfig();
        config.setModelDir(realPath);
        paddlePredictor = PaddlePredictor.createPaddlePredictor(config);

        modelName = realPath.substring(realPath.lastIndexOf("/") + 1);
        return true;
    }

    public void releaseModel() {
        paddlePredictor = null;
        isLoaded = false;
        modelName = "";
    }

    public Tensor getInput(int idx) {
        if (!isLoaded()) {
            return null;
        }
        return paddlePredictor.getInput(idx);
    }

    public Tensor getOutput(int idx) {
        if (!isLoaded()) {
            return null;
        }
        return paddlePredictor.getOutput(idx);
    }

    public boolean runModel() {
        if (!isLoaded()) {
            return false;
        }
        // warm up
        for (int i = 0; i < warmupIterNum; i++) {
            paddlePredictor.run();
        }
        // inference
        Date start = new Date();
        for (int i = 0; i < inferIterNum; i++) {
            paddlePredictor.run();
        }
        Date end = new Date();
        inferenceTime = (end.getTime() - start.getTime()) / (float) inferIterNum;
        return true;
    }

    public boolean isLoaded() {
        return paddlePredictor != null && isLoaded;
    }

    public String modelName() {
        return modelName;
    }

    public float inferenceTime() {
        return inferenceTime;
    }
}
