package com.baidu.paddle.lite.demo;

import android.content.Context;

import com.baidu.paddle.lite.CxxConfig;
import com.baidu.paddle.lite.PaddlePredictor;
import com.baidu.paddle.lite.Place;
import com.baidu.paddle.lite.Tensor;

import java.util.ArrayList;
import java.util.Date;

public class Predictor {
    private static final String TAG = Predictor.class.getSimpleName();

    public boolean isLoaded = false;
    public int warmupIterNum = 1;
    public int inferIterNum = 1;
    protected Context appCtx = null;
    public String modelName = "";
    protected int whichDevice = 0; // 0: CPU 1: NPU
    protected ArrayList<PaddlePredictor> paddlePredictors = new ArrayList<PaddlePredictor>(); // 0: CPU 1: NPU
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
        String realPath = Utils.copyFromAssetsToCache(appCtx, modelPath);
        if (realPath.isEmpty()) {
            return false;
        }
        // CPU
        {
            CxxConfig config = new CxxConfig();
            config.setModelDir(realPath);
            Place preferredPlace = new Place(Place.TargetType.ARM, Place.PrecisionType.FLOAT);
            Place[] validPlaces = new Place[2];
            validPlaces[0] = new Place(Place.TargetType.HOST, Place.PrecisionType.FLOAT);
            validPlaces[1] = new Place(Place.TargetType.ARM, Place.PrecisionType.FLOAT);
            config.setPreferredPlace(preferredPlace);
            config.setValidPlaces(validPlaces);
            paddlePredictors.add(PaddlePredictor.createPaddlePredictor(config));
        }
        // NPU
        {
            CxxConfig config = new CxxConfig();
            config.setModelDir(realPath);
            Place preferredPlace = new Place(Place.TargetType.NPU, Place.PrecisionType.FLOAT);
            Place[] validPlaces = new Place[3];
            validPlaces[0] = new Place(Place.TargetType.HOST, Place.PrecisionType.FLOAT);
            validPlaces[1] = new Place(Place.TargetType.ARM, Place.PrecisionType.FLOAT);
            validPlaces[2] = new Place(Place.TargetType.NPU, Place.PrecisionType.FLOAT);
            config.setPreferredPlace(preferredPlace);
            config.setValidPlaces(validPlaces);
            paddlePredictors.add(PaddlePredictor.createPaddlePredictor(config));
        }
        modelName = realPath.substring(realPath.lastIndexOf("/") + 1);
        return true;
    }

    public void releaseModel() {
        paddlePredictors.clear();
        isLoaded = false;
        modelName = "";
        whichDevice = 0;
    }

    public void useCPU() {
        whichDevice = 0;
    }

    public void useNPU() {
        whichDevice = 1;
    }

    public Tensor getInput(int idx) {
        if (paddlePredictors.size() < whichDevice + 1) {
            return null;
        }
        return paddlePredictors.get(whichDevice).getInput(idx);
    }

    public Tensor getOutput(int idx) {
        if (paddlePredictors.size() < whichDevice + 1) {
            return null;
        }
        return paddlePredictors.get(whichDevice).getOutput(idx);
    }

    public boolean runModel() {
        if (paddlePredictors.size() < whichDevice + 1) {
            return false;
        }
        // warm up
        for (int i = 0; i < warmupIterNum; i++) {
            paddlePredictors.get(whichDevice).run();
        }
        // inference
        Date start = new Date();
        for (int i = 0; i < inferIterNum; i++) {
            paddlePredictors.get(whichDevice).run();
        }
        Date end = new Date();
        inferenceTime = (end.getTime() - start.getTime()) / (float) inferIterNum;
        return true;
    }

    public boolean isLoaded() {
        return isLoaded;
    }

    public String modelName() {
        return modelName;
    }

    public float inferenceTime() {
        return inferenceTime;
    }
}


