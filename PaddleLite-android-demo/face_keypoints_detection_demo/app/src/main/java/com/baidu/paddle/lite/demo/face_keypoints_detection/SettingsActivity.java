package com.baidu.paddle.lite.demo.face_keypoints_detection;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.preference.EditTextPreference;
import android.preference.ListPreference;
import android.preference.PreferenceManager;
import android.support.v7.app.ActionBar;

import com.baidu.paddle.lite.demo.common.AppCompatPreferenceActivity;
import com.baidu.paddle.lite.demo.common.Utils;

import java.util.ArrayList;
import java.util.List;

public class SettingsActivity extends AppCompatPreferenceActivity implements SharedPreferences.OnSharedPreferenceChangeListener {
    private static final String TAG = SettingsActivity.class.getSimpleName();

    // Face detector
    static public int fdtSelectedModelIdx = 0;
    static public String fdtModelDir = "";
    static public int fdtCPUThreadNum = 0;
    static public String fdtCPUPowerMode = "";
    static public float fdtInputScale = 0.f;
    static public float[] fdtInputMean = new float[]{};
    static public float[] fdtInputStd = new float[]{};
    static public float fdtScoreThreshold = 0.f;

    ListPreference lpFdtChoosePreInstalledModel = null;
    EditTextPreference etFdtModelDir = null;
    ListPreference lpFdtCPUThreadNum = null;
    ListPreference lpFdtCPUPowerMode = null;
    EditTextPreference etFdtInputScale = null;
    EditTextPreference etFdtInputMean = null;
    EditTextPreference etFdtInputStd = null;
    EditTextPreference etFdtScoreThreshold = null;

    List<String> fdtPreInstalledModelDirs = null;
    List<String> fdtPreInstalledCPUThreadNums = null;
    List<String> fdtPreInstalledCPUPowerModes = null;
    List<String> fdtPreInstalledInputScales = null;
    List<String> fdtPreInstalledInputMeans = null;
    List<String> fdtPreInstalledInputStds = null;
    List<String> fdtPreInstalledScoreThresholds = null;

    // FaceKeypoints detection
    static public int fkpSelectedModelIdx = 0;
    static public String fkpModelDir = "";
    static public int fkpCPUThreadNum = 0;
    static public String fkpCPUPowerMode = "";
    static public int fkpInputWidth = 0;
    static public int fkpInputHeight = 0;

    ListPreference lpFkpChoosePreInstalledModel = null;
    EditTextPreference etFkpModelDir = null;
    ListPreference lpFkpCPUThreadNum = null;
    ListPreference lpFkpCPUPowerMode = null;
    EditTextPreference etFkpInputWidth = null;
    EditTextPreference etFkpInputHeight = null;

    List<String> fkpPreInstalledModelDirs = null;
    List<String> fkpPreInstalledCPUThreadNums = null;
    List<String> fkpPreInstalledCPUPowerModes = null;
    List<String> fkpPreInstalledInputWidths = null;
    List<String> fkpPreInstalledInputHeights = null;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        addPreferencesFromResource(R.xml.settings);
        ActionBar supportActionBar = getSupportActionBar();
        if (supportActionBar != null) {
            supportActionBar.setDisplayHomeAsUpEnabled(true);
        }

        // Face detector
        // Initialize pre-installed models
        fdtPreInstalledModelDirs = new ArrayList<String>();
        fdtPreInstalledCPUThreadNums = new ArrayList<String>();
        fdtPreInstalledCPUPowerModes = new ArrayList<String>();
        fdtPreInstalledInputScales = new ArrayList<String>();
        fdtPreInstalledInputMeans = new ArrayList<String>();
        fdtPreInstalledInputStds = new ArrayList<String>();
        fdtPreInstalledScoreThresholds = new ArrayList<String>();
        fdtPreInstalledModelDirs.add(getString(R.string.FDT_MODEL_DIR_DEFAULT));
        fdtPreInstalledCPUThreadNums.add(getString(R.string.FDT_CPU_THREAD_NUM_DEFAULT));
        fdtPreInstalledCPUPowerModes.add(getString(R.string.FDT_CPU_POWER_MODE_DEFAULT));
        fdtPreInstalledInputScales.add(getString(R.string.FDT_INPUT_SCALE_DEFAULT));
        fdtPreInstalledInputMeans.add(getString(R.string.FDT_INPUT_MEAN_DEFAULT));
        fdtPreInstalledInputStds.add(getString(R.string.FDT_INPUT_STD_DEFAULT));
        fdtPreInstalledScoreThresholds.add(getString(R.string.FDT_SCORE_THRESHOLD_DEFAULT));
        // Setup UI components
        lpFdtChoosePreInstalledModel =
                (ListPreference) findPreference(getString(R.string.FDT_CHOOSE_PRE_INSTALLED_MODEL_KEY));
        String[] fdtPreInstalledModelNames = new String[fdtPreInstalledModelDirs.size()];
        for (int i = 0; i < fdtPreInstalledModelDirs.size(); i++) {
            fdtPreInstalledModelNames[i] =
                    fdtPreInstalledModelDirs.get(i).substring(fdtPreInstalledModelDirs.get(i).lastIndexOf("/") + 1);
        }
        lpFdtChoosePreInstalledModel.setEntries(fdtPreInstalledModelNames);
        lpFdtChoosePreInstalledModel.setEntryValues(fdtPreInstalledModelDirs.toArray(new String[fdtPreInstalledModelDirs.size()]));
        lpFdtCPUThreadNum =
                (ListPreference) findPreference(getString(R.string.FDT_CPU_THREAD_NUM_KEY));
        lpFdtCPUPowerMode =
                (ListPreference) findPreference(getString(R.string.FDT_CPU_POWER_MODE_KEY));
        etFdtModelDir = (EditTextPreference) findPreference(getString(R.string.FDT_MODEL_DIR_KEY));
        etFdtModelDir.setTitle("Model dir (SDCard: " + Utils.getSDCardDirectory() + ")");
        etFdtInputScale = (EditTextPreference) findPreference(getString(R.string.FDT_INPUT_SCALE_KEY));
        etFdtInputMean = (EditTextPreference) findPreference(getString(R.string.FDT_INPUT_MEAN_KEY));
        etFdtInputStd = (EditTextPreference) findPreference(getString(R.string.FDT_INPUT_STD_KEY));
        etFdtScoreThreshold = (EditTextPreference) findPreference(getString(R.string.FDT_SCORE_THRESHOLD_KEY));

        // Face keypoints detector
        // Initialize pre-installed models
        fkpPreInstalledModelDirs = new ArrayList<String>();
        fkpPreInstalledCPUThreadNums = new ArrayList<String>();
        fkpPreInstalledCPUPowerModes = new ArrayList<String>();
        fkpPreInstalledInputWidths = new ArrayList<String>();
        fkpPreInstalledInputHeights = new ArrayList<String>();
        fkpPreInstalledModelDirs.add(getString(R.string.FKP_MODEL_DIR_DEFAULT));
        fkpPreInstalledCPUThreadNums.add(getString(R.string.FKP_CPU_THREAD_NUM_DEFAULT));
        fkpPreInstalledCPUPowerModes.add(getString(R.string.FKP_CPU_POWER_MODE_DEFAULT));
        fkpPreInstalledInputWidths.add(getString(R.string.FKP_INPUT_WIDTH_DEFAULT));
        fkpPreInstalledInputHeights.add(getString(R.string.FKP_INPUT_HEIGHT_DEFAULT));
        // Setup UI components
        lpFkpChoosePreInstalledModel =
                (ListPreference) findPreference(getString(R.string.FKP_CHOOSE_PRE_INSTALLED_MODEL_KEY));
        String[] fkpPreInstalledModelNames = new String[fkpPreInstalledModelDirs.size()];
        for (int i = 0; i < fkpPreInstalledModelDirs.size(); i++) {
            fkpPreInstalledModelNames[i] =
                    fkpPreInstalledModelDirs.get(i).substring(fkpPreInstalledModelDirs.get(i).lastIndexOf("/") + 1);
        }
        lpFkpChoosePreInstalledModel.setEntries(fkpPreInstalledModelNames);
        lpFkpChoosePreInstalledModel.setEntryValues(fkpPreInstalledModelDirs.toArray(new String[fkpPreInstalledModelDirs.size()]));
        lpFkpCPUThreadNum =
                (ListPreference) findPreference(getString(R.string.FKP_CPU_THREAD_NUM_KEY));
        lpFkpCPUPowerMode =
                (ListPreference) findPreference(getString(R.string.FKP_CPU_POWER_MODE_KEY));
        etFkpModelDir = (EditTextPreference) findPreference(getString(R.string.FKP_MODEL_DIR_KEY));
        etFkpModelDir.setTitle("Model dir (SDCard: " + Utils.getSDCardDirectory() + ")");
        etFkpInputWidth = (EditTextPreference) findPreference(getString(R.string.FKP_INPUT_WIDTH_KEY));
        etFkpInputHeight = (EditTextPreference) findPreference(getString(R.string.FKP_INPUT_HEIGHT_KEY));
    }

    private void reloadSettingsAndUpdateUI() {
        SharedPreferences sharedPreferences = getPreferenceScreen().getSharedPreferences();

        // Face detector
        String selectedModelPath = sharedPreferences.getString(getString(R.string.FDT_CHOOSE_PRE_INSTALLED_MODEL_KEY),
                getString(R.string.FDT_MODEL_DIR_DEFAULT));
        int selectedModelIdx = lpFdtChoosePreInstalledModel.findIndexOfValue(selectedModelPath);
        if (selectedModelIdx >= 0 && selectedModelIdx < fdtPreInstalledModelDirs.size() && selectedModelIdx != fdtSelectedModelIdx) {
            SharedPreferences.Editor editor = sharedPreferences.edit();
            editor.putString(getString(R.string.FDT_MODEL_DIR_KEY), fdtPreInstalledModelDirs.get(selectedModelIdx));
            editor.putString(getString(R.string.FDT_CPU_THREAD_NUM_KEY), fdtPreInstalledCPUThreadNums.get(selectedModelIdx));
            editor.putString(getString(R.string.FDT_CPU_POWER_MODE_KEY), fdtPreInstalledCPUPowerModes.get(selectedModelIdx));
            editor.putString(getString(R.string.FDT_INPUT_SCALE_KEY), fdtPreInstalledInputScales.get(selectedModelIdx));
            editor.putString(getString(R.string.FDT_INPUT_MEAN_KEY), fdtPreInstalledInputMeans.get(selectedModelIdx));
            editor.putString(getString(R.string.FDT_INPUT_STD_KEY), fdtPreInstalledInputStds.get(selectedModelIdx));
            editor.putString(getString(R.string.FDT_SCORE_THRESHOLD_KEY), fdtPreInstalledScoreThresholds.get(selectedModelIdx));
            editor.commit();
            lpFdtChoosePreInstalledModel.setSummary(selectedModelPath);
            fdtSelectedModelIdx = selectedModelIdx;
        }

        String modelDir = sharedPreferences.getString(getString(R.string.FDT_MODEL_DIR_KEY),
                getString(R.string.FDT_MODEL_DIR_DEFAULT));
        String cpuThreadNum = sharedPreferences.getString(getString(R.string.FDT_CPU_THREAD_NUM_KEY),
                getString(R.string.FDT_CPU_THREAD_NUM_DEFAULT));
        String cpuPowerMode = sharedPreferences.getString(getString(R.string.FDT_CPU_POWER_MODE_KEY),
                getString(R.string.FDT_CPU_POWER_MODE_DEFAULT));
        String inputScale = sharedPreferences.getString(getString(R.string.FDT_INPUT_SCALE_KEY),
                getString(R.string.FDT_INPUT_SCALE_DEFAULT));
        String inputMean = sharedPreferences.getString(getString(R.string.FDT_INPUT_MEAN_KEY),
                getString(R.string.FDT_INPUT_MEAN_DEFAULT));
        String inputStd = sharedPreferences.getString(getString(R.string.FDT_INPUT_STD_KEY),
                getString(R.string.FDT_INPUT_STD_DEFAULT));
        String scoreThreshold = sharedPreferences.getString(getString(R.string.FDT_SCORE_THRESHOLD_KEY),
                getString(R.string.FDT_SCORE_THRESHOLD_DEFAULT));

        etFdtModelDir.setSummary(modelDir);
        lpFdtCPUThreadNum.setValue(cpuThreadNum);
        lpFdtCPUThreadNum.setSummary(cpuThreadNum);
        lpFdtCPUPowerMode.setValue(cpuPowerMode);
        lpFdtCPUPowerMode.setSummary(cpuPowerMode);
        etFdtInputScale.setSummary(inputScale);
        etFdtInputMean.setSummary(inputMean);
        etFdtInputMean.setText(inputMean);
        etFdtInputStd.setSummary(inputStd);
        etFdtInputStd.setText(inputStd);
        etFdtScoreThreshold.setSummary(scoreThreshold);

        // Face keypoints detector
        selectedModelPath = sharedPreferences.getString(getString(R.string.FKP_CHOOSE_PRE_INSTALLED_MODEL_KEY),
                getString(R.string.FKP_MODEL_DIR_DEFAULT));
        selectedModelIdx = lpFkpChoosePreInstalledModel.findIndexOfValue(selectedModelPath);
        if (selectedModelIdx >= 0 && selectedModelIdx < fkpPreInstalledModelDirs.size() && selectedModelIdx != fkpSelectedModelIdx) {
            SharedPreferences.Editor editor = sharedPreferences.edit();
            editor.putString(getString(R.string.FKP_MODEL_DIR_KEY), fkpPreInstalledModelDirs.get(selectedModelIdx));
            editor.putString(getString(R.string.FKP_CPU_THREAD_NUM_KEY), fkpPreInstalledCPUThreadNums.get(selectedModelIdx));
            editor.putString(getString(R.string.FKP_CPU_POWER_MODE_KEY), fkpPreInstalledCPUPowerModes.get(selectedModelIdx));
            editor.putString(getString(R.string.FKP_INPUT_WIDTH_KEY), fkpPreInstalledInputWidths.get(selectedModelIdx));
            editor.putString(getString(R.string.FKP_INPUT_HEIGHT_KEY), fkpPreInstalledInputHeights.get(selectedModelIdx));
            editor.commit();
            lpFkpChoosePreInstalledModel.setSummary(selectedModelPath);
            fkpSelectedModelIdx = selectedModelIdx;
        }

        modelDir = sharedPreferences.getString(getString(R.string.FKP_MODEL_DIR_KEY),
                getString(R.string.FKP_MODEL_DIR_DEFAULT));
        cpuThreadNum = sharedPreferences.getString(getString(R.string.FKP_CPU_THREAD_NUM_KEY),
                getString(R.string.FKP_CPU_THREAD_NUM_DEFAULT));
        cpuPowerMode = sharedPreferences.getString(getString(R.string.FKP_CPU_POWER_MODE_KEY),
                getString(R.string.FKP_CPU_POWER_MODE_DEFAULT));
        String inputWidth = sharedPreferences.getString(getString(R.string.FKP_INPUT_WIDTH_KEY),
                getString(R.string.FKP_INPUT_WIDTH_DEFAULT));
        String inputHeight = sharedPreferences.getString(getString(R.string.FKP_INPUT_HEIGHT_KEY),
                getString(R.string.FKP_INPUT_HEIGHT_DEFAULT));

        etFkpModelDir.setSummary(modelDir);
        lpFkpCPUThreadNum.setValue(cpuThreadNum);
        lpFkpCPUThreadNum.setSummary(cpuThreadNum);
        lpFkpCPUPowerMode.setValue(cpuPowerMode);
        lpFkpCPUPowerMode.setSummary(cpuPowerMode);
        etFkpInputWidth.setSummary(inputWidth);
        etFkpInputHeight.setSummary(inputHeight);
    }

    static boolean checkAndUpdateSettings(Context ctx) {
        boolean settingsChanged = false;
        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(ctx);

        // Face detector
        String modelDir = sharedPreferences.getString(ctx.getString(R.string.FDT_MODEL_DIR_KEY),
                ctx.getString(R.string.FDT_MODEL_DIR_DEFAULT));
        settingsChanged |= !fdtModelDir.equalsIgnoreCase(modelDir);
        fdtModelDir = modelDir;

        String cpuThreadNum = sharedPreferences.getString(ctx.getString(R.string.FDT_CPU_THREAD_NUM_KEY),
                ctx.getString(R.string.FDT_CPU_THREAD_NUM_DEFAULT));
        settingsChanged |= fdtCPUThreadNum != Integer.parseInt(cpuThreadNum);
        fdtCPUThreadNum = Integer.parseInt(cpuThreadNum);

        String cpuPowerMode = sharedPreferences.getString(ctx.getString(R.string.FDT_CPU_POWER_MODE_KEY),
                ctx.getString(R.string.FDT_CPU_POWER_MODE_DEFAULT));
        settingsChanged |= !fdtCPUPowerMode.equalsIgnoreCase(cpuPowerMode);
        fdtCPUPowerMode = cpuPowerMode;

        String inputScale = sharedPreferences.getString(ctx.getString(R.string.FDT_INPUT_SCALE_KEY),
                ctx.getString(R.string.FDT_INPUT_SCALE_DEFAULT));
        settingsChanged |= fdtInputScale != Float.parseFloat(inputScale);
        fdtInputScale = Float.parseFloat(inputScale);

        String inputMean = sharedPreferences.getString(ctx.getString(R.string.FDT_INPUT_MEAN_KEY),
                ctx.getString(R.string.FDT_INPUT_MEAN_DEFAULT));
        float[] input_mean = Utils.parseFloatsFromString(inputMean, ",");
        settingsChanged |= input_mean.length != fdtInputMean.length;
        if (!settingsChanged) {
            for (int i = 0; i < input_mean.length; i++) {
                settingsChanged |= input_mean[i] != fdtInputMean[i];
            }
        }
        fdtInputMean = input_mean;

        String inputStd = sharedPreferences.getString(ctx.getString(R.string.FDT_INPUT_STD_KEY),
                ctx.getString(R.string.FDT_INPUT_STD_DEFAULT));
        float[] input_std = Utils.parseFloatsFromString(inputStd, ",");
        settingsChanged |= input_std.length != fdtInputStd.length;
        if (!settingsChanged) {
            for (int i = 0; i < input_std.length; i++) {
                settingsChanged |= input_std[i] != fdtInputStd[i];
            }
        }
        fdtInputStd = input_std;

        String scoreThreshold = sharedPreferences.getString(ctx.getString(R.string.FDT_SCORE_THRESHOLD_KEY),
                ctx.getString(R.string.FDT_SCORE_THRESHOLD_DEFAULT));
        settingsChanged |= fdtScoreThreshold != Float.parseFloat(scoreThreshold);
        fdtScoreThreshold = Float.parseFloat(scoreThreshold);

        // Face keypoints detector
        modelDir = sharedPreferences.getString(ctx.getString(R.string.FKP_MODEL_DIR_KEY),
                ctx.getString(R.string.FKP_MODEL_DIR_DEFAULT));
        settingsChanged |= !fdtModelDir.equalsIgnoreCase(modelDir);
        fkpModelDir = modelDir;

        cpuThreadNum = sharedPreferences.getString(ctx.getString(R.string.FKP_CPU_THREAD_NUM_KEY),
                ctx.getString(R.string.FKP_CPU_THREAD_NUM_DEFAULT));
        settingsChanged |= fkpCPUThreadNum != Integer.parseInt(cpuThreadNum);
        fkpCPUThreadNum = Integer.parseInt(cpuThreadNum);

        cpuPowerMode = sharedPreferences.getString(ctx.getString(R.string.FKP_CPU_POWER_MODE_KEY),
                ctx.getString(R.string.FKP_CPU_POWER_MODE_DEFAULT));
        settingsChanged |= !fkpCPUPowerMode.equalsIgnoreCase(cpuPowerMode);
        fkpCPUPowerMode = cpuPowerMode;

        String inputWidth = sharedPreferences.getString(ctx.getString(R.string.FKP_INPUT_WIDTH_KEY),
                ctx.getString(R.string.FKP_INPUT_WIDTH_DEFAULT));
        settingsChanged |= fkpInputWidth != Integer.parseInt(inputWidth);
        fkpInputWidth = Integer.parseInt(inputWidth);

        String inputHeight = sharedPreferences.getString(ctx.getString(R.string.FKP_INPUT_HEIGHT_KEY),
                ctx.getString(R.string.FKP_INPUT_HEIGHT_DEFAULT));
        settingsChanged |= fkpInputHeight != Integer.parseInt(inputHeight);
        fkpInputHeight = Integer.parseInt(inputHeight);
        return settingsChanged;
    }


    @Override
    protected void onResume() {
        super.onResume();
        getPreferenceScreen().getSharedPreferences().registerOnSharedPreferenceChangeListener(this);
        reloadSettingsAndUpdateUI();
    }

    @Override
    protected void onPause() {
        super.onPause();
        getPreferenceScreen().getSharedPreferences().unregisterOnSharedPreferenceChangeListener(this);
    }

    @Override
    public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String key) {
        reloadSettingsAndUpdateUI();
    }
}
