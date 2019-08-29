package com.baidu.paddle.lite.demo;

import android.content.SharedPreferences;
import android.os.Bundle;
import android.preference.CheckBoxPreference;
import android.preference.EditTextPreference;
import android.preference.ListPreference;
import android.support.v7.app.ActionBar;


public class ImgClassifySettingsActivity extends AppCompatPreferenceActivity implements SharedPreferences.OnSharedPreferenceChangeListener {
    ListPreference lpChoosePreInstalledModel = null;
    CheckBoxPreference cbEnableCustomModelSettings = null;
    EditTextPreference etModelPath = null;
    EditTextPreference etLabelPath = null;
    EditTextPreference etImagePath = null;
    EditTextPreference etInputShape = null;
    EditTextPreference etInputMean = null;
    EditTextPreference etInputScale = null;

    String[] preInstalledModelPaths = null;
    String[] preInstalledLabelPaths = null;
    String[] preInstalledImagePaths = null;
    String[] preInstalledInputShapes = null;
    String[] preInstalledInputMeans = null;
    String[] preInstalledInputScales = null;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        addPreferencesFromResource(R.xml.settings_img_classify);
        ActionBar supportActionBar = getSupportActionBar();
        if (supportActionBar != null) {
            supportActionBar.setDisplayHomeAsUpEnabled(true);
        }

        // initialize the settings of pre-installed models
        preInstalledModelPaths = new String[]{getString(R.string.ICS_MODEL_PATH_DEFAULT)};
        preInstalledLabelPaths = new String[]{getString(R.string.ICS_LABEL_PATH_DEFAULT)};
        preInstalledImagePaths = new String[]{getString(R.string.ICS_IMAGE_PATH_DEFAULT)};
        preInstalledInputShapes = new String[]{getString(R.string.ICS_INPUT_SHAPE_DEFAULT)};
        preInstalledInputMeans = new String[]{getString(R.string.ICS_INPUT_MEAN_DEFAULT)};
        preInstalledInputScales = new String[]{getString(R.string.ICS_INPUT_SCALE_DEFAULT)};
        /*
        preInstalledModelPaths = new String[]{getString(R.string.ICS_MODEL_PATH_DEFAULT), "image_classification" +
                "/models/mobilenet_v2_relu"};
        preInstalledLabelPaths = new String[]{getString(R.string.ICS_LABEL_PATH_DEFAULT), "image_classification" +
                "/labels/synset_words.txt"};
        preInstalledImagePaths = new String[]{getString(R.string.ICS_IMAGE_PATH_DEFAULT), "image_classification" +
                "/images/egypt_cat.jpg"};
        preInstalledInputShapes = new String[]{getString(R.string.ICS_INPUT_SHAPE_DEFAULT), "1,3,224,224"};
        preInstalledInputMeans = new String[]{getString(R.string.ICS_INPUT_MEAN_DEFAULT), "0,0,0"};
        preInstalledInputScales = new String[]{getString(R.string.ICS_INPUT_SCALE_DEFAULT), "0.0039,0.0039,0.0039"};
        */

        // intialize all of UI components
        lpChoosePreInstalledModel =
                (ListPreference) findPreference(getString(R.string.ICS_CHOOSE_PRE_INSTALLED_MODEL_KEY));
        String[] preInstalledModelNames = new String[preInstalledModelPaths.length];
        for (int i = 0; i < preInstalledModelPaths.length; i++) {
            preInstalledModelNames[i] =
                    preInstalledModelPaths[i].substring(preInstalledModelPaths[i].lastIndexOf("/") + 1);
        }
        lpChoosePreInstalledModel.setEntries(preInstalledModelNames);
        lpChoosePreInstalledModel.setEntryValues(preInstalledModelPaths);
        cbEnableCustomModelSettings =
                (CheckBoxPreference) findPreference(getString(R.string.ICS_ENABLE_CUSTOM_MODEL_SETTINGS_KEY));
        etModelPath = (EditTextPreference) findPreference(getString(R.string.ICS_MODEL_PATH_KEY));
        etModelPath.setTitle("Model Path (SDCard: " + Utils.getSDCardDirectory() + ")");
        etLabelPath = (EditTextPreference) findPreference(getString(R.string.ICS_LABEL_PATH_KEY));
        etImagePath = (EditTextPreference) findPreference(getString(R.string.ICS_IMAGE_PATH_KEY));
        etInputShape = (EditTextPreference) findPreference(getString(R.string.ICS_INPUT_SHAPE_KEY));
        etInputMean = (EditTextPreference) findPreference(getString(R.string.ICS_INPUT_MEAN_KEY));
        etInputScale = (EditTextPreference) findPreference(getString(R.string.ICS_INPUT_SCALE_KEY));
    }

    private void reloadPreferenceAndUpdateUI() {
        SharedPreferences sharedPreferences = getPreferenceScreen().getSharedPreferences();
        boolean enableCustomModelSettings =
                sharedPreferences.getBoolean(getString(R.string.ICS_ENABLE_CUSTOM_MODEL_SETTINGS_KEY), false);
        String modelPath = sharedPreferences.getString(getString(R.string.ICS_CHOOSE_PRE_INSTALLED_MODEL_KEY),
                getString(R.string.ICS_MODEL_PATH_DEFAULT));
        int modelIdx = lpChoosePreInstalledModel.findIndexOfValue(modelPath);
        if (modelIdx >= 0 && modelIdx < preInstalledModelPaths.length) {
            if (!enableCustomModelSettings) {
                SharedPreferences.Editor editor = sharedPreferences.edit();
                editor.putString(getString(R.string.ICS_MODEL_PATH_KEY), preInstalledModelPaths[modelIdx]);
                editor.putString(getString(R.string.ICS_LABEL_PATH_KEY), preInstalledLabelPaths[modelIdx]);
                editor.putString(getString(R.string.ICS_IMAGE_PATH_KEY), preInstalledImagePaths[modelIdx]);
                editor.putString(getString(R.string.ICS_INPUT_SHAPE_KEY), preInstalledInputShapes[modelIdx]);
                editor.putString(getString(R.string.ICS_INPUT_MEAN_KEY), preInstalledInputMeans[modelIdx]);
                editor.putString(getString(R.string.ICS_INPUT_SCALE_KEY), preInstalledInputScales[modelIdx]);
                editor.commit();
            }
            lpChoosePreInstalledModel.setSummary(modelPath);
        }
        cbEnableCustomModelSettings.setChecked(enableCustomModelSettings);
        etModelPath.setEnabled(enableCustomModelSettings);
        etLabelPath.setEnabled(enableCustomModelSettings);
        etImagePath.setEnabled(enableCustomModelSettings);
        etInputShape.setEnabled(enableCustomModelSettings);
        etInputMean.setEnabled(enableCustomModelSettings);
        etInputScale.setEnabled(enableCustomModelSettings);
        modelPath = sharedPreferences.getString(getString(R.string.ICS_MODEL_PATH_KEY),
                getString(R.string.ICS_MODEL_PATH_DEFAULT));
        String labelPath = sharedPreferences.getString(getString(R.string.ICS_LABEL_PATH_KEY),
                getString(R.string.ICS_LABEL_PATH_DEFAULT));
        String imagePath = sharedPreferences.getString(getString(R.string.ICS_IMAGE_PATH_KEY),
                getString(R.string.ICS_IMAGE_PATH_DEFAULT));
        String inputShape = sharedPreferences.getString(getString(R.string.ICS_INPUT_SHAPE_KEY),
                getString(R.string.ICS_INPUT_SHAPE_DEFAULT));
        String inputMean = sharedPreferences.getString(getString(R.string.ICS_INPUT_MEAN_KEY),
                getString(R.string.ICS_INPUT_MEAN_DEFAULT));
        String inputScale = sharedPreferences.getString(getString(R.string.ICS_INPUT_SCALE_KEY),
                getString(R.string.ICS_INPUT_SCALE_DEFAULT));
        etModelPath.setSummary(modelPath);
        etModelPath.setText(modelPath);
        etLabelPath.setSummary(labelPath);
        etLabelPath.setText(labelPath);
        etImagePath.setSummary(imagePath);
        etImagePath.setText(imagePath);
        etInputShape.setSummary(inputShape);
        etInputShape.setText(inputShape);
        etInputMean.setSummary(inputMean);
        etInputMean.setText(inputMean);
        etInputScale.setSummary(inputScale);
        etInputScale.setText(inputScale);
    }

    @Override
    protected void onResume() {
        super.onResume();
        getPreferenceScreen().getSharedPreferences().registerOnSharedPreferenceChangeListener(this);
        reloadPreferenceAndUpdateUI();
    }

    @Override
    protected void onPause() {
        super.onPause();
        getPreferenceScreen().getSharedPreferences().unregisterOnSharedPreferenceChangeListener(this);
    }

    @Override
    public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String key) {
        if (key.equals(getString(R.string.ICS_CHOOSE_PRE_INSTALLED_MODEL_KEY))) {
            SharedPreferences.Editor editor = sharedPreferences.edit();
            editor.putBoolean(getString(R.string.ICS_ENABLE_CUSTOM_MODEL_SETTINGS_KEY), false);
            editor.commit();
        }
        reloadPreferenceAndUpdateUI();
    }
}
