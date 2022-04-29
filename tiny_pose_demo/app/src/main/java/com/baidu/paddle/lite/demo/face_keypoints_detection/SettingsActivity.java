package com.baidu.paddle.lite.demo.face_keypoints_detection;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.preference.EditTextPreference;
import android.preference.ListPreference;
import android.preference.PreferenceManager;
import android.support.v7.app.ActionBar;
import android.util.Log;

import com.baidu.paddle.lite.demo.common.AppCompatPreferenceActivity;
import com.baidu.paddle.lite.demo.common.Utils;

import java.util.ArrayList;
import java.util.List;

public class SettingsActivity extends AppCompatPreferenceActivity implements SharedPreferences.OnSharedPreferenceChangeListener {
    private static final String TAG = SettingsActivity.class.getSimpleName();

    ListPreference lpBackend = null;
    List<String> preInstalledBackend = null;
    static public String backend = "";
    static public int fdtSelectedModelIdx = 0;
    static public String fdtModelDir = "";
    static public int fdtCPUThreadNum = 1;
    static public String fdtCPUPowerMode = "LITE_POWER_HIGH";
    static public float fdtInputScale = 0.25f;
    static public float[] fdtInputMean = new float[]{0.407843f, 0.694118f, 0.482353f};
    static public float[] fdtInputStd = new float[]{0.5f, 0.5f, 0.5f};
    static public float fdtScoreThreshold = 0.7f;

    static public int fkpSelectedModelIdx = 0;
    static public String fkpModelDir = "";
    static public int fkpCPUThreadNum = 0;
    static public String fkpCPUPowerMode = "";
    static public int fkpInputWidth = 0;
    static public int fkpInputHeight = 0;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        addPreferencesFromResource(R.xml.settings);
        ActionBar supportActionBar = getSupportActionBar();
        if (supportActionBar != null) {
            supportActionBar.setDisplayHomeAsUpEnabled(true);
        }

        preInstalledBackend = new ArrayList<String>();
        preInstalledBackend.add(getString(R.string.BACKEND_DEFAULT));
        lpBackend =
                (ListPreference) findPreference(getString(R.string.BACKEND_KEY));
    }

    private void reloadSettingsAndUpdateUI() {
        SharedPreferences sharedPreferences = getPreferenceScreen().getSharedPreferences();

        String backend = sharedPreferences.getString(getString(R.string.BACKEND_KEY),
                getString(R.string.BACKEND_DEFAULT));
        Log.d(TAG, "reloadSettingsAndUpdateUI :" + backend);
        lpBackend.setValue(backend);
        lpBackend.setSummary(backend);
    }

    static boolean checkAndUpdateSettings(Context ctx) {
        boolean settingsChanged = false;
        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(ctx);

        String backendSetting = sharedPreferences.getString(ctx.getString(R.string.BACKEND_KEY),
                ctx.getString(R.string.BACKEND_DEFAULT));
        settingsChanged |= !backend.equalsIgnoreCase(backendSetting);
        backend = backendSetting;

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
