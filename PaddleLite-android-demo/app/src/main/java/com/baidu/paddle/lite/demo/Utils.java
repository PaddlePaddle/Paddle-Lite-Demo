package com.baidu.paddle.lite.demo;

import android.content.Context;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

public class Utils {
    private static final String TAG = Utils.class.getSimpleName();

    public static String copyFromAssetsToCache(Context appCtx, String filePath) {
        String newPath = appCtx.getCacheDir() + "/" + filePath;
        File desDir = new File(newPath);
        try {
            if (!desDir.exists()) {
                desDir.mkdirs();
            }
            for (String fileName : appCtx.getAssets().list(filePath)) {
                InputStream stream = appCtx.getAssets().open(filePath + "/" + fileName);
                OutputStream output = new BufferedOutputStream(new FileOutputStream(newPath + "/" + fileName));
                byte data[] = new byte[1024];
                int count;
                while ((count = stream.read(data)) != -1) {
                    output.write(data, 0, count);
                }
                output.flush();
                output.close();
                stream.close();
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return desDir.getPath();
    }

    public static boolean isSupportNPU() {
        String hardware = android.os.Build.HARDWARE;
        return hardware.equalsIgnoreCase("kirin810");
    }
}
