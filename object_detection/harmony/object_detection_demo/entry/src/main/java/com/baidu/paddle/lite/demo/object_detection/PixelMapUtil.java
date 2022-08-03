package com.baidu.paddle.lite.demo.object_detection;

import ohos.agp.render.Canvas;
import ohos.agp.render.Paint;
import ohos.agp.render.PixelMapHolder;
import ohos.agp.render.Texture;
import ohos.agp.utils.Matrix;
import ohos.app.Context;
import ohos.app.Environment;
import ohos.global.resource.RawFileEntry;
import ohos.media.image.ImagePacker;
import ohos.media.image.ImageSource;
import ohos.media.image.PixelMap;
import ohos.media.image.common.PixelFormat;

import java.io.*;

public final class PixelMapUtil {
    private static final int DEGREE_90 = 90;
    private static final int FIVE = 5;

    private static PixelMap createPixelMap(int width, int height) {
        PixelMap pixelMap;
        PixelMap.InitializationOptions options =
                new PixelMap.InitializationOptions();
        options.size = new ohos.media.image.common.Size(width, height);
        options.pixelFormat = PixelFormat.ARGB_8888;
        options.editable = true;
        pixelMap = PixelMap.create(options);
        return pixelMap;
    }

    public static PixelMap adjustRotation(PixelMap map, int degree) {
        Matrix matrix = new Matrix();
        matrix.setRotate(degree, (float) map.getImageInfo().size.width / 2,
                (float) map.getImageInfo().size.height / 2);
        float outputX;
        float outputY;
        if (degree == DEGREE_90) {
            outputX = map.getImageInfo().size.height;
            outputY = 0;
        } else {
            outputX = map.getImageInfo().size.height;
            outputY = map.getImageInfo().size.width;
        }
        float[] values = matrix.getData();
        float x1 = values[2];
        float y1 = values[FIVE];
        matrix.postTranslate(outputX - x1, outputY - y1);
        ohos.media.image.common.Size size = map.getImageInfo().size;
        PixelMap pixelMap = createPixelMap(size.height, size.width);
        Paint paint = new Paint();
        Canvas canvas = new Canvas(new Texture(pixelMap));
        canvas.setMatrix(matrix);
        canvas.drawPixelMapHolder(new PixelMapHolder(map), 0, 0, paint);
        return pixelMap;
    }

    public static PixelMap getPixelMapByResPath(Context context, String src) throws IOException {
        PixelMap pixelMap = null;
        InputStream resource = null;
        RawFileEntry rawFileEntry =
                context.getResourceManager().getRawFileEntry(src);
        resource = rawFileEntry.openRawFile();
        ImageSource imageSource = ImageSource.create(resource, null);
        ImageSource.DecodingOptions decodingOpts =
                new ImageSource.DecodingOptions();
        pixelMap = imageSource.createPixelmap(decodingOpts);
        resource.close();
        if (resource != null) {
            resource.close();
        }
        return pixelMap;
    }

    public static PixelMap getPixelMapByFilePath(Context context, String externalFile) throws IOException {
        PixelMap pixelMap = null;
        InputStream resource = null;
        ImageSource imageSource = ImageSource.create(externalFile, null);
        ImageSource.DecodingOptions decodingOpts =
                new ImageSource.DecodingOptions();
        pixelMap = imageSource.createPixelmap(decodingOpts);
        resource.close();
        if (resource != null) {
            resource.close();
        }
        return pixelMap;
    }

    public static boolean savePixelMap(Context context, PixelMap pixelMap, String filename){
        ImagePacker imagePacker = ImagePacker.create();
        File file = new File(context.getExternalFilesDir(Environment.DIRECTORY_PICTURES), filename);
        FileOutputStream outputStream = null;
        try {
            outputStream = new FileOutputStream(file);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        ImagePacker.PackingOptions packingOptions = new ImagePacker.PackingOptions();
        packingOptions.format = "image/jpeg";
        packingOptions.quality = 100;// 设置图片质量
        boolean result = imagePacker.initializePacking(outputStream, packingOptions);
        result = imagePacker.addImage(pixelMap);
        long dataSize = imagePacker.finalizePacking();
        return result;
    }
}
