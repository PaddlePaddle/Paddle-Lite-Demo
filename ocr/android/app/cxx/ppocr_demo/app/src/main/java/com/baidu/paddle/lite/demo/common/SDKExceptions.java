package com.baidu.paddle.lite.demo.common;

public class SDKExceptions {
    public static class NoSDCardPermission extends Exception {

    }

    public static class MissingModleFileInAssetFolder extends Exception {
    }

    public static class ModelInitError extends Exception {
    }

    public static class NotInit extends Throwable {
    }

    public static class LoadLicenseLibraryError extends Throwable {
    }

    public static class LoadNativeLibraryError extends Throwable {
    }

    public static class NV21BytesLengthNotMatch extends Exception {
    }

    public static class PathNotExist extends Throwable {
    }
}
