package com.baidu.paddle.lite.demo.object_detection;

import ohos.agp.utils.RectFloat;

import java.util.Locale;

public class Recognition {
    /**
     * A unique identifier for what has been recognized. Specific to the class, not the instance of
     * the object.
     */
    private final String id;

    /** Display name for the recognition. */
    private final String title;

    /**
     * A sortable score for how good the recognition is relative to others. Higher should be better.
     */
    private final Float confidence;

    /** Location within the source image for the location of the recognized object. */
    private RectFloat location;

    /** Detected class of the recognized object. */
    private int classId;

    public Recognition(
            final String id,
            final String title,
            final Float confidence,
            final RectFloat location,
            int classId) {
        this.id = id;
        this.title = title;
        this.confidence = confidence;
        this.location = location;
        this.classId = classId;
    }

    public String getId() {
        return id;
    }

    public String getTitle() {
        return title;
    }

    public Float getConfidence() {
        return confidence;
    }

    public RectFloat getLocation() {
        return new RectFloat(location);
    }

    public void setLocation(RectFloat location) {
        this.location = location;
    }

    public int getClassId() {
        return classId;
    }

    @Override
    public String toString() {
        String resultString = "";
        if (id != null) {
            resultString += "[" + id + "] ";
        }

        if (title != null) {
            resultString += title + " ";
        }

        if (confidence != null) {
            resultString += String.format(Locale.US, "(%.1f%%) ", confidence * 100.0f);
        }

        if (location != null) {
            resultString += location + " ";
        }

        return resultString.trim();
    }
}
