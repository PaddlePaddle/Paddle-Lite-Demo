package com.baidu.paddle.lite.demo.face_detection.visual;


public class NMS {
    private final static float INVALID_ANCHOR = -10000.0f;

    private static void quickSortScore(float[][] anchors, float[] scores, int left, int right) {
        int dp;
        if (left < right) {
            dp = partitionScore(anchors, scores, left, right);
            quickSortScore(anchors, scores, left, dp - 1);
            quickSortScore(anchors, scores, dp + 1, right);
        }
    }

    private static int partitionScore(float[][] anchors, float scores[], int left, int right) {
        float pivot = scores[left];
        float[] pivotA = anchors[left];
        while (left < right) {
            while (left < right && scores[right] <= pivot)
                right--;
            if (left < right) {
                anchors[left] = anchors[right];
                scores[left++] = scores[right];
            }

            while (left < right && scores[left] >= pivot)
                left++;
            if (left < right) {
                anchors[right] = anchors[left];
                scores[right--] = scores[left];
            }
        }
        scores[left] = pivot;
        anchors[left] = pivotA;
        return left;
    }

    private static float computeOverlapAreaRate(float[] anchor1, float[] anchor2){
        float xx1 = anchor1[0]>anchor2[0]?anchor1[0]:anchor2[0];
        float yy1 = anchor1[1]>anchor2[1]?anchor1[1]:anchor2[1];
        float xx2 = anchor1[2]<anchor2[2]?anchor1[2]:anchor2[2];
        float yy2 = anchor1[3]<anchor2[3]?anchor1[3]:anchor2[3];

        float w = xx2 - xx1 + 1;
        float h = yy2 - yy1 + 1;
        if(w<0||h<0){
            return 0;
        }

        float inter = w * h;

        float anchor1_area1 = (anchor1[2] - anchor1[0] + 1)*(anchor1[3] - anchor1[1] + 1);
        float anchor2_area1 = (anchor2[2] - anchor2[0] + 1)*(anchor2[3] - anchor2[1] + 1);

        return inter / (anchor1_area1 + anchor2_area1 - inter);
    }

    public static int[] nmsScoreFilter(float[][] anchors, float[] score, int topN, float thresh){
        int length = anchors.length;
        int count = 0;

        for(int i=0;i<length;i++){
            if(score[i]==INVALID_ANCHOR){
                continue;
            }
            if (++count >= topN) {
                break;
            }
            for(int j=i+1;j<length;j++){
                if(score[j]!=INVALID_ANCHOR) {
                    if (computeOverlapAreaRate(anchors[i], anchors[j]) > thresh) {
                        score[j] = INVALID_ANCHOR;
                    }
                }
            }
        }

        int outputIndex[] = new int[count];
        int j = 0;
        for(int i=0;i<length && count>0;i++){
            if(score[i]!=INVALID_ANCHOR){
                outputIndex[j++] = i;
                count--;
            }
        }
        return outputIndex;
    }

    public static void sortScores(float[][] anchors, float[] scores){
        quickSortScore(anchors, scores, 0, scores.length - 1);
    }
}