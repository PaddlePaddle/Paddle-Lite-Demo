#!/bin/bash
# push
adb push ./ppocr_demo /data/local/tmp/
ppocr_demo_path="/data/local/tmp/ppocr_demo"

# run
adb shell "cd ${ppocr_demo_path} \
           && chmod +x ./ppocr_demo \
           && export LD_LIBRARY_PATH=${ppocr_demo_path}:${LD_LIBRARY_PATH} \
           && ./ppocr_demo \
                ./models/ch_ppocr_mobile_v2.0_det_slim_opt.nb \
                ./models/ch_ppocr_mobile_v2.0_rec_slim_opt.nb \
                ./models/ch_ppocr_mobile_v2.0_cls_slim_opt.nb \
                ./images/test.jpg \
                ./test_img_result.jpg \
                ./labels/ppocr_keys_v1.txt \
                ./config.txt"

adb pull ${ppocr_demo_path}/test_img_result.jpg .
