#!/bin/bash
# push
adb -s bcd71650 push ./ocr_demo_exec /data/local/tmp/
ocr_demo_path="/data/local/tmp/ocr_demo_exec"

# run
adb -s bcd71650 shell "cd ${ocr_demo_path} \
           && chmod +x ./pipeline \
           && export LD_LIBRARY_PATH=${ocr_demo_path}:${LD_LIBRARY_PATH} \
           && ./pipeline \
                ./models/ch_ppocr_mobile_v2.0_det_slim_opt.nb \
                ./models/ch_ppocr_mobile_v2.0_rec_slim_opt.nb \
                ./models/ch_ppocr_mobile_v2.0_cls_slim_opt.nb \
                ./images/test.jpg \
                ./test_img_result.jpg \
                ./labels/ppocr_keys_v1.txt \
                ./config.txt"

adb -s bcd71650 pull ${ocr_demo_path}/test_img_result.jpg .
