#!/bin/bash
# push
adb push ./ocr_demo_exec /data/local/tmp/
ocr_demo_path="/data/local/tmp/ocr_demo_exec"

# run
adb shell "cd ${ocr_demo_path} \
           && chmod +x ./ocr_db_crnn \
           && export LD_LIBRARY_PATH=${ocr_demo_path}:${LD_LIBRARY_PATH} \
           && ./ocr_db_crnn \
                ./models/ch_ppocr_mobile_v2.0_det_slim_opt.nb \
                ./models/ch_ppocr_mobile_v2.0_rec_slim_opt.nb \
                ./models/ch_ppocr_mobile_v2.0_cls_slim_opt.nb \
                ./images/test.jpg \
                ./test_img_result.jpg \
                ./labels/ppocr_keys_v1.txt"

adb pull ${ocr_demo_path}/test_img_result.jpg .
