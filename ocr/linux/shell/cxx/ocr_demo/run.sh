#!/bin/bash
# push
cd ./ocr_demo_exec
chmod +x ./pipeline
export LD_LIBRARY_PATH=./
./pipeline ./models/ch_ppocr_mobile_v2.0_det_slim_opt.nb ./models/ch_ppocr_mobile_v2.0_rec_slim_opt.nb ./models/ch_ppocr_mobile_v2.0_cls_slim_opt.nb ./images/test.jpg ./test_img_result.jpg ./labels/ppocr_keys_v1.txt ./config.txt
