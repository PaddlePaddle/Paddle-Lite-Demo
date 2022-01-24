#!/bin/bash
python ./image_classification.py --model_dir ../../../assets/models/mobilenet_v1_for_cpu/model.nb \
--input_shape=1,3,224,224 --image_path ../../../assets/images/tabby_cat.jpg \
--label_path ../../../assets/labels/labels.txt \
--topk=3 --repeats=100 --warmup=10
