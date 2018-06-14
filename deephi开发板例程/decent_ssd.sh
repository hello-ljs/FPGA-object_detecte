#!/usr/bin/env bash

#working directory
work_dir=$(pwd)
#path of float model
model_dir=${work_dir}
#output directory
output_dir=${work_dir}/decent_output

decent     fix                                    \
           -model ${model_dir}/deploy.prototxt     \
           -weights ${model_dir}/VGG_VOC2018_SSD_300x300_iter_28000.caffemodel \
           -output_dir ${output_dir} \
           -method 1
