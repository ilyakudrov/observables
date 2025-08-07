#!/bin/bash
dir_path="../../../../data/gluon_propagator/su2/gluodynamics/66^3x8/beta2.701/original"
file_start="gluon_propagator_"
file_end=""
output_path="./result/gluon_propagator_color_optimized.csv"
padding=4
num_max=1000
../gluon_propagator_color_optimized_test --dir_path ${dir_path} --file_start ${file_start} --file_end "${file_end}" --output_path ${output_path} --padding ${padding} --num_max ${num_max}