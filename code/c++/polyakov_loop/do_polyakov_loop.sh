#!/bin/bash
gauge_group="su2"
theory="qc2dstag"
beta=""
mu="mu0.05"
decomposition="original"
#additional_parameters="steps_0/copies=20"
additional_parameters="/"
#lattice_size=( "24^4" "24^4" "32^3x16" "32^3x20"  "32^3x24" "32^3x28" "32^3x28" "32^3x32" "32^3x32" "32^3x36" "32^3x36" "32^3x40" "32^3x40" "32^3x8" "32^4" "32^4" "32^4" "32^4" )
#mu=( "mu0.05" "mu0.30" "mu0.15" "mu0.15" "mu0.15" "mu0.15" "mu0.20" "mu0.15" "mu0.20" "mu0.15" "mu0.20" "mu0.15" "mu0.20" "mu0.15" "mu0.00" "mu0.25" "mu0.30" "mu0.40" )
lattice_size=( "32^3x40" )
mu=( "mu0.15" )


for i in "${!lattice_size[@]}"; do
smearing="HYP10_alpha=1_1_0.5"
dir_path="/home/ilya/soft/lattice/observables/data/smearing/polyakov_loop/${gauge_group}/${theory}/${lattice_size[i]}/${beta}/${mu[i]}/${original}/${decomposition}/${smearing}/${additional_parameters}"
file_start="polyakov_loop_"
file_end=""
output_path_start="/home/ilya/soft/lattice/observables/result/smearing/polyakov_loop/${gauge_group}/${theory}/${lattice_size[i]}/${beta}/${mu[i]}/${original}/${decomposition}/${smearing}/${additional_parameters}"
mkdir -p ${output_path_start}
output_path="${output_path_start}/polyakov_loop.csv"
padding=4
num_max=6000
/home/ilya/soft/lattice/observables/code/c++/polyakov_loop/polyakov_loop_test --dir_path ${dir_path} --file_start ${file_start}\
 --file_end "${file_end}" --output_path ${output_path} --padding ${padding} --num_max ${num_max}
done
