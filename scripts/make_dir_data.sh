#!/bin/bash
# wilon_loop and flux_tube

for observable in "flux_tube" "wilson_loop"
do
for conf_type in "mon_wl" "mag" "offd" "qc2dstag" "su2"
do
for smearing in "HYP_APE" "unsmeared"
do
for mu in 0 5 10 15 25 30 40
do
w=$(($mu/10))
q=$(($mu-$w*10))
mkdir -p ../data/${observable}/${conf_type}/${smearing}/mu0.$w$q
done
done
done
done

# smearing_test

for observable in "flux_tube" "wilson_loop"
do
for conf_type in "mon_wl" "mag" "offd" "qc2dstag" "su2"
do
for smearing in "HYP_APE"
do
for mu in 0 5 10 15 25 30 40
do
w=$(($mu/10))
q=$(($mu-$w*10))
mkdir -p ../data/smearing_test/${observable}/${conf_type}/${smearing}/mu0.$w$q
done
done
done
done
