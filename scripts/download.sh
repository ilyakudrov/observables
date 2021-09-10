#!/bin/bash
conf_type="qc2dstag"
smearing="HYP_APE"
observable="wilson_loop"
for mu in 5 #0 10 15 25 30 40
do
w=$(($mu/10))
q=$(($mu-$w*10))
rsync -r kudrov@rrcmpi.itep.ru:/home/clusters/rrcmpi/kudrov/observables/result/${observable}/${conf_type}/${smearing}/mu0.$w$q ../data/${observable}/${conf_type}/${smearing}
#done
#done
done
