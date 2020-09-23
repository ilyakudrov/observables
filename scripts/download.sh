#!/bin/bash
type="qc2dstag"
smearing="/HYP_APE"
observable="wilson_loop"
for mu in 5 #0 10 15 25 30 40
do
w=$(($mu/10))
q=$(($mu-$w*10))
rsync -r kudrov@rrcmpi-a.itep.ru:/home/clusters/rrcmpi/kudrov/observables/get_result/result/${observable}/${type}${smearing}/mu0.$w$q /home/ilya/lattice/observables/data/${observable}/${type}${smearing}
#done
#done
done
