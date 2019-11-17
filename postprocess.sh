#!/bin/sh
#
# This scripts selects the interesting data and outputs the results in csv format.

set -x
for i in *.out; do

  grep -e "gr-truth$(printf "\t")0" $i > $i.posttmp2

  #grep -e "baseline" $i > $i.posttmp
  grep -e "gr-truth" $i > $i.posttmp

  { grep -e "Sample dimensions" $i.posttmp  | sed "s/$/\t\//" &&
    grep -e "Highest dispersion" $i.posttmp  | sed "s/$/\t\//" &&
    grep -e "PCA"  $i.posttmp  | sed "s/$/\t\//" &&
    grep -e "Factor Analysis" $i.posttmp  | sed "s/$/\t\//" &&
    grep -e "Coin flip" $i.posttmp &&
    grep -e "Fast JL Transform" $i.posttmp; } >> $i.posttmp2

  { grep -e "n$(printf '\t')d" $i &&
    cat $i.posttmp2; } > $i.posttmp3
  ./postprocess-helper.py $i.posttmp3 > $i.posttmp;
  # ./postprocess-helper.py $i.posttmp3; exit 0

  { echo "n,d,n_clust,test_pr,eval_against,method,methodname,k,k/d,vmeas,vm_min,vm_max,t_red,tred_min,tred_max,t_test,ttest_min,ttest_max,t_tot,ttot_min,ttot_max,eps" &&  grep "kmeans_" $i.posttmp; } > $i-kmeans.csv
  { echo "n,d,n_clust,test_pr,eval_against,method,methodname,k,k/d,vmeas,vm_min,vm_max,t_red,tred_min,tred_max,t_test,ttest_min,ttest_max,t_tot,ttot_min,ttot_max,eps" &&  grep "kneigh_" $i.posttmp; } > $i-kneigh.csv

  # rm $i.posttmp $i.posttmp2 $i.posttmp3
done

exit 0;
