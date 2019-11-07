#!/bin/sh
#

set -x
for i in *.out; do

  grep -e "gr-truth$(printf '\t')-" $i | sed 's/\([0-9]\+\t\([0-9]\+\)\t[0-9]\+\t\w\+\tgr-truth\t\)-\t-\t-\t\([0-9\.]\+\t[0-9\.]\+\t[0-9\.]\+\)/\1Base\t\2\t1.00\t\3\t\\/g' > $i.posttmp2

  grep -e "baseline" $i > $i.posttmp
  { grep -e "sampDim" $i.posttmp  | sed "s/$/\//" &&
    grep -e "HVarDim" $i.posttmp  | sed "s/$/\//" &&
    grep -e "Achliop" $i.posttmp &&
    grep -e "FastJLT" $i.posttmp; } >> $i.posttmp2

  { grep -e "n$(printf '\t')d" $i | sed "s/$/\teps/" &&
    cat $i.posttmp2; } > $i.posttmp3
  ./postprocess-helper.py $i.posttmp3 > $i.posttmp;
  # ./postprocess-helper.py $i.posttmp3; exit 0
  { echo "n,d,n_clust,test_pr,eval_against,method,methodname,k,k/d,vmeas,vmeas_e1,time,eps" &&  grep "kmeans_" $i.posttmp; } > $i-kmeans.csv
  { echo "n,d,n_clust,test_pr,eval_against,method,methodname,k,k/d,vmeas,vmeas_e1,time,eps" &&  grep "kneigh_" $i.posttmp; } > $i-kneigh.csv

  rm $i.posttmp $i.posttmp2 $i.posttmp3
done

exit 0;
