#!/usr/bin/env bash

# echo "Concatenating files..."
# cat $1 knowledge_styletip.out knowledge_attribute.out knowledge_celebrity.out > all_text.out

# echo "Spliting files..."
python tools/split.py $1

# echo "Converting xml..."
python tools/convert.py src text true pred $1'.true'
python tools/convert.py ref text true pred $1'.true'
python tools/convert.py tst text true pred $1'.pred'

# echo "Evaluating..."
perl tools/mteval-v14.pl -s $1'.true_src.xml' -r $1'.true_ref.xml' -t $1'.pred_tst.xml'
