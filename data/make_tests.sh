# Randomly segment the files corpus2_train.labels and corpus3_train.labels so as
# to create a similar file structure to corpus1, with 1/3 train-test split.
#
# Run on a Mac OSX (note the use of gshuf, ghead and gtail). Simply remove the
# g's to run on any other OS.

gshuf corpus2_train.labels > segmented_corpus2_train.labels
gtail -n 298 segmented_corpus2_train.labels > segmented_corpus2_test.labels
ghead -n -298 segmented_corpus2_train.labels > temp.txt
mv temp.txt segmented_corpus2_train.labels
cp segmented_corpus2_test.labels segmented_corpus2_test.list
sed 's/ .$//' segmented_corpus2_test.list > temp.txt
mv temp.txt segmented_corpus2_test.list

gshuf corpus3_train.labels > segmented_corpus3_train.labels
gtail -n 318 segmented_corpus3_train.labels > segmented_corpus3_test.labels
ghead -n -318 segmented_corpus3_train.labels > temp.txt
mv temp.txt segmented_corpus3_train.labels
cp segmented_corpus3_test.labels segmented_corpus3_test.list
sed 's/ ...$//' segmented_corpus3_test.list > temp.txt
mv temp.txt segmented_corpus3_test.list
