# This script is untested, but should work (and, otherwise, give the general idea
# for how to construct the ISIC2018 data directory)
mkdir ISIC2018
wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip
unzip ISIC2018_Task1-2_Training_Input.zip
rm ISIC2018_Task1-2_Training_Input.zip
mv ISIC2018_Task1-2_Training_Input ISIC2018
wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip
unzip ISIC2018_Task1_Training_GroundTruth.zip
rm ISIC2018_Task1_Training_GroundTruth.zip
mv ISIC2018_Task1_Training_GroundTruth ISIC2018
wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip
unzip ISIC2018_Task3_Training_Input.zip
rm ISIC2018_Task3_Training_Input.zip
mv ISIC2018_Task3_Training_Input ISIC2018
wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip
unzip ISIC2018_Task3_Training_GroundTruth.zip
rm ISIC2018_Task3_Training_GroundTruth.zip
mv ISIC2018_Task3_Training_GroundTruth ISIC2018