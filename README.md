This is README file for ML-2020fall-final.
======
Team 18

Task: House Prices: Advanced Regression Technology
Author: 盧志賢 B06502167 楊欣睿 B06901080 劉佳婷 B06901105
Date: 2021/01/15
======
SYNOPSIS:

bash final_best.sh <training_data_set> <testing_test_set> <prediction>
Example: bash final_best.sh ./data/train.csv ./data/test.csv ./submit.csv
This program predicts the price of each house in the testing data set with 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa.
======
DIRECTORY:

src/ 	  source Python codes
data/     training data and testing data
======
HOW TO EXECUTE:

To execute the codes which predict the best result, simply follow the following steps
	bash final_best.sh <training_data_set> <testing_test_set> <prediction_set>
    For example,
    bash final_best.sh ./data/train.csv ./data/test.csv ./submit.csv
To execute the codes which use the different model, simply follow the following steps
	bash final_NN.sh <training_data_set> <testing_test_set> <prediction_set>
    For example,
    bash final_NN.sh ./data/train.csv ./data/test.csv ./submit.csv
======
TOOL-KITS VERSION

# Name               Version
Python               3.6.5
numpy                1.19.2
pandas               1.1.1
matplotlib           2.2.2
seaborn              0.11.0
scikit-learn         0.23.0
scikit_image         0.14.2
keras                2.3.1
keras-applications   1.0.8
keras-base           2.3.1
keras-prepocessing   1.1.0
tensorflow           1.13.1
tensorflow-base      1.13.1
tensorflow-estimator 1.13.0
tensorflow-gpu       1.13.1