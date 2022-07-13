# Machine-Learning-HTRU2-Analysis

The main goal of this work is to make a comparison between the most common techniques in Machine Learning in the classification of the HTRU dataset. 
In the first part, after an introduction to the dataset, we analyze the features to understand their distribution and their correlation and how to manipulate them in order to obtain better results. Then, we discuss the training of the models, and we choose the best ones. In the end, we do the score calibration of these and we see how they actually work on the testing dataset to see if the results are what we expected.

The dataset is split into two files: Train.txt and Test.txt, inside the folder data.
The analyzed pre-processing techniques are:
-	Z-normalization
- Gaussianization
- PCA

The analyzed models are:
-	MVG (Full, Naive Bayes, Tied, Tied Naive Bayes)
-	Prior Weighted Logistic Regression (Linear and Quadratic, through feature expansion)
-	Balanced SVM (Linear, Quadratic kernel, RBF Kernel)
-	GMM (Full, Diagonal, Tied, Tied Diagonal)

