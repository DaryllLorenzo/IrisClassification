# IrisClassification
Iris dataset classification with multilayer Neural Net

## Iris Classification with FFNN and PCA
Feedforward Neural Network (FFNN) implementation for classifying the Iris dataset. A key aspect of this project is the use of Principal Component Analysis (PCA) for dimensionality reduction.

▌Overview

The goal of this project is to build a robust and interpretable classifier for the Iris dataset. We leverage PCA to reduce the dimensionality of the data while retaining the most important information for classification. The model is a simple FFNN trained on the reduced feature set.

▌Data Preprocessing

Before applying PCA, the data was standardized using StandardScaler. This ensures that each feature contributes equally to the PCA without being influenced by its scale. The standardization is crucial for optimal PCA performance.

▌Principal Component Analysis (PCA)

We applied Principal Component Analysis (PCA) to reduce the original four features of the Iris dataset to two principal components (PC1 and PC2). This dimensionality reduction simplifies the model and can potentially improve generalization.

Standardization:
As mentioned above, before applying PCA, we used StandardScaler from scikit-learn to standardize the features. This is important because PCA is sensitive to the scale of the features. Standardization ensures that each feature has a mean of 0 and a standard deviation of 1, preventing features with larger scales from dominating the principal components.

Principal Components Visualization:

The following plot visualizes the data projected onto the first two principal components (PC1 and PC2):

![First plot](/images/PCA/1.png)

Explanation: This plot shows how the different Iris species are distributed in the reduced two-dimensional space. You can visually assess the separability of the classes based on these two components.

▌Model Architecture and Training

The classification model is a Feedforward Neural Network (FFNN) with one hidden layer. The architecture is as follows:

*  Input layer: 2 neurons (corresponding to the 2 principal components)
*  Hidden layer: 128 neurons, ReLU activation
*  Output layer: 3 neurons (corresponding to the 3 Iris species), Softmax activation

Model Performance:

The model was trained for 800 epochs. The following results were achieved:

Epoch: 0 | Loss:  1.0664 | Acc:  0.54% | Test_loss:  1.0495 | Test_acc:  0.67%

Epoch: 10 | Loss:  0.9073 | Acc:  0.82% | Test_loss:  0.8931 | Test_acc:  0.93%

Epoch: 20 | Loss:  0.8025 | Acc:  0.86% | Test_loss:  0.7841 | Test_acc:  0.93%

Epoch: 30 | Loss:  0.7250 | Acc:  0.86% | Test_loss:  0.7010 | Test_acc:  0.93%

Epoch: 40 | Loss:  0.6650 | Acc:  0.85% | Test_loss:  0.6356 | Test_acc:  0.93%

Epoch: 50 | Loss:  0.6169 | Acc:  0.85% | Test_loss:  0.5829 | Test_acc:  0.93%

Epoch: 60 | Loss:  0.5770 | Acc:  0.85% | Test_loss:  0.5391 | Test_acc:  0.90%

Epoch: 70 | Loss:  0.5432 | Acc:  0.85% | Test_loss:  0.5023 | Test_acc:  0.93%

Epoch: 80 | Loss:  0.5143 | Acc:  0.85% | Test_loss:  0.4708 | Test_acc:  0.93%

Epoch: 90 | Loss:  0.4892 | Acc:  0.85% | Test_loss:  0.4437 | Test_acc:  0.93%

Epoch: 100 | Loss:  0.4674 | Acc:  0.85% | Test_loss:  0.4201 | Test_acc:  0.93%

Epoch: 110 | Loss:  0.4481 | Acc:  0.85% | Test_loss:  0.3992 | Test_acc:  0.93%

Epoch: 120 | Loss:  0.4310 | Acc:  0.87% | Test_loss:  0.3806 | Test_acc:  0.93%

Epoch: 130 | Loss:  0.4157 | Acc:  0.88% | Test_loss:  0.3640 | Test_acc:  0.93%

Epoch: 140 | Loss:  0.4021 | Acc:  0.88% | Test_loss:  0.3490 | Test_acc:  0.93%

Epoch: 150 | Loss:  0.3898 | Acc:  0.88% | Test_loss:  0.3354 | Test_acc:  0.93%

...

Epoch: 780 | Loss:  0.1842 | Acc:  0.92% | Test_loss:  0.1634 | Test_acc:  0.90%

Epoch: 790 | Loss:  0.1834 | Acc:  0.92% | Test_loss:  0.1633 | Test_acc:  0.90%

This indicates that the model achieves a high accuracy on both the training and test sets.

▌Model Visualization

Untrained Model Classification:

The following image shows the decision boundaries of the FFNN before training. Note how the boundaries are random, reflecting the uninitialized weights and biases of the network:

![Untrained model](/images/PCA/2.png)

Trained Model Decision Boundaries:

The following image shows the decision boundaries of the trained FFNN after 800 epochs. The boundaries are now well-defined and reflect the learned relationships between the principal components and the Iris species:

![Trained model](/images/PCA/3.png)

Explanation: These decision boundaries demonstrate how the model separates the different Iris species in the PC1-PC2 space. The trained model exhibits clear separation, indicating good performance. The initial, untrained model shows the importance of training.

▌Conclusion

This project demonstrates a successful application of PCA for dimensionality reduction in conjunction with a Feedforward Neural Network for Iris classification. The model achieves high accuracy and provides a clear visualization of the learned decision boundaries.

## Iris Classification with FFNN and all features

▌Overview

The experiment involved training a simple FFNN directly on the original features, without any dimensionality reduction techniques such as PCA.

▌Data Preprocessing

The Iris dataset was preprocessed by converting the data into tensors. No further feature engineering or scaling was applied.

▌Model Architecture and Training

The classification model is a Feedforward Neural Network (FFNN) with one hidden layer. The architecture is as follows:

*  Input layer: 4 neurons (corresponding to the four features in the Iris dataset: sepal length, sepal width, petal length, and petal width)
*  Hidden layer: 128 neurons, ReLU activation
*  Output layer: 3 neurons (corresponding to the 3 Iris species), Softmax activation

Model Performance:

The model was trained for 800 epochs. The following results were achieved:

Epoch: 0 | Loss:  1.1465 | Acc:  0.00% | Test_loss:  1.1170 | Test_acc:  0.03%

Epoch: 10 | Loss:  1.0188 | Acc:  0.33% | Test_loss:  1.0057 | Test_acc:  0.37%

Epoch: 20 | Loss:  0.9412 | Acc:  0.65% | Test_loss:  0.9280 | Test_acc:  0.70%

Epoch: 30 | Loss:  0.8662 | Acc:  0.68% | Test_loss:  0.8532 | Test_acc:  0.73%

Epoch: 40 | Loss:  0.7897 | Acc:  0.78% | Test_loss:  0.7779 | Test_acc:  0.77%

Epoch: 50 | Loss:  0.7237 | Acc:  0.80% | Test_loss:  0.7125 | Test_acc:  0.80%

Epoch: 60 | Loss:  0.6658 | Acc:  0.87% | Test_loss:  0.6555 | Test_acc:  0.83%

Epoch: 70 | Loss:  0.6157 | Acc:  0.87% | Test_loss:  0.6065 | Test_acc:  0.87%

Epoch: 80 | Loss:  0.5734 | Acc:  0.92% | Test_loss:  0.5652 | Test_acc:  0.87%

Epoch: 90 | Loss:  0.5377 | Acc:  0.92% | Test_loss:  0.5306 | Test_acc:  0.93%

Epoch: 100 | Loss:  0.5075 | Acc:  0.93% | Test_loss:  0.5015 | Test_acc:  0.93%

Epoch: 110 | Loss:  0.4817 | Acc:  0.94% | Test_loss:  0.4768 | Test_acc:  0.93%

Epoch: 120 | Loss:  0.4593 | Acc:  0.94% | Test_loss:  0.4552 | Test_acc:  0.97%

Epoch: 130 | Loss:  0.4393 | Acc:  0.95% | Test_loss:  0.4362 | Test_acc:  0.97%

Epoch: 140 | Loss:  0.4215 | Acc:  0.95% | Test_loss:  0.4193 | Test_acc:  0.97%

Epoch: 150 | Loss:  0.4051 | Acc:  0.96% | Test_loss:  0.4039 | Test_acc:  0.97%

...

Epoch: 780 | Loss:  0.1023 | Acc:  0.98% | Test_loss:  0.1328 | Test_acc:  1.00%

Epoch: 790 | Loss:  0.1015 | Acc:  0.98% | Test_loss:  0.1320 | Test_acc:  1.00%

The results indicate that the FFNN model achieves high accuracy on both the training and test sets. Notably, the model reaches >97% accuracy by epoch 200, and eventually achieves 100% accuracy on the test set.

▌Conclusion

In comparison to the PCA experiment (90% accuracy), this approach yields a superior model. This suggests that, for the Iris dataset, training the FFNN directly on the original features is more effective than using PCA for dimensionality reduction *before* training the FFNN. Possible reasons could be that PCA may discard information relevant for classification, or that the FFNN is capable of learning the relevant features directly from the original data.
