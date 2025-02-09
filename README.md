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

Epoch: 790 | Loss: 0.1894 | Acc: 0.93% | Test_loss: 0.1749 | Test_acc: 0.90%

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
