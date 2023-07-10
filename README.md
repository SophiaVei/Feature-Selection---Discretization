# Feature Selection - Discretization

Using the HTRU2 dataset, and combined with a machine learning algorithm (which will predict whether an instance of the dataset is a pulsar star
or not), 2 Feature Importance methods are applied.  
The code displays:  
• the performance of the corresponding algorithm (Accuracy, Precision, Recall, F1, graph
AUC) before and after using the PCA method for n=4  
• a list of the most important features based on the model, in descending order
(ie from most important to least important)  
• the performance of the corresponding algorithm which will use only the 4 most
important features found previously. Then we compare them with the performances found in the first case
