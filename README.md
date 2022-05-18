![badge](http://ForTheBadge.com/images/badges/made-with-python.svg) ![badge](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

# customer_segmentation
 This program is used to develop a deep learning model with 2 hidden layers to segment customers into one of four categories for automobile purchases based on their profile.

# How to use
Clone the repository and use the following scripts per your use case:
1. train.py is the script that is used to train the model.
2. deploy.py is the script that is used to deploy the model, and appends the predictions made directly into the .csv dataset.
3. segmentation_modules.py is the class file that contains the defined functions used during training and evaluation for added robustness and reusability of the processes used.
4. The saved model and encoders are available in .h5 and .pkl format respectively in the 'saved_model' folder.
5. The original dataset (train.csv) and prediction dataset (new_customers.csv) are available in the 'dataset' folder.
6. Screenshots of the model architecture, feature/target correlation, train/test results, and TensorBoard view are available in the 'results' folder.

# Results
The model developed using 2 hidden layers with ReLu activation was scored using accuracy and f1-score, attaining 52.29% accuracy on the test dataset.

Model architecture:
![model](https://github.com/khaiyuann/customer_segmentation/blob/main/results/model.png)

Feature/target correlation:
![correlation](https://github.com/khaiyuann/customer_segmentation/blob/main/results/correlation.png)

Train/test results (achieved 52.29% accuracy and f1 score):
![train_test_results](https://github.com/khaiyuann/customer_segmentation/blob/main/results/train_test_score.png)

TensorBoard view:
![tensorboard](https://github.com/khaiyuann/customer_segmentation/blob/main/results/tensorboard.png)

# Discussion of results
From the results obtained of 52.29%, it falls short of the targeted 80%.
There are various limitations that contribute to the result deficiencies:
    
1. The dataset consists of many categorical features that have NaN data, and to perform precise imputation on the dataset will be costly in both time and computational power, hence only a simple imputer was used to complete the model within the available time and resources.

2. Customer segmentation is a multiclass classification problem that is highly complex and difficult to achieve high accuracy on, especially given point (1) where the data provided consists of many categories each with multiple items and missing data. Simply dropping the data is not an option due to the varied nature of the data, however imputation as a solution is not optimal as it is unable to replicate actual data. Hence, the lower accuracy obtained can be attributed to the nature of the dataset and is limited by such.
        
3. The correlation indices obtained from the correlation heatmap have very low correlation, with the higest of 0.24 for the 'Age' column. The overall low correlation between features and the target heavily limits the potential of the DL model to accurately predict the segment of a customer. If visualized, it would show clusters of the customer segment in very close proximity and without clear borders, as the features used in training does not strongly correlate to a segment. Hence, it resulted in a high quantity of wrong predictions.
        
Future improvements to the accuracy of the model can be achieved by employing more sophisticated techniques and DL architectures, or machine learning approaches may be more effective for this particular problem. These proposed solutions are to be considered on future work concerning this customer segmentation dataset.

# Credits
Thanks to Abishek Sudarshan (Kaggle: abisheksudarshan) for providing the Customer Segmentation Dataset used for the training of the model on Kaggle. 
Check it out here for detailed information: https://www.kaggle.com/datasets/abisheksudarshan/customer-segmentation