# AIAP Batch 15 Technical Assessment  
Full Name: Lim Jun Ming  
Email Address: billy.ljm@gmail.com  
  
## Folder Structure  
- `src/`: Folder of the machine learning pipeline  
    - `datapipeline.py`: Datapipeline class to read and clean the datasets, and separate into features and labels  
    - `model_dummy.py`: Dummy classifier to set the baseline  
    - `model_svm.py`: Linear support vector machine model  
    - `model_forest.py`: Random forest model  
    - `model_boost.py`: Gradient-boosted tree model  
- `eda.ipynb`: IPython notebook of exploratory data exploration process  
- `requirements.txt`: Python requirement file  
- `run.sh`: Bash script that trains all the ML models and tests them  
  
## Demo  
The bash script is meant as a demo and test of all the ML models used in this repo.  
An example of its output is given below.  
  
```  
$ bash run.sh  
-------  
 Dummy  
-------
Hyperparameters used are {}
The test F1-macro score is 0.21581929516985543
-----
 SVM
-----
Hyperparameters used are {'C': 0.03125}
The test F1-macro score is 0.4957812932938626
---------------
 Random Forest
---------------
Hyperparameters used are {'criterion': 'gini', 'max_depth': 80, 'max_features': 'sqrt', 'n_estimators': 10}
The test F1-macro score is 0.5420678411521804
----------------
 Gradient Boost
----------------
Hyperparameters used are {'learning_rate': 0.1, 'max_depth': 9, 'max_iter': 500}
The test F1-macro score is 0.5482298987666266
```  
  
## Pipeline Instructions  
All of the models in `model_xxx.py` has the same methods.  
- `Model(pre_path, post_path)` instantiates the model, providing the filepath to the dataset that the model will be trained and tested on.  
- `Model.train()` trains the model on the training dataset, using grid search and cross-validated F1-macro score to optimise the hyperparameters.  
- `Model.test()` tests the trained model on the test dataset, returning the associated F1-macro score.  
- `Model.predict(features)` applies the model to a new set of features, and predicts the labels for them.  
  
The parameters (of the grid search, etc) can be modified directly in the corresponding `model_xxx.py` file.  
- `self.preprocess` is a pipeline that imputes nulls values, encodes variables, manually selects features, and applies any other pre-processing steps like PCA to the features.  
- `clf` contains the grid search and the parameter space it will optimise over.  
  
## Pipeline Flow  
The flow of the pipelines in each `model_xxx.py` are very similar.  
1. The datasets are read in, cleaned, and separated using `Datapipeline()`.  
2. The multi-class labels are encoded with sklearn's `LabelEncoder()`  
   The features are preprocessed via the pipeline in `model.preprocess`  
3. Calling `model.train()` trains the model, using a grid search to optimise hyperparameters.  
4. Calling `model.test()` returns the F1-macro score of the trained model on the test dataset  
5. Calling `model.predict(features)` the applies the trained model onto new data beyond the train/test dataset.  
  
## EDA Findings  
The main EDA findings are:  
1. The labels are very imbalanced with Luxury, Deluxe, and Standard making up 48%, 7%, and 45% of the non-null labels respectively.  
   Thus, we have to stratify when splitting, and weight each class appropriately when training and scoring.  
2. The post-trip survey data, namely `WiFi`, `Entertainment`, and `Dining`, is very untrustworthy.  
   Since the first 2 have 50% null values while the last has 0% null values.  
   And they are uncorrelated even with their directly-related pre-purchase ratings on the importance of `Onboard Wifi Service`, `Onboard Entertainment`, and `Onboard Dining Service` respectively.  
3. There are strong correlations within subsets of the 13 pre-purchase ratings of `Onboard Wifi Service`, `Embarkation/Disembarkation time convenient`, `Ease of Online booking`, `Gate location`, `Onboard Dining Service`,  `Online Check-in`, `Cabin Comfort`, `Onboard Entertainment`, `Cabin service`, `Baggage handling`, `Port Check-in Service`, `Onboard Service`, `Cleanliness`.  
   These features might be suitable for aggregating via principal component analysis.  
4. `Logging` and `Cruise Distance` are not normally-distributed  
5. `Logging` is not correlated with any other features, as expected from its description.  
6. The null values in the dataset is not correlated, are seem to be mostly sparsely spread out across different columns for different rows  
7. `Gender`, `Source of Traffic`, and `Cruise Name` are nominal categorical variables and have to be encoded appropriately.  
  
Based on these, the features we will correspondingly engineer are:  
1. stratify when splitting to ensure all labels are included in every dataset we use to train, test, and cross-validate.  
   And we will use F1-macro score when optimising hyperparameters, which is more suitable for the imbalance and gives equal weights to all 3 labels.  
2. drop `WiFi`, `Entertainment`, and `Dining`, to avoid feeding bad data into our ML model.  
3. principal component analysis of the 13 pre-purchase ratings, to aggregate some of the highly correlated ones together and reduce the dimensionality of the dataset  
4. used rank correlations in the EDA to sidestep the non-normality  
5. drop `Logging`, to reduce dimensionality of the dataset  
6. We will use a simple imputer to impute null values with the mean of continuous/ordinal variables, and the most frequency category for nominal variables.  
7. one-hot encode `Gender`, `Source of Traffic`, and `Cruise Name`  
  
  
## Data Pre-Processing  
The feature engineering steps listed above will be implemented as a sklearn datapipeline in each of the `src/model_xxx.py`.  
The action of this preprocessing pipeline on each column is illustrated below.  
  
<table>  
	<tr>  
		<td>Ticket Type</td>  
		<td>Cruise Distance</td>  
		<td>Gender</td>  
		<td>Cruise Name</td>  
		<td>Source of Traffic</td>  
		<td>Date of Birth</td>  
		<td>Onboard Wifi Service</td>  
		<td>Embarkation/Disembarkation time convenient</td>  
		<td>Ease of Online booking</td>  
		<td>Gate location</td>  
		<td>Onboard Dining Service`,  `Online Check-in</td>  
		<td>Cabin Comfort</td>  
		<td>Onboard Entertainment</td>  
		<td>Cabin service</td>  
		<td>Baggage handling</td>  
		<td>Port Check-in Service</td>  
		<td>Onboard Service</td>  
		<td>Cleanliness</td>  
		<td>Logging</td>  
		<td>WiFi</td>  
		<td>Dining</td>  
		<td>Entertainment</td>  
	</tr>  
	<tr>  
		<td rowspan=3>Label Encoder</td>  
		<td colspan=4>-</td>  
		<td>Year()</td>  
		<td colspan=12>-</td>  
		<td colspan=4 rowspan=3>Dropped</td>  
	</tr>  
	<tr>  
		<td colspan=4>Impute Most Frequent Category</td>  
		<td colspan=13>Impute Mean</td>  
	</tr>  
	<tr>  
		<td colspan=4>One-Hot Encode</td>  
		<td colspan=13>Principal Component Analysis</td>  
	</tr>  
</table>  
  
This is a common pipeline that will was ultimately used across all models in this repo.  
However, it is defined separately in each `model_xxx.py` to give the option for different models to have different pre-processing steps.  
  
## Model Choice  
The 3 ML models + 1 dummy model and that we have chosen to train are  
- **Dummy Classifier**, to give a baseline for a model which just predicts the most frequent class.  
- **Support Vector Machine**, which represents a simple model that only have 1 hyperparameter to choose.  
- **Random Forests**, as a more complicated ensemble model with bagging to minimise variance  
- **Gradient Boosted Trees**, as another ensemble model but with boosting to minimise bias.  
  
## Evaluation Metrics  
For our model training, we have chosen to use the macro-averaged F1 score.  
Since I assumed we are still concerned with correctly predicting Deluxe tickets, despite their low  proportion in the population.  
Thus, we have chosen to use F1 score which is suitable for imbalanced data, and macro-averaging to give equal weight to all ticket types.  
  
This measure was used to optimise the hyperparameters of all the models using a grid search.  
And the maximum score for each of the optimised models is  
  
|          Model | F1-macro |  
|----------------|----------|  
|          Dummy |    0.226 |  
|            SVM |    0.496 |  
|  Random Forest |    0.543 |  
| Gradient Boost |    0.546 |  
  
Thus, the gradient boost tree yields the best F1-macro score, and will be the one I'll propose to use in production.  
  
## Other Considerations  
- If the models will be deployed on more powerful compute clusters, we can consider doing away with some of the manual feature selection done here, like the principal component analysis.  
  That would ensure the ML model have access to everything in the original dataset, at the expense of some longer training times.  
- We can consider weighting out F1 score differently, for example by weighing each ticket type by their ticket price to encapsulate their monetary importance.  