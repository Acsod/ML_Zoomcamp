# Drug classification

**Description**

I work in the healthcare field. This is an example of a model for identifying a drug from patient tests.
We did it with dataset from here: https://www.kaggle.com/datasets/prathamtripathi/drug-classification/code

**How to use**

You can see in notebook.ipynb:

1. Data preparation and data cleaning
2. EDA, feature importance analysis
3. Model selection process and parameter tuning

You can see train and model saving in train.py

After then, you can run docker and test predict with test_prediction.ipynb:

1. Download folder with project from git.
2. Run docker and go to folder in cmd or powershell
3. Enter the command: docker build -t drug_predict .
4. Enter the command: docker run -it -p 9696:9696 drug_predict:latest
5. Execute test_prediction.ipynb.

