# Disaster Response Pipeline Project

## Summary
For this project, the dataset contains real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

### Task
The final delivery of this project is to show a web app where an emergency worker can input a new message and get classification results in several categories.

### Process
This project contains three parts. First write a data cleaning pipeline to extract, transform, and load dataset. 
Then, write a machine learning pipeline to train and tune the classification model. Finally, deploy a web application.

## Files Explanation
- app \
| - template \
| |- master.html  # main page of web app \
| |- go.html  # classification result page of web app  \
|- run.py  # Flask file that runs app 

- data \
|- disaster_categories.csv  # data to process \
|- disaster_messages.csv  # data to process \
|- process_data.py \
|- InsertDatabaseName.db   # database to save clean data to 

- models \
|- train_classifier.py \
|- classifier.pkl  # saved model 

- README.md 


1. Flask Web App (app/run.py)

2. ETL Pipeline (data/process_data.py)
In a Python script, process_data.py, write a data cleaning pipeline that:

Loads the messages and categories datasets (data/disaster_messages.csv & data/disaster_categories.csv)
Merges the two datasets
Cleans the data
Stores it in a SQLite database (data/DisasterResponse.db)

3. ML Pipeline (models/train_classifier.py)
In a Python script, train_classifier.py, write a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file (models/classifier.pkl)

## Instructions for runing the code:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
(Then you need to find the workspace environmental variables with `env | grep WORK`, and you can open a new browser window and go to the address:
`http://WORKSPACESPACEID-3001.WORKSPACEDOMAIN` replacing WORKSPACEID and WORKSPACEDOMAIN with your values.)
(for example: https://view6914b2f4-3001.udacity-student-workspaces.com/)
