import sys
import pandas as pd
from sqlalchemy import create_engine
import os

# test it
def load_data(messages_filepath, categories_filepath):
    '''
    Loads the messages and categories datasets
    Merges the two datasets
    '''
    # load database
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = pd.merge(messages, categories, on='id', how='inner')

    return df
    
def clean_data(df):
    '''
    Cleans the data:
    1. Split categories into separate category columns
    2. Convert category values to just numbers 0 or 1
    3. Drop duplicates
    '''
    
    #  Split categories into separate category columns.
    categories = df['categories'].str.split(';',expand=True)  # create a dataframe of the 36 individual category columns
    row = categories.loc[0,:]   # select the first row of the categories dataframe
    category_colnames = [(lambda i: ''.join(x for x in i if x.isalpha() or x == '_'))(i) for i in row]  # extract a list of new column names for categories
    categories.columns = category_colnames   # rename the columns of `categories`
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = [i[-1] for i in categories[column]] #     # set each value to be the last character of the string       
        categories[column] = [int(i) for i in categories[column]]     # convert column from string to numeric
        
    df.drop('categories', axis = 1, inplace = True) # drop the original categories column from `df` 
    
    # Concatenate the original dataframe with the new 'categories' dataframe
    df = pd.concat([df, categories], axis = 1)

    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    '''
    Stores it in a SQLite database (data/DisasterResponse.db)
    '''
    
    # remove the DisasterResonse.db
    os.remove('data/DisasterResponse.db')
    
    # create a sql db; save the clean database into an sqlite database
    engine = create_engine('sqlite:////home/workspace/Disaster_Pipeline_Project/'+database_filename)
    print(engine)
    # convert df framework into the sql
    df.to_sql('DisasterResponse', engine, index=False)
    
    # read sql
#     pd.read_sql('SELECT * FROM DisasterResponse', engine)
    
    print(pd.read_sql('SELECT * FROM DisasterResponse', engine).tail())
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        print(messages_filepath)
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
