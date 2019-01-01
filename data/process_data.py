import sys
import pandas as pd

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Input:
        messages_filepath: str, filepath to .csv file with message text
        categories_filepath: str, filepath to .csv file with information about message category
    Output:
        pandas DataFrame
    """

    # load message and category datasets into pandas DataFrames
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)

    # merge both datasets into single pandas DataFrame
    df = pd.merge(df_messages, df_categories, how="inner", on='id', left_index=True, right_index=True)

    return df


def clean_data(df):
    """
    Input:
        df: pandas DataFrame, columns [id, message in english,
                                       original message text, genre, message category]
    Output:
        return DataFrame without duplicates
    """

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)

    # extract a list of new column names using the first row of the categories DataFrame
    category_colnames = categories.iloc[0].apply(lambda x: x[:-2]).values.tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1.
    for column in categories:
        # 1. split string:
        #   related-1 becomes ["related", "1"]
        # 2. convert second list element to integer if it is equal to 1, else 0
        #   ["1"] becomes 1
        categories[column] = categories[column].str.split("-").apply(lambda x: int(x[1]) if x[1] == "1" else 0)

    # drop the original categories column from df
    df = df.drop(['categories'], axis=1)

    # concatenate df DataFrame with the new categories DataFrame
    df = pd.concat([df, categories], axis=1)

    # return DataFrame without duplicates
    return df.drop_duplicates()


def save_data(df, database_filename):
    """
    saves pandas DataFrame to SQL .db file

    Input:
        df: pandas DataFrame,
        database_filename: str
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql("messages", engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
