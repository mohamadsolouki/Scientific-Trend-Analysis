import pandas as pd

def concatenate_columns(input_file, output_file):
    """Concatenate 'title' and 'abstract' columns and save the result."""
    column_names = ['id', 'title', 'abstract', 'categories', 'update_date']
    df = pd.read_csv(input_file, names=column_names)

    # Concatenate 'title' and 'abstract' columns
    df['concat'] = df['title'] + ' ' + df['abstract']
    df['concat'] = df['concat'].fillna('')

    # delete the first row
    df = df.drop(0)

    #drop Id, the 'title' and 'abstract' columns
    df = df.drop(['id', 'title', 'abstract'], axis=1)

    # Save the concatenated data to the output file
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    input_file = 'data_preprocessed.csv'
    output_file = 'data_concatenated.csv'
    concatenate_columns(input_file, output_file)
