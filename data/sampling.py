import pandas as pd

# Specify the file paths
input_file = 'data_preprocessed.csv'
output_file = 'data_preprocessed-sample.csv'

# Read the first 100 rows of the original dataset
df = pd.read_csv(input_file, nrows=100)

# Write the sample dataset to a new file
df.to_csv(output_file, index=False)

print(f'Sample processed data saved to {output_file}')

# Specify the file paths
input_file = 'data_clustered.csv'
output_file = 'data_clustered_sample.csv'

# Read the first 100 rows of the original dataset
df = pd.read_csv(input_file, nrows=100)

# Write the sample dataset to a new file
df.to_csv(output_file, index=False)

print(f'Sample clustered data saved to {output_file}')

# Specify the file paths
input_file = 'data_concatenated.csv'
output_file = 'data_concatenated_sample.csv'

# Read the first 100 rows of the original dataset
df = pd.read_csv(input_file, nrows=100)

# Write the sample dataset to a new file
df.to_csv(output_file, index=False)

print(f'Sample concatenated data saved to {output_file}')