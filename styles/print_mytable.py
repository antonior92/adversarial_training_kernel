import pandas as pd

data = 'out/error_vs_sample_size.csv'

# Load and pivot using stack/unstack
df = pd.read_csv(data)
df.set_index(['kernel','curve',  'method'], inplace=True)
reshaped = df['rate'].unstack(['curve', 'method'])

# Reset index for display or export
reshaped = reshaped.sort_index(axis=1)


print(reshaped.round(2).to_latex())