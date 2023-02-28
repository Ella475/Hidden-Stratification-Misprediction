import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the data
data = pd.read_csv('german_credit_data.csv')

# Drop irrelevant columns
data = data.drop(['Unnamed: 0', 'Purpose'], axis=1)

# Encode categorical features
cat_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Job']
data = pd.get_dummies(data, columns=cat_cols)

# Scale the numerical features
num_cols = ['Age', 'Credit amount', 'Duration']
scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

# Convert the labels to binary values
data['Risk'] = data['Risk'].apply(lambda x: 1 if x == 'good' else 0)

# Split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Separate the features and labels
train_features = train_data.drop(['Risk'], axis=1).values
train_labels = train_data['Risk'].values
test_features = test_data.drop(['Risk'], axis=1).values
test_labels = test_data['Risk'].values
