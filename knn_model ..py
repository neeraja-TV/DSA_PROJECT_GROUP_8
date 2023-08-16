import pandas as pd
import numpy as np
import pickle

df=pd.read_csv('sampled_dataset.csv')
dataset=df.drop(columns=['Unnamed: 0'])

# Initialize the SimpleImputer with 'mean' strategy
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
# Fit and transform the 'heart_rate' column to fill missing values
dataset['heart_rate'] = imputer.fit_transform(dataset[['heart_rate']])

# Define a function to detect outliers using Z-score
def find_outliers_zscore(dataset, threshold=3):
    z_scores = np.abs((dataset - dataset.mean()) / dataset.std())
    return z_scores > threshold
outliers_all_columns = dataset.drop(columns=['activityID', 'PeopleId']).apply(find_outliers_zscore)
outliers_rows = outliers_all_columns.any(axis=1)
outliers_detected = dataset[outliers_rows]
potential_outliers_index = outliers_detected.index
df_cleaned = dataset.drop(index=potential_outliers_index)
df_cleaned.reset_index(drop=True, inplace=True)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_cleaned['activityID'] = le.fit_transform(df_cleaned['activityID'])

x = df_cleaned.drop('activityID',axis=1)
y = df_cleaned['activityID']
# Create the MinMaxScaler object
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
x_scaled=pd.DataFrame(x_scaled,columns=x.columns)

from sklearn.decomposition import PCA
pca=PCA(n_components=0.97)
pca.fit(x_scaled)
x_pca=pca.transform(x_scaled)
x_pca=pd.DataFrame(x_pca)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)



from sklearn.neighbors import KNeighborsClassifier

# Lets create knn model for 3 neighbors
knn_model=KNeighborsClassifier(n_neighbors=3,metric='euclidean')
final_model=knn_model.fit(x_train,y_train)
pickle.dump(final_model,open('knn_model.pkl','wb'))


  
