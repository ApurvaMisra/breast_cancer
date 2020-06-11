import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from collections import defaultdict
from sklearn.model_selection import train_test_split
#Accessing the dataset
df=pd.read_csv("/home/apurva/Documents/code/breast_cancer/breast-cancer-wisconsin.data",sep=",",names=['sample_code_number','clump_thickness','uniformity_of_cell_size','uniformity_of_cell_shape','marginal_adhesion','single_epithilial_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses','target'])
#Printing the total number of benign and malignant cases, here '2' means benign and '4' means malignant
print(df.target.value_counts())
#Replacing "?" with NaN values
df.replace("?", np.nan, inplace=True)
#Dropping rows containing NaN values
df.dropna(axis=0,how='any',inplace=True)
#Dropped "sample_code_number" column as its not a 
#feature we would want to consider for making the model 
df.drop(columns=['sample_code_number'],inplace=True)
#Converting the "bare_nuclei" feature to numeric datatype
df.bare_nuclei=pd.to_numeric(df.bare_nuclei)
col_feat=['clump_thickness','uniformity_of_cell_size','uniformity_of_cell_shape','marginal_adhesion','single_epithilial_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses']
#Initialising the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
#Splitting the dataset into train and test set, here test set is 30\% of the dataset
X_train, X_test, y_train, y_test = train_test_split(df.loc[:,df.columns!="target" ], df.target, shuffle=True, test_size=0.30, random_state=42)
clf.fit(X_train, y_train)
#Accuracy with all features included
print("Accuracy on test data with all features included: {:.2f}".format(clf.score(X_test, y_test)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
#Finding the correlation between features
corr = spearmanr(df.loc[:,df.columns!="target" ]).correlation
#Forming a heirarchy for the correlations between features
corr_linkage = hierarchy.ward(corr)
dendro = hierarchy.dendrogram(corr_linkage, labels=df.loc[:,df.columns!="target" ].columns, ax=ax1,
                            leaf_rotation=90)
dendro_idx = np.arange(0, len(dendro['ivl']))
ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
ax2.set_yticklabels(dendro['ivl'])
#Including a red horizontal line to use as a threshold
ax1.axhline(y=0.6, color='r', linestyle='-')
fig.tight_layout()
#plt.show()
plt.savefig("cluster_breast.png")
#Using a threshold to get the specific clusters
cluster_ids = hierarchy.fcluster(corr_linkage,0.6 , criterion='distance')
cluster_id_to_feature_ids = defaultdict(list)
#Going over each of the features in a cluster
for idx, cluster_id in enumerate(cluster_ids):
    #Adding the lost of features for each cluster in a dictionary
    cluster_id_to_feature_ids[cluster_id].append(idx)
    #Selecting the first feature in each cluster's list
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    #printing the selected features
    print([col_feat[x] for x in selected_features])
    #Intitalising and training the RF model on a reduced dataset
    clf_sel = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_sel.fit(X_train.iloc[:,selected_features], y_train)
    print("Accuracy on test data with features removed: {:.2f}".format(
    clf_sel.score(X_test.iloc[:,selected_features], y_test)))


from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=1, step=1)
selector = selector.fit(X_train, y_train)
indices=selector.get_support(indices=True)
print(indices)
print([col_feat[x] for x in indices])
