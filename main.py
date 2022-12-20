import area as area
import inline as inline
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from numpy import int64, float64
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

seg = pd.read_csv("G:\\classification\\train.csv")
print(seg.isnull().any())

# Encoding categorical data
encoding = LabelEncoder()
seg['Gender'] = encoding.fit_transform(seg['Gender'])

seg['Ever_Married'] = encoding.fit_transform(seg['Ever_Married'])

seg['Graduated'] = encoding.fit_transform(seg['Graduated'])

seg['Profession'] = encoding.fit_transform(seg['Profession'])

seg['Spending_Score'] = encoding.fit_transform(seg['Spending_Score'])

seg['Var_1'] = encoding.fit_transform(seg['Var_1'])

seg['Segmentation'] = encoding.fit_transform(seg['Segmentation'])

# Data cleaning
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
x1 = np.array(seg['Ever_Married'], dtype=int64)
seg['Ever_Married'] = imp.fit_transform(x1.reshape(-1, 1))

x2 = np.array(seg['Graduated'], dtype=int64)
seg['Graduated'] = imp.fit_transform(x2.reshape(-1, 1))

x3 = np.array(seg['Profession'], dtype=int64)
seg['Profession'] = imp.fit_transform(x3.reshape(-1, 1))

x4 = np.array(seg['Work_Experience'], dtype=int64)
seg['Work_Experience'] = imp.fit_transform(x4.reshape(-1, 1))

x5 = np.array(seg['Family_Size'], dtype=int64)
seg['Family_Size'] = imp.fit_transform(x5.reshape(-1, 1))

x6 = np.array(seg['Var_1'], dtype=int64)
seg['Var_1'] = imp.fit_transform(x6.reshape(-1, 1))

# print(seg.nunique())
# print(seg.shape)
# Declaration
print(seg.isnull().any())
X = seg.iloc[:, :10]
Y = seg.iloc[:, -1:]
# print(Y.head())

# feature selection from model
# select2 = SelectFromModel(RandomForestClassifier())
# Selected = select2.fit_transform(X, Y)
# print(Selected.shape)
# print(select2.get_support())

# # preprocessing MinMaxscaler
scale = MinMaxScaler(copy=True, feature_range=(0, 1))
a = np.array(seg['Age'], dtype=int64)
seg['Age'] = scale.fit_transform(a.reshape(-1, 1))

b = np.array(seg['Family_Size'], dtype=int64)
seg['Family_Size'] = scale.fit_transform(b.reshape(-1, 1))

# Splitting data to train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=False, )
# =================================ENSEMLE==========================================
# Gradient Boosting Classifier
model1 = GradientBoostingClassifier(learning_rate=0.04)
# -------------------------------------------------------------
# Random forest classifier
model2 = RandomForestClassifier()

# =================================NORMAL===========================================
# Logistic regression
model3 = LogisticRegression()
# ---------------------------------------------------------------
# KNN
model4 = KNeighborsClassifier()
# ---------------------------------------------------------------
# Decision tree
model5 = DecisionTreeClassifier(max_depth=10)
# ---------------------------------------------------------------
# naive bayes
model6 = GaussianNB()
model7 = BernoulliNB()
# ---------------------------------------------------------------
# LDA
model8 = LinearDiscriminantAnalysis()
# ====================================
# model8.fit(x_train, y_train)
# # sorted(model11.cv_results_.keys())
# y_pred = model8.predict(x_train)
# print('my predictions:', y_pred)
# print('accuracy score is:', accuracy_score(y_train, y_pred))

# # ========================================PLOT============================================
# eclf = VotingClassifier(
#     estimators=[('1', model1), ('2', model2), ('3', model3), ('5', model5), ('6', model6),
#                 ('7', model7), ('8', model8), ('9', model9), ('10', model10)], voting='hard')
# for clf, label in zip([model1, model2, model3, model5, model6, model7, model8, model9, model10, eclf],
#                       ['GradientBoostingClassifier', 'RandomForestClassifier','LogisticRegression',
#                        'KNN', 'DecisionTreeClassifier', 'GaussianNB',
#                        'BernoulliNB', 'LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis']):
#     scores = cross_val_score(clf, X, Y, scoring='accuracy')
#     print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


# decision boundray
from sklearn.datasets import make_classification
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

X, Y = make_classification(n_samples=7165, n_features=2, n_informative=2, n_redundant=0, n_classes=2)
import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(4, 2)

fig = plt.figure(figsize=(20, 10))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

labels = ['GradientBoostingClassifier', 'RandomForestClassifier', 'LogisticRegression', 'KNeighborsClassifier',
          'DecisionTreeClassifier', 'GaussianNB', 'BernoulliNB', 'LinearDiscriminantAnalysis']
for clf, lab, grd in zip([model1, model2, model3, model4, model5, model6, model7, model8],
                         labels,
                         [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]):
    clf.fit(X, Y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=Y, clf=clf, legend=2)
    plt.title(lab)

plt.show()
# confusion matrix
# cm = confusion_matrix(y_train, y_pred)
# sns.heatmap(cm, center=True)
# plt.title('Segment it')
# plt.show()

# def plotGraph(y_train, y_pred, rand):
#     if max(y_train) >= max(y_pred):
#         my_range = int(max(y_train))
#     else:
#         my_range = int(max(y_pred))
#     plt.scatter(range(len(y_train)), y_train, color='blue')
#     plt.scatter(range(len(y_pred)), y_pred, color='red')
#     plt.title(rand)
#     plt.show()
#     return
#
#
# plotGraph(y_train, y_pred, 'Decision tree')

# ============================================= GET TEST ==============================================
pre = pd.read_csv('G:\\classification\\test.csv')

# Encoding categorical data
encoding = LabelEncoder()
pre['Gender'] = encoding.fit_transform(pre['Gender'])

pre['Ever_Married'] = encoding.fit_transform(pre['Ever_Married'])

pre['Graduated'] = encoding.fit_transform(pre['Graduated'])

pre['Profession'] = encoding.fit_transform(pre['Profession'])

pre['Spending_Score'] = encoding.fit_transform(pre['Spending_Score'])

pre['Var_1'] = encoding.fit_transform(pre['Var_1'])

# Data cleaning
x11 = np.array(pre['Work_Experience'], dtype=int64)
pre['Work_Experience'] = imp.fit_transform(x11.reshape(-1, 1))

x22 = np.array(pre['Family_Size'], dtype=int64)
pre['Family_Size'] = imp.fit_transform(x22.reshape(-1, 1))

# preprocessing MinMaxscaler
# aa = np.array(pre['Age'], dtype=int64)
# pre['Age'] = scale.fit_transform(aa.reshape(-1, 1))
#
# bb = np.array(pre['Family_Size'], dtype=int64)
# pre['Family_Size'] = scale.fit_transform(bb.reshape(-1, 1))

# Check on the nan cells
# print(pre.columns[pre.isnull().any()].tolist())
# print(pre.isnull().any())
# # printing
# id = pre['ID'].values
# X2 = pre.iloc[:, :]
# y_pred2 = model1.predict(X2)
# final_frame = pd.DataFrame({'ID': id, 'Segmentation': y_pred2})
# print(final_frame)
# final_frame.to_csv('G:\\classifications.csv')
# print('pass')
