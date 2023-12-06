#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import pickle
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso, Ridge

#Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#Loading the Dataframe
df = pd.read_csv("bodyfat.csv")
print(df.head())

#Checking Missing Values
print(df.isna().sum())

print(df.info())


#Data Visualization
def plotdistplots(col):
    plt.figure(figsize=(12, 5))
    sns.distplot(df["BodyFat"], color="magenta", hist=False, label="BodyFat")
    sns.distplot(df[col], color="red", hist=False, label=col)
    plt.legend()
    plt.show()
    
cols = list(df.columns)
for i in cols:
    print(f"Distribution plots for {i} feature is shown below ")
    plotdistplots(i)
    print("_"*75)


def drawplots(df, col):
    
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 3, 1)
    plt.hist(df[col], bins=20, color="brown")
    
    plt.subplot(1, 3, 2)
    stats.probplot(df[col], dist="norm", plot=plt)
    
    plt.subplot(1, 3, 3)
    sns.boxplot(df[col], color="purple")
    
    plt.show()
    

cols = list(df.columns)

for i in cols:
    print(f"Distribution plots for the feature {i} are shown below ")
    drawplots(df, i)
    print("_"*75)


#Checking Outlier
upperlimit = []
lowerlimit = []

for i in df.columns:
    upperlimit.append(df[i].mean()+(df[i].std())*4)
    lowerlimit.append(df[i].mean()-(df[i].std())*4)

cols = list(df.columns)
j = 0

for i in cols:
    temp = df.loc[(df[i]>upperlimit[j])&(df[i]<lowerlimit[j])]
    j += 1

print(temp)




#Using ExtraTrees Regressor for Feature Selection
data = df.copy()
train = data.drop(columns="BodyFat")
test = data["BodyFat"]


er = ExtraTreesRegressor()
er.fit(train, test)

series = pd.Series(er.feature_importances_, index=train.columns)
series.nlargest(5).plot(kind="barh", color="green")
plt.show()



#Using Mutual Information Gain for Feature Selection
mr = mutual_info_regression(train, test)

plotdata = pd.Series(mr, index=train.columns)
plotdata.nlargest(5).plot(kind="barh", color="green")
plt.show()



#Removing correlation
print(data)

plt.figure(figsize=(15, 7))
sns.heatmap(df.corr(), annot=True, cmap="plasma")
plt.show()



def correlation(df, threshold):
    colcor = set()
    cormat = df.corr()
    for i in range(len(cormat)):
        for j in range(i):
            """
            for each cell get the value of that cell by .iloc[i][j],
            where i is the row and j is the col if that abs(value) is greater
            than the threshold, get the col_name and add it in the set
            """
            if abs(cormat.iloc[i][j])>threshold:
                colname=cormat.columns[i]
                colcor.add(colname)
                
    return colcor

ans = correlation(train, threshold=0.85)
print(ans)

"""From the above feature selection techniques we can say that the features recommended by the Extra Trees Regressor and the mutual_information_gain are correct and 
from the correlation map we get to observe the similar pattern we noticed that abdomen and Hip are having similar features, they're having collinearity, same gors with 
knee and thigh, we can either keep any one of them and we noticed that the feature Abdomen gave more feature importance score in comparision to Hip, so I will be 
selecting that"""

temp = data[list(data.columns)]
info = pd.DataFrame()
info["VIF"] = [variance_inflation_factor(temp.values, i)  for i in range(temp.shape[1])]
info["Column"] = temp.columns
print(info)


#Selecting the features
cols1 = list(series.nlargest(5).index)
cols2 = list(plotdata.nlargest(5).index)
print(cols1)
print(cols2)

"""We'll go with the weight and Hip method, as Hip and Thigh are very much related, so we'll select the cols1 features and drop every other feature, if that doesn't 
produce any further importance we will try with some other feature"""

totrain = train[cols1]
print(totrain.head())



#Data splitting
X_train, X_test, y_train, y_test = train_test_split(totrain, test, test_size=0.2)

print(X_train.shape)
print(X_test.shape)


reg =DecisionTreeRegressor()
reg.fit(X_train, y_train)
plt.figure(figsize=(15, 7))
tree.plot_tree(reg, filled=True)
plt.show()

#Prunning
path = reg.cost_complexity_pruning_path(X_train, y_train)
ccp_alpha = path.ccp_alphas

alphalist = []
for i in range(len(ccp_alpha)):
    reg = DecisionTreeRegressor(ccp_alpha=ccp_alpha[i])
    reg.fit(X_train, y_train)
    alphalist.append(reg)


trainscore = [alphalist[i].score(X_train, y_train)  for i in range(len(alphalist))]
testscore = [alphalist[i].score(X_test, y_test)  for i in range(len(alphalist))]

plt.xlabel("alpha")
plt.ylabel("accuracy")
plt.plot(ccp_alpha, trainscore, marker="o", label="training", color="black", drawstyle="steps-post")
plt.plot(ccp_alpha, testscore, marker="+", label="testing", color="red", drawstyle="steps-post")
plt.legend()
plt.show()




#Normal Approach
clf = DecisionTreeRegressor(ccp_alpha=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Decision Tree Normal Approach : {metrics.r2_score(y_test, y_pred)}")

rf = RandomForestRegressor(n_estimators=1000, ccp_alpha=1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"Random Forest Normal Approach : {metrics.r2_score(y_test, y_pred_rf)}")




#Hyper parameter Tuning
params = {
    "RandomForest":{
        "model":RandomForestRegressor(),
        "params":{
            "n_estimators": [int(x)  for x in np.linspace(start=1, stop=1200, num=10)],
            "criterion" : ["squared_error", "absolute_error"],
            "max_depth": [int(x)  for x in np.linspace(start=1, stop=30, num=5)],
            "min_samples_split": [2, 5, 10, 12],
            "min_samples_leaf": [2, 5, 10, 12],
            "max_features": [1.0, "sqrt"],
            "ccp_alpha": [1, 2, 2.5, 3, 3.5, 4, 5]
        }
    },
    
    "D-tree":{
        "model":DecisionTreeRegressor(),
        "params":{
            "criterion": ["squared_error", "absolute_error"],
            "splitter": ["best", "random"],
            "min_samples_split": [1, 2, 5, 10, 12],
            "min_samples_leaf": [1, 2, 5, 10, 12],
            "max_features": [1.0, "sqrt"],
            "ccp_alpha": [1, 2, 2.5, 3, 3.5, 4, 5]
        }
    },
    
    "SVM":{
        "model":SVR(),
        "params":{
            "C": [0.25, 0.50, 0.75, 1.0],
            "tol": [1e-10, 1e-5, 1e-4, 0.025, 0.50, 0.75],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "max_iter": [int(x)  for x in np.linspace(start=1, stop=250, num=10)]
        }
    }
}


scores = []
for modelname, mp in params.items():
    try:
        clf = RandomizedSearchCV(mp["model"], param_distributions=mp["params"], cv=5, n_jobs=-1, n_iter=10, scoring="neg_mean_squared_error")
        clf.fit(X_train, y_train)
        scores.append({
            "model_name": modelname,
            "best_score": clf.best_score_,
            "best_estimator": clf.best_estimator_
        })
    except Exception as e:
        scores.append({
            "model_name": modelname,
            "error": str(e)
        })

print(scores)


scoresdf = pd.DataFrame(scores, columns=["model_name", "best_score", "best_estimator"])
print(scoresdf)
print(scores[0]["best_estimator"])




#Model
model = scores[0]["best_estimator"]
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(metrics.r2_score(y_test, y_pred))


totrainlist = np.array(totrain)
predicted = []
for i in range(len(totrainlist)):
    predicted.append(model.predict([totrainlist[i]]))

totrain["Actual Result"] = test
totrain["Predicted Result"] = np.array(predicted)
print(totrain)


sns.distplot(totrain["Actual Result"], hist=False, color="black", label="Actual Result")
sns.distplot(totrain["Predicted Result"], hist=False, color="red", label="Predicted Result")
plt.legend()
plt.show()


#Save the Model
with open("bodyfatmodel.pkl", "wb")as file:
    pickle.dump(model, file)