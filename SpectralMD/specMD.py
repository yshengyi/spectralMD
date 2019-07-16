#!/usr/bin/python
import pandas as pd
import numpy as np
from matplotlib import pylab
import matplotlib.pyplot as plt
from pylab import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import scipy.misc
import glob
from sklearn.cluster import KMeans
from pysptools.distance.dist import *


df = pd.DataFrame()
for pngfile in glob.glob('superballs_ms_*.png'):
    # print(pngfile)
    sfn = pngfile.strip('.png')
    name, x, y = sfn.split('_')

    png = imread(pngfile)
    sz = shape(png)

    reshaped = png.reshape((sz[0]*sz[1], 1))
    # print(reshaped)
    df[y]  = reshaped[:, 0]

sel = np.random.choice(range(sz[0]*sz[1]), 500)

png = imread('Red_Mask.png')
sz = shape(png)
print(sz)
reshaped = png.reshape((sz[0]*sz[1], sz[2]))

masklabel = [(reshaped[i] == [1, 1, 1, 1]).all() for i in range(sz[0]*sz[1])]
index = np.array(masklabel) == True
red_vec = df.loc[index].mean(axis = 0)
#print(red_vec)

# data = pd.read_csv("training.csv", skiprows=0)
# X = data.loc[:,'V1':'V31']
# Y = data[['Class']] == 'Red'
# df = X
# X = (df - df.min()) / (df.max() - df.min())

# ind = Y['Class']== True
# pink_vec =df.loc[ind].mean(axis = 0)
# print(pink_vec)
# print(SAM(red_vec, pink_vec))

num_cluster = 7

spec_angle = np.zeros((df.shape[0], 1))
sid_vec = np.zeros((df.shape[0], 1))
chebyshev_vec = np.zeros((df.shape[0], 1))
NormXCorr_vec = np.zeros((df.shape[0], 1))
for i in range(df.shape[0]):
    spec_angle[i] = SAM(df.values[i,:], red_vec)
    sid_vec[i] = SID(df.values[i,:], red_vec)
    chebyshev_vec[i] = chebyshev(df.values[i,:], red_vec)
    NormXCorr_vec[i] = NormXCorr(df.values[i,:], red_vec)
plt.figure()
plt.gray()
plt.imshow(spec_angle.reshape(sz[0], sz[1]))

plt.figure()
plt.imshow(sid_vec.reshape(sz[0], sz[1]))

plt.figure()
plt.imshow(chebyshev_vec.reshape(sz[0], sz[1]))

plt.figure()
plt.imshow(NormXCorr_vec.reshape(sz[0], sz[1]))

kmeans = KMeans(n_clusters= num_cluster, random_state=0).fit(df.values)
plt.figure()
plt.imshow(kmeans.labels_.reshape((sz[0], sz[1])))

df['SAM']  = spec_angle
df['Kmeans'] = kmeans.labels_
df['SID'] = sid_vec
df['chebyshev'] = chebyshev_vec
df['NormXCorr'] = NormXCorr_vec

#for label in range(num_cluster):
#    ind = kmeans.labels_ == label
#    print(ind)
#    pink_vec =df.loc[ind].mean(axis=0)
#    print SAM(red_vec, pink_vec)

#ind = Y['Class']== True
#X.loc[ind].plot.box()

#ind = Y['Class'] == False
#X.loc[ind].plot.box()

#kf = KFold(n_splits=10, shuffle=True)
#for train, test in kf.split(X):
    #print(test)
    #I train the classifier
#    clf = LDA()
#    clf.fit(X.as_matrix()[train],ravel(Y.as_matrix()[train]))

    #I make the predictions
#    predicted=clf.predict(X.as_matrix()[test])

    #I obtain the accuracy of this fold
    #ac=accuracy_score(predicted,Y.as_matrix()[test])

    #I obtain the confusion matrix
#    cm=confusion_matrix(Y.as_matrix()[test], predicted)

    #print cm
    #print

#for k in range(1,10):
#    clf = KNeighborsClassifier(k)
#    scores = cross_val_score(clf, X.as_matrix(), ravel(Y), cv=10)
#    print( "k =  %d " %k)
#    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#clf = LDA(store_covariance = True)
#clf = RandomForestClassifier()
clf = KNeighborsClassifier(5)
#clf = svm.SVC()

#scores = cross_val_score(clf, X.as_matrix(), ravel(Y.as_matrix()), cv=10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#clf.fit(X,ravel(Y))

#xtrain = np.zeros((len(kmeans.labels_),2))
#for i in range(len(kmeans.labels_)):
#    xtrain[i,0] = kmeans.labels_[i]
#    xtrain[i,1] = spec_angle[i]

train_data = df.loc[sel]
test_data = np.array(masklabel)[sel]
#print train_data
#print test_data
#clf.fit(df, ravel(Y))
clf.fit(train_data,test_data)

#testdata = pd.read_csv("test.csv", skiprows=0)
#testX = testdata.loc[:,'V1':'V31']
#testY = testdata[['Class']] == 'Red'

#df = testX
#spec_angle = np.zeros((df.shape[0], 1))
#for i in range(df.shape[0]):
#    spec_angle[i] = SAM(df.values[i,:], red_vec)

#kmeans = KMeans(n_clusters= num_cluster, random_state=0).fit(df.values)

#df['sam']  = spec_angle
#df['kmeans'] = kmeans.labels_

#testX = (df - df.min()) / (df.max() - df.min())
#prediction = clf.predict(testX)

#xtest = np.zeros((len(kmeans.labels_), 2))
#for i in range(len(kmeans.labels_)):
#    xtest[i,0] = kmeans.labels_[i]
#    xtest[i,1] = spec_angle[i]

#prediction = clf.predict(df)
#print

#testcm = confusion_matrix(testY,prediction)
#print(testcm)
#testacc  = np.float(testcm[0,0] + testcm[1,1])/testcm.sum()
#print(testacc)

#plt.figure()
#plt.gray()
#plt.imshow(X.corr())
#plt.colorbar()

# feature extraction
#test = SelectKBest(score_func=chi2, k="all")
#fit = test.fit(X, ravel(Y))
# summarize scores
#np.set_printoptions(precision=3)
#print(fit.scores_)
#plt.figure()
#xt = range(len(fit.scores_))
#plt.bar(xt, fit.scores_)
#plt.xticks(xt, X.columns.values.tolist())
#features = fit.transform(X)
# summarize selected features
#print(features[0:5, :])

#rfe = RFE(clf, 1)
#fit = rfe.fit(X, ravel(Y))
#print("Num Features: %d") % fit.n_features_
#print("Selected Features: %s") % fit.support_
#print("Feature Ranking: %s") % fit.ranking_

#print clf.feature_importances_
#plt.figure()
#xt = range(len(clf.feature_importances_))
#plt.bar(xt,clf.feature_importances_)
#plt.xticks(xt, X.columns.values.tolist())

#pca = PCA(n_components=15)
#X_pca = pca.fit_transform(X)
#print(pca.explained_variance_ratio_)

#plt.figure()
#boxplot(X_pca)

#clf.fit(X_pca,ravel(Y))

#channels = (df - df.min()) / (df.max() - df.min())  # normalized
#channels_pca = pca.transform(channels)
#print channels
#plt.figure()
#boxplot(channels_pca)

#prediction = clf.predict(channels_pca)
#pred = np.array([(x == True).all() for x in prediction])

#plt.figure()
#plt.imshow(pred.reshape((sz[0], sz[1])))


#xtest = np.zeros((len(kmeans.labels_), 2))
#for i in range(len(kmeans.labels_)):
#    xtest[i,0] = kmeans.labels_[i]
#    xtest[i,1] = spec_angle[i]

pred = clf.predict(df )
plt.figure()
plt.imshow(pred.reshape((sz[0], sz[1])))
#plt.imshow(png)

testcm = confusion_matrix(masklabel, pred)
print(testcm)
testacc  = np.float(testcm[0,0] + testcm[1,1]) / testcm.sum()
print(testacc)

show()
