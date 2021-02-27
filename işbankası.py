# iş bankası
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
data_sample_submission = pd.read_csv('sample_submission.csv')
data_mounthly_expenditures = pd.read_csv('monthly_expenditures.csv')

d = pd.DataFrame(np.zeros((372106,1)))

data_yeni = data_train.merge(data_mounthly_expenditures, how='left', on = 'musteri')
data_test_yeni1=data_test.merge(data_mounthly_expenditures, how='left', on = 'musteri')
# data_test_yeni1=data_test_yeni.merge(data_sample_submission, how='left', on = 'musteri')
#1. Function to impute null value with new category
def impute_nan_create_category(DataFrame,ColName):
     DataFrame[ColName] = np.where(DataFrame[ColName].isnull(),"unknown",DataFrame[ColName])
## Call function to create new category for variables
for Columns in ['egitim','is_durumu','meslek_grubu']:
    impute_nan_create_category(data_yeni,Columns)
for Columns in ['egitim','is_durumu','meslek_grubu']:
    impute_nan_create_category(data_test_yeni1,Columns)
# #2. Display result
# data_train[['egitim','is_durumu','meslek_grubu']].head(10)

train_eğitim=data_yeni.iloc[:,3:4].values
train_işdurumu=data_yeni.iloc[:,4:5].values
train_meslekgrubu=data_yeni.iloc[:,5:6].values
train_sektor=data_yeni.iloc[:,8:9].values
train_target=data_yeni.iloc[:,7:8]


test_eğitim=data_test_yeni1.iloc[:,3:4].values
test_isdurumu=data_test_yeni1.iloc[:,4:5].values
test_meslekgrubu=data_test_yeni1.iloc[:,5:6].values
test_sektor=data_test_yeni1.iloc[:,7:8].values
# test_target=data_test_yeni1.iloc[:,11:]

# null=data_test_yeni.isnull().sum()

le=preprocessing.LabelEncoder()
train_eğitim[:,0]=le.fit_transform(data_yeni.iloc[:,3])
train_işdurumu[:,0]=le.fit_transform(data_yeni.iloc[:,4])
train_meslekgrubu[:,0]=le.fit_transform(data_yeni.iloc[:,5])
train_sektor[:,0]=le.fit_transform(data_yeni.iloc[:,8])

test_eğitim[:,0]=le.fit_transform(data_test_yeni1.iloc[:,3])
test_isdurumu[:,0]=le.fit_transform(data_test_yeni1.iloc[:,4])
test_meslekgrubu[:,0]=le.fit_transform(data_test_yeni1.iloc[:,5])
test_sektor[:,0]=le.fit_transform(data_test_yeni1.iloc[:,7])

ohe=preprocessing.OneHotEncoder()
train_eğitim=ohe.fit_transform(train_eğitim).toarray()
train_işdurumu=ohe.fit_transform(train_işdurumu).toarray()
train_meslekgrubu=ohe.fit_transform(train_meslekgrubu).toarray()
train_sektor=ohe.fit_transform(train_sektor).toarray()

test_eğitim=ohe.fit_transform(test_eğitim).toarray()
test_isdurumu=ohe.fit_transform(test_isdurumu).toarray()
test_meslekgrubu=ohe.fit_transform(test_meslekgrubu).toarray()
test_sektor=ohe.fit_transform(test_sektor).toarray()

sonuç1=pd.DataFrame(data=train_eğitim, index=range(560038),columns=['e1','e2','e3','e4','e5','e6'])
sonuç2=pd.DataFrame(data=train_işdurumu, index=range(560038),columns=['i1','i2','i3','i4','i5','i6','i7','i8','i9','i10','i11','i12','i13','i14'])
sonuç3=pd.DataFrame(data=train_meslekgrubu, index=range(560038),columns=['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12','m13','m14','m15','m16','m17','m18','m19','m20','m21'])
sonuç4=pd.DataFrame(data=train_sektor, index=range(560038),columns=['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13'])

test1=pd.DataFrame(data=test_eğitim, index=range(372106),columns=['e1','e2','e3','e4','e5'])
test2=pd.DataFrame(data=test_isdurumu, index=range(372106),columns=['i1','i2','i3','i4','i5','i6','i7','i8','i9','i10','i11','i12','i13'])
test3=pd.DataFrame(data=test_meslekgrubu, index=range(372106),columns=['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12','m13','m14','m15','m16','m17','m18','m19','m20','m21'])
test4=pd.DataFrame(data=test_sektor, index=range(372106),columns=['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13'])

s1=pd.concat([sonuç1,sonuç2],axis=1)
s2=pd.concat([s1,sonuç3],axis=1)
s3=pd.concat([s2,sonuç4],axis=1)
son_train=pd.concat([s3,data_yeni.iloc[:,[1,2,9,10]]],axis=1)

t1=pd.concat([test1,test2],axis=1)
t2=pd.concat([t1,test3],axis=1)
t3=pd.concat([t2,test4],axis=1)
son_test=pd.concat([t3,data_test_yeni1.iloc[:,[1,2,8,9]]],axis=1)
son_test1=pd.concat([son_test,d],axis=1)
son_test2=pd.concat([son_test1,d],axis=1)
müsteri=data_test_yeni1.iloc[:,[0]]
# null=son_train.isnull().sum()
son_train1,son__train_test1 =train_test_split(son_train,test_size=0.33, random_state=0)
train_target1,train_test_target1=train_test_split(train_target,test_size=0.33, random_state=0)
son_train1 = son_train1.sort_index()
son__train_test1=son__train_test1.sort_index()
train_target1=train_target1.sort_index()
train_test_target1=train_test_target1.sort_index()
# logistic
# from sklearn.linear_model import LogisticRegression
# logr = LogisticRegression(random_state=0,max_iter=1000)
# logr.fit(son_train1,train_target1.values.ravel())

# y_pred1 = logr.predict(son__train_test1)

# cm = confusion_matrix(train_test_target1,y_pred1)
# print("********************************************************************************")
# print("Logistic")
# print(cm)


# df = pd.DataFrame(data=y_pred1, index=range(372106),columns=['target'])
# sab=data_test_yeni1.iloc[:,0:1]
# sub=pd.concat([sab,df],axis=1)
# df1=[]
# df2=[]
# # duplicateRowsDF = sub[sub.duplicated(['musteri'])]
# b=sub.drop_duplicates()
# for i in range(48680):
#     f=b.iloc[i-1,0]
#     a=b.iloc[i,0]
#     c=b.iloc[i+1,0]
#     if a==c:
#         df1.append(a)
#         df2.append(1)
#         i=i+1
#     elif a==f:
#         continue
#     elif a!=c:
#         df1.append(a)
#         df2.append(b.iloc[i,1])
    
# x=pd.DataFrame(data=df1,columns=['musteri'])
# x1=pd.DataFrame(data=df2,columns=['target'])
# x2=pd.concat([x,x1],axis=1)
# file = open("submission.csv","w",encoding="utf-8")
# x2.to_csv (r'C:\Users\Samsung\Desktop\iş bankası yarışma\submission.csv', index = False, header=True)

# KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10,leaf_size=100 ,metric='minkowski')
knn.fit(son_train1,train_target1.values.ravel())

y_pred2 = knn.predict(son__train_test1)



fpr, tpr, thresholds = metrics.roc_curve(train_test_target1, y_pred2, pos_label=2)
metrics.auc(fpr, tpr)

# cm = confusion_matrix(train_test_target1,y_pred2)
print("********************************************************************************")
print("knn")
print(metrics.auc(fpr, tpr))



# df = pd.DataFrame(data=y_pred2, index=range(372106),columns=['target'])
# sab=data_test_yeni1.iloc[:,0:1]
# sub=pd.concat([sab,df],axis=1)
# df1=[]
# df2=[]
# # duplicateRowsDF = sub[sub.duplicated(['musteri'])]
# b=sub.drop_duplicates()
# for i in range(43791):
#     try:
#         f=b.iloc[i-1,0]
#         a=b.iloc[i,0]
#         c=b.iloc[i+1,0]
#         if a==c:
#             df1.append(a)
#             df2.append(1)
#             i=i+1
#         elif a==f:
#             continue
#         elif a!=c:
#             df1.append(a)
#             df2.append(b.iloc[i,1])
#     except:
#         f=b.iloc[i-1,0]
#         a=b.iloc[i,0]
#         if a==f:
#             continue
#         elif a!=f:
#             df1.append(a)
#             df2.append(b.iloc[i,1])

# x=pd.DataFrame(data=df1,columns=['musteri'])
# x1=pd.DataFrame(data=df2,columns=['target'])
# x2=pd.concat([x,x1],axis=1)
# file = open("submission1.csv","w",encoding="utf-8")
# x2.to_csv (r'C:\Users\Samsung\Desktop\iş bankası yarışma\submission1.csv', index = False, header=True)


# sc=StandardScaler()

# x_train = sc.fit_transform(son_train)
# x_test = sc.transform(son_test2)

# # SVC
# from sklearn.svm import SVC
# svc = SVC(kernel='rbf')
# svc.fit(x_train,train_target.values.ravel())

# y_pred3 = svc.predict(x_test)

# cm = confusion_matrix(test_target,y_pred3)
# print("********************************************************************************")
# print('SVC')
# print(cm)


# svc = SVC(kernel='linear')
# svc.fit(son_train,train_target.values.ravel())

# y_pred4 = svc.predict(son_test2)

# cm = confusion_matrix(test_target,y_pred4)
# print("********************************************************************************")
# print('SVC linear')
# print(cm)

# from sklearn.svm import SVC
# svc = SVC(kernel='poly')
# svc.fit(son_train,train_target.values.ravel())

# y_pred5 = svc.predict(son_test2)

# cm = confusion_matrix(test_target,y_pred5)
# print("********************************************************************************")
# print('SVC poly')
# print(cm)

# from sklearn.svm import SVC
# svc = SVC(kernel='sigmoid')
# svc.fit(son_train,train_target.values.ravel())

# y_pred6 = svc.predict(son_test2)

# cm = confusion_matrix(test_target,y_pred6)
# print("********************************************************************************")
# print('SVC sigmoid')
# print(cm)

# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# gnb.fit(son_train,train_target.values.ravel())

# y_pred7 = gnb.predict(son_test2)

# cm = confusion_matrix(test_target,y_pred7)
# print("********************************************************************************")
# print('GNB')
# print(cm)



# from sklearn.tree import DecisionTreeClassifier
# dtc = DecisionTreeClassifier(criterion = 'entropy')
# dtc.fit(son_train,train_target.values.ravel())

# y_pred8 = dtc.predict(son_test2)

# df = pd.DataFrame(data=y_pred8, index=range(372106),columns=['target'])
# sab=data_test_yeni1.iloc[:,0:1]
# sub=pd.concat([sab,df],axis=1)
# df1=[]
# df2=[]
# b=sub.drop_duplicates()
# index = b. index
# lvl= len(index)
# for i in range(lvl):
#     try:
#         f=b.iloc[i-1,0]
#         a=b.iloc[i,0]
#         c=b.iloc[i+1,0]
#         if a==c:
#             df1.append(a)
#             df2.append(1)
#             i=i+1
#         elif a==f:
#             continue
#         elif a!=c:
#             df1.append(a)
#             df2.append(b.iloc[i,1])
#     except:
#         f=b.iloc[i-1,0]
#         a=b.iloc[i,0]
#         if a==f:
#             continue
#         elif a!=f:
#             df1.append(a)
#             df2.append(b.iloc[i,1])

# x=pd.DataFrame(data=df1,columns=['musteri'])
# x1=pd.DataFrame(data=df2,columns=['target'])
# x2=pd.concat([x,x1],axis=1)
# file = open("submission2.csv","w",encoding="utf-8")
# x2.to_csv (r'C:\Users\Samsung\Desktop\iş bankası yarışma\submission2.csv', index = False, header=True)


# from sklearn.ensemble import RandomForestClassifier
# rfc = RandomForestClassifier(n_estimators=3, criterion = 'entropy')
# rfc.fit(son_train,train_target.values.ravel())

# y_pred9 = rfc.predict(son_test2)

# df = pd.DataFrame(data=y_pred9, index=range(372106),columns=['target'])
# sab=data_test_yeni1.iloc[:,0:1]
# sub=pd.concat([sab,df],axis=1)
# df1=[]
# df2=[]
# b=sub.drop_duplicates()
# index = b. index
# lvl= len(index)
# for i in range(lvl):
#     try:
#         f=b.iloc[i-1,0]
#         a=b.iloc[i,0]
#         c=b.iloc[i+1,0]
#         if a==c:
#             df1.append(a)
#             df2.append(1)
#             i=i+1
#         elif a==f:
#             continue
#         elif a!=c:
#             df1.append(a)
#             df2.append(b.iloc[i,1])
#     except:
#         f=b.iloc[i-1,0]
#         a=b.iloc[i,0]
#         if a==f:
#             continue
#         elif a!=f:
#             df1.append(a)
#             df2.append(b.iloc[i,1])

# x=pd.DataFrame(data=df1,columns=['musteri'])
# x1=pd.DataFrame(data=df2,columns=['target'])
# x2=pd.concat([x,x1],axis=1)
# file = open("submission3.csv","w",encoding="utf-8")
# x2.to_csv (r'C:\Users\Samsung\Desktop\iş bankası yarışma\submission3.csv', index = False, header=True)




