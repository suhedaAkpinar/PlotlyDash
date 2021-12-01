# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 15:43:42 2020

@author: Administrator
"""


#kütüphaneler eklendi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

#Dosya yüklendi
data1 = pd.read_csv('telefon_veriseti.csv')

#Excelin içeriği görüntülendi
data1.head()
data=data1.copy()
data.info() #Özelliklerin veri tipleri incelendi.Çoğunluğu kategorik veri

empty_cols=['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']

for i in empty_cols:
    data[i]=data[i].replace(" ",np.nan)
    

#Veri setinde eksik sayımız varmı diye bakıldı
 #df_churn.isnull().any()   11 adet TotalCharges sutunu eksik cıktı 
print(data.isnull().sum().sort_values(ascending=False))

data.shape #7043 satır ve 21 sutun buludu
data.columns#bulunan sutunlar listelendi

#Sutunların içerdiği unique değerler listelendi
for item in data.columns:
 print(item)
 print (data[item].unique())
    

data.info()  
#TotalCharges i object olarak almış düzeltiyoruz
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.info() #float olarak düzeltildi  ancak float daha sonra sorun cıkardı
print(data.describe()) #Sutunların istatiksel özelliklerine bakalım



#önişleme adımlarından ilk olarak kategorik verilerimizi nümerik verilere dönüştürdük
# kategorik değerleri sayısal değerlere dönüştürtük 
#algoritmalarımızı uygulayabilmek için

data['gender'].replace(['Male','Female'],[0,1],inplace=True)
data['Partner'].replace(['Yes','No'],[0,1],inplace=True)
data['Dependents'].replace(['Yes','No'],[0,1],inplace=True)
data['PhoneService'].replace(['Yes','No'],[0,1],inplace=True)
data['MultipleLines'].replace(['No phone service','No', 'Yes'],[2,1,0],inplace=True)
data['InternetService'].replace(['No','DSL','Fiber optic'],[2,0,1],inplace=True)
data['OnlineSecurity'].replace(['No','Yes','No internet service'],[1,0,2],inplace=True)
data['OnlineBackup'].replace(['No','Yes','No internet service'],[1,0,2],inplace=True)
data['DeviceProtection'].replace(['No','Yes','No internet service'],[1,0,2],inplace=True)
data['TechSupport'].replace(['No','Yes','No internet service'],[1,0,2],inplace=True)
data['StreamingTV'].replace(['No','Yes','No internet service'],[1,0,2],inplace=True)
data['StreamingMovies'].replace(['No','Yes','No internet service'],[1,0,2],inplace=True)
data['Contract'].replace(['Month-to-month','One year','Two year'],[0,1,2],inplace=True)
data['PaperlessBilling'].replace(['Yes','No'],[0,1],inplace=True)

print(data.describe())

#Float sutunları int yaptık 
print(data[data['TotalCharges'].isnull()])
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.dropna(subset=['TotalCharges'])
data['TotalCharges'] = data['TotalCharges'].astype(int)

print(data[data['MonthlyCharges'].isnull()])
data['MonthlyCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.dropna(subset=['MonthlyCharges'])
data['MonthlyCharges'] = data['MonthlyCharges'].astype(int)


data.info()


#Target Variable data distribution
plt.figure(figsize=(6,6))
sns.countplot(x = data.Churn,palette='deep')
plt.xlabel('Customer churn', fontsize= 12)
plt.ylabel('Count', fontsize= 12)
plt.title("Distribution of Customer Churning ",fontsize= 20)
plt.show()



data['Churn'].replace(['Yes','No'],[0,1],inplace=True)

data.info() 


data.pop('customerID') 

corr = data.corr()#veri çerçevesindeki tüm sütunların çift yönlü korelasyonunu bulmak için kullanılır.
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})
#Burada verileri 2 boyutlu grafik şeklinde gösterirmek için kullandık
heat_map=plt.gcf()
heat_map.set_size_inches(20,15)
plt.xticks(fontsize=10) # x eksenini ayarlıyoruz
plt.yticks(fontsize=10)
plt.show()



data.info()


count_yes_churn = (data['Churn'] == 0).sum()
print("Çalkantılı müşteri sayısı:",count_yes_churn)

count_no_churn = (data['Churn']==1).sum()
print("Çalkantılı olmayan müşteri sayısı:",count_no_churn)

pct_of_no_churn = count_no_churn/(count_no_churn+count_yes_churn)
print("Çalkantılı olmayan müşterilerin yüzdesi:", pct_of_no_churn*100)

pct_of_yes_churn = count_yes_churn/(count_no_churn+count_yes_churn)
print("Çalkantılı olan müşterilerin yüzdesi:", pct_of_yes_churn*100)


#Hocaya sor
y=data['Churn']
X=data.drop(['Churn'], axis=1)










X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns =X_train.columns
 

 
from sklearn.ensemble import RandomForestClassifier
m1= RandomForestClassifier(n_estimators=50, min_samples_leaf=3, max_features=0.7, n_jobs=-1, oob_score=True)
m1.fit(X_train,y_train)
print("Örnekleme yapmadan önceki RandomForest Başarı oranı:")
print(m1.score(X_test,y_test))



from sklearn.metrics import confusion_matrix
ypred=m1.predict(X_test)
cmm=confusion_matrix(y_test,ypred)
print('Confusion Matrix\n',cmm)



sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('Aşırı örneklemeden(OverSampling) sonra train_X: {}'.format(X_train_res.shape))
print('Aşırı örneklemeden(OverSampling) sonra train_y: {} \n'.format(y_train_res.shape))

print("Aşırı Örneklemeden sonra '1' etiketinin sayısı: {}".format(sum(y_train_res==1)))
print("Aşırı Örneklemeden sonra '0' etiketinin sayısı: {}".format(sum(y_train_res==0)))

from sklearn.model_selection import train_test_split
y=y_train_res
X=X_train_res
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns =X_train.columns


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
clf.fit(X_train,y_train)

print("Örnekleme yapıldıktan sonra RandomForest Başarı oranı:")
print(clf.score(X_test,y_test))


ypred=clf.predict(X_test)
cm=confusion_matrix(y_test,ypred)
print('Confusion Matrix\n',cm)
