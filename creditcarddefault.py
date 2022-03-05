#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas


# In[71]:


df = pandas.read_csv("Credit Card Default.csv")


# In[72]:


print(df)


# In[73]:


df = df.dropna()


# In[74]:


X = df.loc[:, ["income", "age", "loan"]]


# In[75]:


Y = df.loc[:, ["default"]]


# In[76]:


from sklearn.model_selection import train_test_split


# In[77]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y)


# In[78]:


from sklearn import tree


# In[79]:


model = tree.DecisionTreeClassifier(max_depth=3)


# In[80]:


from sklearn.metrics import confusion_matrix


# In[81]:


model.fit(X_train,Y_train)
pred = model.predict(X_test)
cm= confusion_matrix(Y_test, pred)
print(cm)
print((cm[0,0]+cm[1,1])/(sum(sum(cm))))


# In[82]:


import joblib


# In[83]:


joblib.dump(model, "CCD_DT")


# In[84]:


from sklearn import linear_model


# In[85]:


model = linear_model.LogisticRegression()


# In[86]:


model.fit(X_train,Y_train)
pred = model.predict(X_test)
cm= confusion_matrix(Y_test, pred)
print(cm)
print((cm[0,0]+cm[1,1])/(sum(sum(cm))))


# In[87]:


joblib.dump(model, "CCD_Reg")


# In[88]:


from sklearn.neural_network import MLPClassifier


# In[89]:


model = MLPClassifier(solver="lbfgs", hidden_layer_sizes=(6,6))


# In[90]:


model.fit(X_train,Y_train)
pred = model.predict(X_test)
cm= confusion_matrix(Y_test, pred)
print(cm)
print((cm[0,0]+cm[1,1])/(sum(sum(cm))))


# In[91]:


joblib.dump(model, "CCD_NN")


# In[92]:


from sklearn.ensemble import RandomForestClassifier


# In[93]:


model = RandomForestClassifier()


# In[94]:


model.fit(X_train,Y_train)
pred = model.predict(X_test)
cm= confusion_matrix(Y_test, pred)
print(cm)
print((cm[0,0]+cm[1,1])/(sum(sum(cm))))


# In[95]:


joblib.dump(model, "CCD_RF")


# In[96]:


from sklearn.ensemble import GradientBoostingClassifier


# In[97]:


model = GradientBoostingClassifier()


# In[98]:


model.fit(X_train,Y_train)
pred = model.predict(X_test)
cm= confusion_matrix(Y_test, pred)
print(cm)
print((cm[0,0]+cm[1,1])/(sum(sum(cm))))


# In[99]:


joblib.dump(model, "CCD_GB")


# In[ ]:




