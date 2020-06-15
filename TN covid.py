
# coding: utf-8

# # COVID-19 CASES PREDICTOR IN THE STATE OF TAMILNADU IN INDIA 
# ## using the dataset available in TN govt. server 
# ## using graphlab create
# ## linear regression model
# ## there are two models created using different features(just to compare the accuracy)

# In[1]:

import graphlab as gl
people=gl.SFrame("Public.csv") ##read the dataset


# In[2]:

x,y=people.dropna_split(columns=None) ##remove the none columns


# In[3]:

people.head()


# In[4]:

x.head() ##after removing none columns


# In[5]:

#x is the dataset of covid 19 in the state of Tamilnadu in India


# In[6]:

train_data,test_data=x.random_split(0.8,seed=0) #training data and test data
my_features=["DATE","NEW_CASES","TOTAL_CONFIRMED","NEW_CONFIRMED","POS_TESTS","NEG_TESTS","TOTAL_TESTS","NEW_DEATHS",
"TOTAL_DEATHS","NEW_RECOVERED",
"TOTAL_RECOVERED",
"NEW_ACTIVE",
"TOTAL_ACTIVE",
"NEW_HOSP",
"TOTAL_HOSP"] #setting the features


# In[7]:

predictor=gl.linear_regression.create(train_data,target="TOTAL_CASES",features=my_features) ##it will start Newton's Method


# In[8]:

predictor.evaluate(test_data) #evaluate the accuracy of this model


# In[9]:

feature1=["TOTAL_TESTS"]
predictor1=gl.linear_regression.create(train_data,target="TOTAL_CASES",features=feature1) ##another model


# In[10]:

predictor1.evaluate(test_data) #evaluate the accuracy of model no.2


# In[11]:

predictor.predict(test_data) ##predictor
predictor1.predict(test_data) ##predictor1

test_data.head() ##compare the predicted values and the original values
# In[12]:

##Put the new features in a dataset and use .predict() function to predict the data


# In[13]:

##Thank you 
##by Sujithkumar M A


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



