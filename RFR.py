import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor



data = pd.read_csv("Position_Salaries.csv")
data


# In[10]:


real_x = data.iloc[:,1:2].values #real_x = data.iloc[:,1] --> now it is vector..but we need array that why we select [1:2]--> means 2-1 =1
real_y = data.iloc[:,2].values
real_x


# In[11]:


reg = RandomForestRegressor(n_estimators=400,random_state = 0) #n_estimators:- we perform decision tree many times--> this represents the no. of pemorme ---> by default itis 10 


# In[12]:


reg.fit(real_x,real_y)


# In[13]:


y_pred = reg.predict([[6.5]])
y_pred


# In[14]:


y_pred = reg.predict([[6.4]])
y_pred


# In[15]:


x_grid = np.arange(min(real_x),max(real_x),0.01)  #(x,x,steps-->ressolution)
x_grid = x_grid.reshape(len(x_grid),1)
x_grid


# In[16]:


plt.scatter(real_x,real_y,color = "green")
plt.plot(x_grid, reg.predict(x_grid), color = "blue")
plt.title("Random Forest Regressor")
plt.xlabel("Pos Level")
plt.ylabel("Salary")
plt.show()


# In[17]:


# Good Result then Decision Tree Regression and Polynomial Regression

