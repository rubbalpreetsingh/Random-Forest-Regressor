import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor



data = pd.read_csv("Position_Salaries.csv")
data



real_x = data.iloc[:,1:2].values
real_y = data.iloc[:,2].values
real_x




reg = RandomForestRegressor(n_estimators=400,random_state = 0) 




reg.fit(real_x,real_y)




y_pred = reg.predict([[6.5]])
y_pred




x_grid = np.arange(min(real_x),max(real_x),0.01)  
x_grid = x_grid.reshape(len(x_grid),1)
x_grid




plt.scatter(real_x,real_y,color = "green")
plt.plot(x_grid, reg.predict(x_grid), color = "blue")
plt.title("Random Forest Regressor")
plt.xlabel("Pos Level")
plt.ylabel("Salary")
plt.show()


