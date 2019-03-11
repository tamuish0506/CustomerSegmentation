# Customer ID, incomes(*1000), gender and age came from records in membership
# cards, and spending scores are something that the mall assigned to each 
# customer based on some defined attributes, such as customer behaviors, 
# feedbacks or items in each purchase, ets.


print()
# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# load data
customer=pd.read_csv('.../Mall_Customers.csv')

# change column names
customer.columns = ['id', 'gender', 'age', 'income', 'spending_score']

# check if there's any null value 
customer.isnull().sum()

customer.head()
customer.info()
customer.describe()

# age distribution
plt.hist(customer.age)
plt.xlabel('Age')
plt.ylabel('Count of Customers')
plt.show()

# score distribution
plt.hist(customer.spending_score)
plt.xlabel('Score')
plt.ylable('Count of Customers')
plt.show()

# income distribution
plt.hist(customer.income)
plt.xlabel('Income *1000')
plt.ylabel('Count of Customers')
plt.show()

# assign age and income into seperated groups
# add age_group
def age_group(age):
    if age > 18 and age <= 25:
        return '18-25'
    elif age > 25 and age <= 35:
        return '26-35'
    elif age > 35 and age <= 45:
        return '36-45'
    elif age > 45 and age<= 55:
        return "46-55"
    elif age > 55 and age <= 65:
        return '56-65'
    else:
        return "over 65"
    
# income groups
def income_group(income):
    if income <= 30:
        return '<30k'
    if income > 30 and income <= 60:
        return '30001-60000'
    if income > 60 and income <= 90:
        return '60001-90000'
    if income > 90 and income <= 120:
        return '90001-120000'
    if income > 120 and income < 150:
        return '120001-150000'
    
    
customer['age_group']=customer.age.apply(age_group)
customer['income_group']=customer.income.apply(income_group)
customer['gender']=customer['gender'].map({'Female':1, 'Male':0})
customer.head()

# gender vs counts of customers
customer['gender'].value_counts()
y_1=[len(customer[customer.gender == 1]), len(customer[customer.gender == 0])]
x_1=['Female', 'Male']
sns.barplot(x=x_1, y=y_1)
plt.title("Number of Customer in Genders")
plt.xlabel("Gender")
plt.ylabel("Number of Customer")
plt.show()

# # of customers in age group
customer['age_group'].value_counts()
x_2=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
y_2=[len(customer[customer.age_group == '18-25']),
     len(customer[customer.age_group == '26-35']),
     len(customer[customer.age_group == '36-45']),
     len(customer[customer.age_group == '46-55']),
     len(customer[customer.age_group == '56-65']),
     len(customer[customer.age_group == 'over 65'])]

sns.barplot(x=x_2, y=y_2, palette="rocket")
plt.title("Number of Customer and Ages")
plt.xlabel("Age")
plt.ylabel("Number of Customer")
plt.show()

# # of customers in income groups
customer['income_group'].value_counts()
x_3=['<30k', '30k-60k', '60k-90k', '90k-120k', '120k-150k']
y_3=[len(customer[customer.income_group == '<30k']),
     len(customer[customer.income_group == '30001-60000']),
     len(customer[customer.income_group == '60001-90000']),
     len(customer[customer.income_group == '90001-120000']),
     len(customer[customer.income_group == '120001-150000'])]

sns.barplot(x=x_3, y=y_3, palette="rocket")
plt.title("Customers' Incomes")
plt.xlabel("Income")
plt.ylabel("Number of Customer")
plt.show()


# spending_score vs gender
sns.barplot(x='gender', y='spending_score', data=customer)
# spending_score vs age and gender
sns.barplot(x='age_group', y="spending_score", hue="gender", data=customer)
# income vs age and gender
sns.barplot(x='age_group',y='income',hue="gender", data=customer)
 
customer.groupby(['gender', 'age_group'])['income*k', 'spending_score'].mean()
plt.scatter(x = "income", y = "spending_score", data = customer)
plt.xlabel('Income(*1000)')
plt.ylabel('Customr Score')
plt.show()

# k-means clustering method by using income, age and spending scores features
# 3D charts
sns.set_style("white")
fig = plt.figure(figsize=(20,10))
ax=fig.add_subplot(111, projection='3d')
ax.scatter(customer['age'], customer['income'], customer['spending_score'], 
           c='blue', s=60)
ax.view_init(30, 185)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()

# The Elbow method
X=customer[['age', 'income', 'spending_score']]

kmeans_inertia=[]

for i in range(1, 15):
    kmeans=KMeans(i)
    kmeans.fit(X)
    kmeans_inertia.append(kmeans.inertia_)

kmeans_inertia

number_clusters = range(1,15)
plt.plot(number_clusters, kmeans_inertia)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster Sum of Squares')
plt.show()
# no. of clusters >=5 is ideal for categorizing customers->
# take no. of clusters = 5

# k-Means clustering 
kmeans=KMeans(5)
kmeans.fit(X)
customer['cluster']=kmeans.fit_predict(X)
customer.head()

# plot the clusters among features
fig = plt.figure(figsize=(20,10))
ax=fig.add_subplot(111, projection='3d')
ax.scatter(customer.age[customer.cluster == 0], 
           customer["income"][customer.cluster == 0], 
           customer["spending_score"][customer.cluster == 0], c='blue', s=60)
ax.scatter(customer.age[customer.cluster == 1], 
           customer["income"][customer.cluster == 1], 
           customer["spending_score"][customer.cluster == 1], c='red', s=60)
ax.scatter(customer.age[customer.cluster == 2], 
           customer["income"][customer.cluster == 2], 
           customer["spending_score"][customer.cluster == 2], c='green', s=60)
ax.scatter(customer.age[customer.cluster == 3], 
           customer["income"][customer.cluster == 3], 
           customer["spending_score"][customer.cluster == 3], c='purple', s=60)
ax.scatter(customer.age[customer.cluster == 4], 
           customer["income"][customer.cluster == 4], 
           customer["spending_score"][customer.cluster == 4], c='orange', s=60)
ax.view_init(30, 185)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()



