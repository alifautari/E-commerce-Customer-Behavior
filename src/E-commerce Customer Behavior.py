#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Data and Required Packages
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[2]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


# In[3]:


import plotly.express as px
import plotly.io as pio  

# List of renderers 
preferred_renderers = ["notebook", "colab", "browser", "iframe"]

for r in preferred_renderers:
    try:
        pio.renderers.default = r
        break
    except Exception:
        continue


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib


# In[5]:


# Load The Dataset
df = pd.read_csv("ecommerce_customer_data_custom_ratios.csv")
df.head()


# In[6]:


# Delete Customer ID and Customer Age (duplicated column with 'Age' column)
df.drop(columns=['Customer ID', 'Customer Age'], inplace=True)
df.head()


# In[7]:


# Check duplicate values
df.duplicated().sum()


# In[8]:


# Check Null Value Counts and DataTypes of the features
df.info()


# In[9]:


# Summary of Data
df.describe(include='all')


# In[10]:


# Data Cleaning & Preprocessing
# Check empty values
df.isnull().sum()


# In[11]:


# Fill missing values
df['Returns'].fillna(df['Returns'].median(), inplace=True) #for numeric values: median


# In[12]:


# Convert to datetime
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], errors='coerce')
df.info()


# In[13]:


# Show Categories in columns
for i in df[['Product Category','Payment Method']]:
    print(f"The catagories in '{i}' are : ",list(df[i].unique()))


# In[14]:


#Export cleaned dataset
df.to_csv('ecommerce_customer_behavior_cleaned.csv', index=False)


# In[15]:


# Exploratory Data Analysis (EDA)
# Univariate Analysis
#Histogram
num_columns=df.select_dtypes(include=np.number).columns.to_list()
plt.figure(figsize=(14,10))
plt.suptitle("Univariate Analysis of Numerical Features",fontsize=20,fontweight='bold',alpha=0.8,y=1.)
for i, col in enumerate (num_columns):
    plt.subplot(3,2,i+1)
    sns.histplot(df[col], bins=20, kde=True)
    plt.title(f"Distribution of {col}")
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig("01_Univariate_Analysis_of_Numerical_Features.jpg", bbox_inches='tight')
plt.show()


# In[16]:


#Countplot
cat_columns=['Product Category','Payment Method', 'Gender']
plt.figure(figsize=(9,9))
plt.suptitle("Univariate Analysis of Categorical Features",fontsize=20,fontweight='bold',alpha=0.8,y=1.)
for i, col in enumerate (cat_columns):
    plt.subplot(3,1,i+1)
    sns.countplot(x=df[col], order=df[col].value_counts().index)
    plt.title(f"Distribution of {col}")
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig("02_Univariate_Analysis_of_Categorical_Features.jpg", bbox_inches='tight')
plt.show()


# In[17]:


# Multivariate Analysis
#Check Multicollinearity of Numerical Features
#Heatmap
plt.figure(figsize=(12,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="flare")
plt.title("Multivariate Analysis (Correlation of Numerical Features)")
plt.savefig("03_Multivariate_Analysis_(Correlation_of_Numerical_Features).jpg", bbox_inches='tight')
plt.show()


# In[18]:


#Total Purchase Amount by Age Group
if 'Age' in df.columns:
    plt.figure(figsize=(14,7))
    sns.boxplot(x='Age', y='Total Purchase Amount', data=df, palette='Paired')
    plt.title("Total Purchase Amount by Age Group")
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig("04_Total_Purchase_Amount_by_Age_(boxplot).jpg", bbox_inches='tight')
    plt.show()


# In[19]:


#Churn
df["Age Group"] = pd.cut(df["Age"], bins=[0,18,30,50,80], labels=["Teen","Young Adult","Adult","Senior"])
churn_columns=['Product Category','Payment Method', 'Gender', 'Age Group']
plt.figure(figsize=(14,7))
plt.suptitle("Churn vs Category",fontsize=20,fontweight='bold',alpha=0.8,y=1.)
for i, col in enumerate (churn_columns):
    plt.subplot(2,2,i+1)
    sns.countplot(data=df, x=col, order=df[col].value_counts().index, hue='Churn', palette='coolwarm')
    plt.title(f"Churn vs {col}")
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig("05_Churn_vs_Category.jpg", bbox_inches='tight')
plt.show()


# In[20]:


#Churn Rate
churn_rate_columns=['Product Category','Payment Method', 'Gender', 'Age Group']
plt.figure(figsize=(14,7))
plt.suptitle("Churn Rate vs Category",fontsize=20,fontweight='bold',alpha=0.8,y=1.)
for i, col in enumerate (churn_rate_columns):
    plt.subplot(2,2,i+1)
    sns.barplot(data=df, x=col, y="Churn", order=df[col].value_counts().index, estimator=lambda x:np.mean(x)*100, palette='prism')
    plt.ylabel('Churn Rate (%)')
    plt.title(f"Churn Rate vs {col}")
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig("06_Churn_Rate_vs_Category.jpg", bbox_inches='tight')
plt.show()


# In[21]:


#Product with highest return
# Total return by product category
returns_by_category = df.groupby("Product Category")["Returns"].sum().sort_values(ascending=False)

# Bar chart
plt.figure(figsize=(9,3))
sns.barplot(x=returns_by_category.values, y=returns_by_category.index, palette="Reds_r")
plt.title("Total Returns by Product Category")
plt.xlabel("Total Returns")
plt.ylabel("Product Category")
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig("07_Total_Returns.jpg", bbox_inches='tight')
plt.show()

# Pie chart 
plt.figure(figsize=(4,4))
returns_by_category.plot(kind="pie", autopct='%1.1f%%', colormap="Reds")
plt.ylabel("")
plt.title("Proportion of Returns by Product Categories")
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig("08_Proportion_of_Returns.jpg", bbox_inches='tight')
plt.show()


# In[22]:


#Return Rate
#Calculate Total Sold & Return by Categories
category_stats = df.groupby("Product Category").agg(
    total_sold=("Quantity", "sum"),
    total_returns=("Returns", "sum")
)

# Add column return rate
category_stats["return_rate"] = category_stats["total_returns"] / category_stats["total_sold"]

# Sort by return rate
category_stats_sorted = category_stats.sort_values("return_rate", ascending=False)
print(category_stats_sorted)

# Highest Return Rate Visualization 
plt.figure(figsize=(9,3))
sns.barplot(x="return_rate", y=category_stats_sorted.index, data=category_stats_sorted, palette="autumn")
plt.title("Product Categories with Highest Return Rate")
plt.xlabel("Return Rate")
plt.ylabel("Product Category")
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig("09_Return_Rate.jpg", bbox_inches='tight')
plt.show()


# In[23]:


#Trend
#Group by Purchase Date (monthly) and Product Category, then sum Total Purchase Amount
group_trend = df.groupby([
    df['Purchase Date'].dt.to_period('M').dt.to_timestamp(),  # group by month
    'Product Category'
])['Total Purchase Amount'].sum().reset_index()


#Convert sales to thousands
group_trend.loc[:, 'Total Purchase Amount (Millions)'] = group_trend['Total Purchase Amount'] / 1e6

# Plot
plt.figure(figsize=(14, 7))
sns.lineplot(data=group_trend, x='Purchase Date', y='Total Purchase Amount (Millions)', hue='Product Category', marker="d")
plt.title('Monthly Total Purchase Amount', fontsize=20)
plt.xlabel('Purchase Date')
plt.ylabel('Total Purchase Amount (Millions, $)')
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # every 1 month
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # format as Jan 2015
plt.xticks(rotation=90)
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.grid(True)
plt.savefig("10_Monthly_Total_Purchase_Amount.jpg", bbox_inches='tight')
plt.show()


# In[24]:


#Total Purchase by Customer Product
top_number = (df['Customer Name']+' - '+df['Product Category']).value_counts().head(10).reset_index()
top_number.columns = ['Customer-Product', 'Total Purchase']

max_val = top_number['Total Purchase'].max()

fig = px.bar(
    top_number,
    x='Customer-Product',
    y='Total Purchase',
    title='Top 10 Total Purchase by Customer Product',
    text='Total Purchase'
)

fig.update_traces(textposition='outside')

fig.update_layout(
    yaxis=dict(range=[0, max_val * 1.2]),
    xaxis_tickangle=-45
)

fig.show()
fig.write_html("11_Top_10_Total_Purchase_by_Customer-Product.html")


# In[25]:


#High Valuer Customer-Product
# Combine brand and model
df['Customer Product'] = df['Customer Name']+' - '+df['Product Category']

# Group by Customer Product and sum the Total Purchase Amount
top_customer_product = (
    df.groupby('Customer Product')['Total Purchase Amount']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
top_customer_product['Total Purchase Amount (Thousands, $)'] = (top_customer_product['Total Purchase Amount'] / 1e3).round(1)

# Get max total purchase amount value to set y-axis limit
max_val = top_customer_product['Total Purchase Amount (Thousands, $)'].max()

# Create the bar chart
fig = px.bar(
    top_customer_product,
    x='Customer Product',
    y='Total Purchase Amount (Thousands, $)',
    title='Top 10 High Valuer Customer-Product',
    text='Total Purchase Amount (Thousands, $)'
)

fig.update_traces(textposition='outside')

fig.update_layout(
    yaxis=dict(range=[0, max_val * 1.2]),  # 20% headroom
    yaxis_title='Total Purchase Amount (Thousands, $)',
    xaxis_tickangle=-45,
)

fig.show()
fig.write_html("12_Top 10_High_Valuer_Customer-Product.html")


# In[26]:


df.info()


# In[27]:


#Predictive Modeling
df.drop(columns=["Purchase Date", "Customer Name", "Age Group", "Customer Product"], inplace=True)

# Encode categorical columns
cat_cols = df.select_dtypes(include=["object"]).columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(columns=["Churn"])
y = df["Churn"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[28]:


#Helper function to evaluate models
def evaluate(name, model, X, y):
    y_pred = model.predict(X)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred)
    }

results = []


# In[29]:


#Baseline Models
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)
results.append(evaluate("Baseline LogReg", logreg, X_test_scaled, y_test))

dtree = DecisionTreeClassifier(max_depth=5, random_state=42)
dtree.fit(X_train, y_train)
results.append(evaluate("Baseline DecisionTree", dtree, X_test, y_test))


# In[30]:


#SMOTE Models
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Logistic Regression + SMOTE
X_train_res_scaled = scaler.fit_transform(X_train_res)
logreg_smote = LogisticRegression(max_iter=1000)
logreg_smote.fit(X_train_res_scaled, y_train_res)
results.append(evaluate("SMOTE LogReg", logreg_smote, X_test_scaled, y_test))

# Decision Tree + SMOTE
dtree_smote = DecisionTreeClassifier(max_depth=5, random_state=42)
dtree_smote.fit(X_train_res, y_train_res)
results.append(evaluate("SMOTE DecisionTree", dtree_smote, X_test, y_test))


# In[31]:


#Final Result
df_results = pd.DataFrame(results)
print(df_results)

#Export Models
joblib.dump(logreg, "baseline_logreg.pkl")
joblib.dump(dtree, "baseline_dtree.pkl")
joblib.dump(logreg_smote, "smote_logreg.pkl")
joblib.dump(dtree_smote, "smote_dtree.pkl")
print("Export Models to .pkl: Success")


# In[32]:


def plot_confusion(ax, y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Wistia", cbar=False,
                xticklabels=['No Churn (0)', 'Churn (1)'],
                yticklabels=['No Churn (0)', 'Churn (1)'],
               ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Logistic Regression Baseline
y_pred_baseline_logreg = logreg.predict(X_test)
plot_confusion(axes[0,0], y_test, y_pred_baseline_logreg, "Baseline Logistic Regression")

# Decision Tree Baseline
y_pred_baseline_dt = dtree.predict(X_test)
plot_confusion(axes[0,1], y_test, y_pred_baseline_dt, "Baseline Decision Tree")

# Logistic Regression SMOTE
y_pred_smote_logreg = logreg_smote.predict(X_test)
plot_confusion(axes[1,0], y_test, y_pred_smote_logreg, "SMOTE Logistic Regression")

# Decision Tree SMOTE
y_pred_smote_dt = dtree_smote.predict(X_test)
plot_confusion(axes[1,1], y_test, y_pred_smote_dt, "SMOTE Decision Tree")

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig("13_Confusion_Matrix_Comparison.jpg", bbox_inches='tight')
plt.show()


# In[ ]:




