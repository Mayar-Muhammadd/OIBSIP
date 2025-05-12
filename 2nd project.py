#Mayar Muhammad elsayed Muhammad

import pandas as pd
import numpy as np

# load the CSV file
df = pd.read_csv("ecommerce.csv")

# display the first few rows

print("First 5 Rows:")
print(df.head())

# get basic info
df.info()

# summary statistics for numerical columns
df.describe()

# check shape
print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns")

# check for missing values
print("Missing Values Per Column:")
print(df.isnull().sum())

duplicates = df.duplicated().sum()
print(f"Number of Duplicate Rows: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicates removed.")

# convert 'purchase_date' to datetime format
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], errors='coerce')
# check again for nulls after date conversion
print("Nulls in 'purchase_date' after conversion:")
print(df['Purchase Date'].isnull().sum())

# explore unique values for categorical columns
print("Unique Product Categories:")
print(df['Product Category'].unique())

print("Unique Payment Methods:")
print(df['Payment Method'].unique())

print("Unique Return Values:")
print(df['Returns'].unique())
#i'll fill the return column null values with 0  (assume No Return)
df['Returns'] = df['Returns'].fillna(0)


# save cleaned version, IT'S ALREADY CLEAN BUT I'LL STILL DO IT
df.to_csv("ecommerce_cleaned.csv", index=False)

#3)Descriptive Statistics: Calculate key metrics such as average purchase value, frequency of
# purchases, etc.

# loading the cleaned dataset
df = pd.read_csv("ecommerce_cleaned.csv")

# average purchase value
avg_purchase_value = df['Total Purchase Amount'].mean()
print(f" Average Purchase Value: {avg_purchase_value:.2f}")

# total number of purchases
total_purchases = df.shape[0]
print(f"🛒 Total Number of Purchases: {total_purchases}")

# total revenue
total_revenue = df['Total Purchase Amount'].sum()
print(f"Total Revenue: {total_revenue:.2f}")

# number of unique customers
unique_customers = df['Customer ID'].nunique()
print(f"Unique Customers: {unique_customers}")

# average quantity per purchase
avg_quantity = df['Quantity'].mean()
print(f"Average Quantity per Purchase: {avg_quantity:.2f}")

# purchase frequency per customer
purchase_frequency = total_purchases / unique_customers
print(f"Purchase Frequency per Customer: {purchase_frequency:.2f}")

# average revenue per customer
revenue_per_customer = total_revenue / unique_customers
print(f" Average Revenue per Customer: {revenue_per_customer:.2f}")

# return rate (% of orders marked as returns)
return_rate = df['Returns'].value_counts(normalize=True).get(1, 0) * 100
print(f"Return Rate: {return_rate:.2f}%")


#4)Customer Segmentation: Utilize clustering algorithms (e.g., K-means) to segment
#customers based on behavior and purchase patterns.

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# aggregate data per customer to capture behavior patterns
customer_behavior = df.groupby('Customer ID').agg({
    'Total Purchase Amount': 'sum',     # Total money spent
    'Purchase Date': 'count',           # Purchase frequency
    'Quantity': 'sum',                  # Total items bought
    'Returns': 'sum'                    # Return behavior
}).reset_index()

# select features for clustering 
X = customer_behavior[['Total Purchase Amount', 'Purchase Date', 'Quantity', 'Returns']]

# standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# elbow method to find optimal number of clusters
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

#  Plot elbow curve
plt.figure(figsize=(8,5))
plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method - Optimal Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# choose number of clusters based on elbow 
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
customer_behavior['Segment'] = kmeans.fit_predict(X_scaled)

# analyze segments (still using original names)
segment_summary = customer_behavior.groupby('Segment').mean().round(2)
print("\n📊 Segment Summary:\n", segment_summary)


#5)Visualization: Create visualizations (e.g., scatter plots, bar charts) to illustrate customer
#segments.

#scatter plot — spending vs Frequency (colored by segment)

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=customer_behavior,
    x='Total Purchase Amount',
    y='Purchase Date',
    hue='Segment',
    palette='Set2',
    s=100
)
plt.title("💸 Spending vs Purchase Frequency by Segment")
plt.xlabel("Total Purchase Amount")
plt.ylabel("Number of Purchases")
plt.grid(True)
plt.show()

#bar plot — average behavior by segment

# calculate average values per segment
segment_avg = customer_behavior.groupby('Segment')[['Total Purchase Amount', 'Quantity', 'Returns']].mean().round(2)

# plot bar charts for each feature
segment_avg.plot(kind='bar', figsize=(12,6))
plt.title("Average Customer Behavior per Segment")
plt.ylabel("Average Value")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
#quantity
segment_avg['Quantity'].plot(kind='bar', color='skyblue')
plt.title("Average Quantity per Segment")
plt.ylabel("Average Quantity")
plt.xlabel("Segment")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()
#reutrns
segment_avg['Returns'].plot(kind='bar', color='skyblue')
plt.title("Average Returns per Segment")
plt.ylabel("Average Returns")
plt.xlabel("Segment")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()
#totl purchase amount
segment_avg['Total Purchase Amount'].plot(kind='bar', color='skyblue')
plt.title("Average Total Purchase Amount per Segment")
plt.ylabel("Average Total Purchase Amount")
plt.xlabel("Segment")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()



#6)Insights and Recommendations: Analyze characteristics of each segment and provide
#insights.

#segment 3 – high-value customers
#insights:

#1)highest average total purchase amount and quantity

#2)shop frequently and regularly

#3)return rate is slightly higher 

#recommendations:

#exclusive offers: give early access to new products or private sales

#personalization: use their history to send custom recommendations


# segment 2 – growing customers
# insights:

#1)mid-to-high spending

#2)active purchasing behavior with moderate return rate

#3)consistent, but not as high-spending as segment 3

#recommendations:

# growth potential: send curated offers to push them into segment 3

#follow-up emails: post-purchase messages or product education

#goal: move them up the value ladder

#segment 1 – at-risk/low engagement
#insights:

#lowest purchase amount and frequency

#very low returns (may means that they’re inactive rather than unsatisfied)

#recommendations:

# re-engagement campaign: Win-back emails or discounts

#surveys/feedback: understand what’s holding them back

#goal: reactivate

# segment 0 – average customers
#insights:

#medium spenders with regular but not frequent purchases

#low return rate

#consistent but not too much

#recommendations:

# bundle deals: promote quantity-based or category combos

# automated reminders: restock alerts

# goal: increase engagement

 #general business recommendations

#focus area	action
#marketing:	segment-based email targeting and promotions
#product Strategy:	analyze return reasons and tweak product offerings
#personalization:	use behavior patterns and history to customize the shopping experience
#customer Service:	prioritize support for high-value customers

