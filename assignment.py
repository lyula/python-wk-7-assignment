import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns  # Optional, for enhanced aesthetics

# Task 1: Load and Explore the Dataset
# Load the Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Add the target column to the DataFrame
df['target'] = iris.target

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Explore data types and missing values
print("\nInformation about the dataset:")
print(df.info())

# Check for missing values
print("\nNumber of missing values in each column:")
print(df.isnull().sum())

# Task 2: Basic Data Analysis
# Basic statistics of numerical columns
print("\nDescriptive statistics:")
print(df.describe())

# Grouping by species and computing the mean of features
print("\nMean of features grouped by species:")
print(df.groupby('target').mean())

# Renaming the target values for better readability
target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species'] = df['target'].map(target_names)
print("\nMean of features grouped by species (with species names):")
print(df.groupby('species').mean())

# Identifying patterns or interesting findings:
print("\nObservations:")
print("- The setosa species generally has smaller sepal and petal measurements compared to versicolor and virginica.")
print("- Virginica tends to have the largest sepal length and petal length among the three species.")
print("- There's a noticeable difference in petal width between the species, which could be a key distinguishing feature.")

# Task 3: Data Visualization
# Line chart (not directly applicable to this dataset in a meaningful way without time-series data)
# For demonstration, let's create a simple line plot of the mean of sepal length for each species index
mean_sepal_length = df.groupby('target')['sepal length (cm)'].mean()
plt.figure(figsize=(8, 6))
plt.plot(mean_sepal_length.index, mean_sepal_length.values, marker='o')
plt.title('Mean Sepal Length per Species (Index)')
plt.xlabel('Species Index')
plt.ylabel('Mean Sepal Length (cm)')
plt.xticks(mean_sepal_length.index, ['setosa', 'versicolor', 'virginica'])
plt.grid(True)
plt.show()

# Bar chart: Average petal length per species
avg_petal_length = df.groupby('species')['petal length (cm)'].mean()
plt.figure(figsize=(8, 6))
sns.barplot(x=avg_petal_length.index, y=avg_petal_length.values, palette='viridis')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# Histogram: Distribution of sepal width
plt.figure(figsize=(8, 6))
plt.hist(df['sepal width (cm)'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot: Sepal length vs. petal length
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, palette='Set2')
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.grid(True)
plt.show()
