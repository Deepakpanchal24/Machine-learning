# Import necessary libraries
from sklearn.decomposition import PCA  # Import PCA for dimensionality reduction
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import seaborn as sns  # Import Seaborn for advanced visualizations
from sklearn.datasets import load_iris  # Import Iris dataset from sklearn

# Load the iris dataset
iris = load_iris()  # Loads the Iris dataset (contains 3 species of iris flowers)
x = iris.data  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Labels (0, 1, 2 representing Setosa, Versicolor, Virginica)

# Apply PCA with 2 components
pca = PCA(n_components=2)  # Initialize PCA to reduce dimensions to 2
X_pca = pca.fit_transform(x)  # Fit PCA on the dataset and apply transformation

# Create a scatter plot of the PCA-transformed data
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', s=50)
# X_pca[:, 0] = Principal Component 1 (new feature after reduction)
# X_pca[:, 1] = Principal Component 2 (new feature after reduction)
# hue=y adds color to distinguish the different iris species
# palette='viridis' sets the color map
# s=50 sets the size of each data point

# Set plot labels and title
plt.title('PCA: Iris Dataset')  # Set the plot title
plt.xlabel('Principal Component 1')  # Label for X-axis
plt.ylabel('Principal Component 2')  # Label for Y-axis
plt.legend(title='Species')  # Add legend with the title 'Species'
plt.show()  # Display the plot
