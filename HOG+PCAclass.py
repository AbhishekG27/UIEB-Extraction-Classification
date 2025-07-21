import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# Load features and labels
features_path = "D:/IIT/Combined_PCA_HOG_color_features.npy"
labels_path = "D:/IIT/Combined_PCA_HOG_color_labels.npy"
features = np.load(features_path)
labels = np.load(labels_path)

# Check the unique labels to ensure they are correct
print("Unique labels:", np.unique(labels))

# Reduce dimensionality with PCA
pca = PCA(n_components=500)  # Adjust the number of components based on memory limitations and performance
features_reduced = pca.fit_transform(features)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features_reduced, labels, test_size=0.2, random_state=42, stratify=labels)

# Define classifiers with hyperparameter grids
classifiers = {
    "Gaussian Naive Bayes": (GaussianNB(), {}),
    "K-Nearest Neighbors": (KNeighborsClassifier(), {'classifier__n_neighbors': [3, 5, 7, 9]}),
    "Logistic Regression": (LogisticRegression(max_iter=1000), {'classifier__C': [0.1, 1, 10]}),
    "SVM": (SVC(), {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['rbf', 'linear']}),
    "MLP": (MLPClassifier(max_iter=1000), {'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)]}),
    "Random Forest": (RandomForestClassifier(), {'classifier__n_estimators': [50, 100, 200]}),
    "Gradient Boosting": (GradientBoostingClassifier(), {'classifier__n_estimators': [50, 100, 200], 'classifier__learning_rate': [0.01, 0.1]}),
    "Decision Tree": (DecisionTreeClassifier(), {'classifier__max_depth': [5, 10, None]})
}

# Create a pipeline with scaling and classifier
def create_pipeline(clf):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])

# Train and evaluate each classifier
best_classifiers = {}

for name, (clf, param_grid) in classifiers.items():
    print(f"Training {name}...")
    pipeline = create_pipeline(clf)
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    y_pred = grid_search.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    
    best_classifiers[name] = (grid_search.best_estimator_, accuracy)

# Create a voting classifier
voting_clf = VotingClassifier(
    estimators=[(name, clf) for name, (clf, _) in best_classifiers.items()],
    voting='soft'
)

voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)
voting_accuracy = accuracy_score(y_test, y_pred_voting)

print("\nVoting Classifier Results:")
print(f"Accuracy: {voting_accuracy:.2f}")
print(classification_report(y_test, y_pred_voting))

# Find the best individual classifier
best_classifier = max(best_classifiers.items(), key=lambda x: x[1][1])
print(f"\nBest individual classifier: {best_classifier[0]} with accuracy {best_classifier[1][1]:.2f}")
