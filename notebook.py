"""
# IN-STK5000 Reproducibility assignment
"""

import handout # Tool for generating report-type documents
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

doc = handout.Handout('output') # handout: exclude

"""
Load the dataset and do label-encoder preprocessing to put all variables on a number format.
"""

features = ["Buying", "Maintenaince", "Doors", "Persons", "Luggage", "Safety"]
target = 'Class'
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', names=features + [target])

doc.add_text(df.head())# handout: exclude
doc.show()# handout: exclude

"""
Some setup code for the KNN classifier, data encoding, splitting and scaling:
"""
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df = df.apply(le.fit_transform) # convert the categorical columns into numeric (= all columns)

doc.add_text(df.head())# handout: exclude
doc.show()# handout: exclude
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data_scaled = pd.DataFrame(StandardScaler().fit_transform(df[features]), columns=features)
data_scaled[target] = df[target]
train_data_s, test_data_s = train_test_split(data_scaled, test_size=0.2)
"""
## Using cross-validation to test which k produces the best results
This code trains KNN models with different k using cross-validation. The point is to get an
impression of how our choice of k will affect the excpected accuracy on the test-set.
"""
from sklearn.model_selection import cross_val_score
neighbor_ks = range(1, 100)
untrained_models = [KNeighborsClassifier(n_neighbors=k) for k in neighbor_ks]
k_fold_scores = [cross_val_score(estimator=m, X=train_data_s[features], y=train_data_s[target], cv=10) for m in untrained_models]
mean_xv_scores = [s.mean() for s in k_fold_scores]
plt.errorbar(neighbor_ks, mean_xv_scores, yerr=[s.std() for s in k_fold_scores])
doc.add_figure(plt.gcf()) # handout: exclude
doc.show() # handout: exclude
plt.clf() # handout: exclude

"""
## Results of using different k's on the test-set
This code compares KNN models trained with different k on the testset, 
plotted alongside results on the training data and the results from the cross-validation.
"""
models = [KNeighborsClassifier(n_neighbors=k).fit(train_data_s[features], train_data_s[target]) for k in neighbor_ks]
train_scores = [accuracy_score(train_data_s[target], m.predict(train_data_s[features])) for m in models]
test_scores = [accuracy_score(test_data_s[target], m.predict(test_data_s[features])) for m in models]
plt.semilogx(neighbor_ks, train_scores, neighbor_ks, test_scores, neighbor_ks, mean_xv_scores)
plt.legend(["Train", "Test", "XV"])
doc.add_figure(plt.gcf()) # handout: exclude
doc.show() # handout: exclude

"""
## Conclusion
We expect the model to be slightly more accurate than the cross-validation on the test-set because the cross-validaton models are trained on a slightly smaller dataset. That is also what we observe in the above graph.
For a deployment to the "real-world", we'd assume the model to perform slightly worse because of possible sampling bias in the dataset we've used.
"""

"""
## Markdown comments
Comments with triple quotes are converted to text blocks.
Text blocks support [Markdown formatting][1], for example:
- Headlines
- Hyperlinks
- Inline `code()` snippets
- **Bold** and *italic*
[1]: https://commonmark.org/help/
"""

"""
## Add text and variables
Write to our handout using the same syntax as Python's `print()`:
"""
for index in range(3):
  doc.add_text('Iteration', index)
doc.show()

"""
## Add Matplotlib figures
Display matplotlib figures on the handout:
"""
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(np.arange(100))
fig.tight_layout()
doc.add_figure(fig)
doc.show()  # Display figure below this line.

"""
Set the width to display multiple figures side by side:
"""

for iteration in range(3):
  fig, ax = plt.subplots(figsize=(3, 2))
  ax.plot(np.sin(np.linspace(0, 20 / (iteration + 1), 100)))
  doc.add_figure(fig, width=0.33)
doc.show()

"""
## Add images and videos
This requires the `imageio` pip package.
"""
image_a = np.random.uniform(0, 255, (200, 400, 3)).astype(np.uint8)
image_b = np.random.uniform(0, 255, (100, 200, 1)).astype(np.uint8)
doc.add_image(image_a, 'png', width=0.4)
doc.add_image(image_b, 'jpg', width=0.4)
doc.show()
video = np.random.uniform(0, 255, (100, 64, 128, 3)).astype(np.uint8)
doc.add_video(video, 'gif', fps=30, width=0.4)
#doc.add_video(video, 'mp4', fps=30, width=0.4)
doc.show()

"""
## Exclude lines
Hide code from the handout with the `# handout: exclude` comment:
"""

# Invisible below:
value = 13  # handout: exclude

"""
## View the handout
The handout is automatically saved when you call `doc.show()`. Just open
`output/index.html` in your browser.
"""



