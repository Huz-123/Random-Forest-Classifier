# Random Forest Digits Classification Project

## Overview

This project demonstrates how to use a Random Forest Classifier to recognize handwritten digits from images using the scikit-learn `digits` dataset. The flowchart (see `Image 1`) visualizes the step-by-step process implemented in the notebook.

![Project Workflow](Image 1)

## Workflow Steps

1. **Import Libraries**  
   - Essential libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, and `scikit-learn`.

2. **Load Dataset**  
   - Load the scikit-learn `digits` dataset, which contains 8x8 pixel images of handwritten digits.

3. **Data Exploration**  
   - Visualize some samples.
   - Inspect data shapes, check for missing values, and basic statistics.

4. **Preprocessing**  
   - Convert the dataset into a pandas DataFrame.
   - Attach the target digit labels.
   - (Optional) Normalize or scale features if needed.

5. **Train-Test Split**  
   - Split the data into training and test sets using `train_test_split`.

6. **Model Training**  
   - Train a `RandomForestClassifier` (with 10,000 trees for demonstration) on the training data.

7. **Prediction**  
   - Use the trained model to predict digit labels for the test set.
   - You can also predict on external images by resizing and preprocessing them to match dataset format (8x8 grayscale, normalized).

8. **Evaluation**  
   - Evaluate the model using accuracy score.
   - Visualize results with a confusion matrix heatmap.
   - Optionally display a classification report.

## How to Run

1. **Install Dependencies**
   - All necessary libraries are available via `pip` and come with standard Python distributions like Anaconda.

2. **Run the Notebook**
   - Follow the notebook cells in order.
   - To test on your own image, preprocess the image as shown (resize to 8x8, grayscale, invert colors if needed, normalize to 0-16).

3. **Inspect Results**
   - The notebook will display accuracy, confusion matrix, and predictions.

## Example Code Snippet

```python
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
digits = load_digits()
df = pd.DataFrame(digits.data)
df['target'] = digits.target

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    df.drop(['target'], axis='columns'), df['target'], test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=10000)
model.fit(x_train, y_train)

# Evaluate
accuracy = model.score(x_test, y_test)
print("Test Accuracy:", accuracy)
```

## Notes & Tips

- For best results, use 100-200 trees in Random Forest (10,000 is for demonstration).
- Image should be preprocessed (resized to 8x8, grayscale, normalized) before prediction.
- The confusion matrix and classification report help you understand which digits are misclassified.

## References

- [Scikit-learn Digits Dataset Documentation](https://scikit-learn.org/stable/auto_examples/datasets/plot_digits_digits.html)
- [Random Forest Classifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

---
**Flowchart Reference:**  
See `Image 1` for the visual workflow of this project.
