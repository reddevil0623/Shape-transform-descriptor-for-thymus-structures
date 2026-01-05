import sys
sys.path.append('eucalc_directory')
import eucalc as ec
import os
import numpy as np
import matplotlib.cm as cm
from PIL import Image
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
datafolder = "Old_Young_Comparison"

all_files = os.listdir(datafolder)
names = [file for file in all_files if file.lower().endswith(('.tif', '.tiff'))]
names_k8 = [nm for nm in names if 'K8' in nm]


'''# Get y-range for the vectorizations'''

def compute_ect(names, datafolder, k=200, points=200, interval=(-2., 2.)):
    data = {}
    ect = np.empty((len(names), k, points), dtype=float)
    thetas = np.linspace(0, 2 * np.pi, k, endpoint=False)
    name_idx = 0
    T = np.linspace(interval[0], interval[1], points)
    for nm in names:
        file_path = os.path.join(datafolder, nm)
        img_array = np.array(Image.open(file_path))
        data[nm] = img_array
        cplx = ec.EmbeddedComplex(img_array)
        cplx.preproc_ect()
        idx = 0 
        

        for theta in thetas:
            x = np.cos(theta)
            y = np.sin(theta)
            direction = np.array([x, y])
            ect_dir = cplx.compute_euler_characteristic_transform(direction)
            ect_vals = np.array([ect_dir.evaluate(t) for t in T])
            ect[name_idx, idx] = ect_vals
            idx += 1
        
        name_idx += 1
    max_value = np.max(ect)
    min_value = np.min(ect)
    return ect, max_value, min_value

def extract_group(name):
    return name[0]+name.split('_')[1]
all_files = os.listdir(datafolder)
names = [file for file in all_files if file.lower().endswith(('.tif', '.tiff'))]
ect_k8,max_value_k8, min_value_k8 = compute_ect(names_k8, datafolder, interval = (-1.5, 1.5))
flattened_k8 = [ect.flatten() for ect in ect_k8]
print(f"Max value: {max_value_k8}, Min value: {min_value_k8}")

'''
# SampEuler vectorization Computation
'''

class EctImg:
    def __init__(self, nm, img, k=20, xinterval=(-1., 1.), xpoints=100, yinterval=(-1., 1.), ypoints=100):
        self.xinterval = xinterval
        self.yinterval = yinterval
        self.xpoints = xpoints
        self.ypoints = ypoints
        self.image = self.compute(img, k, xinterval, xpoints, yinterval, ypoints)
        self.nm = nm
    def compute(self, img, k, xinterval, xpoints, yinterval, ypoints):
        cplx = ec.EmbeddedComplex(img)
        cplx.preproc_ect()
        thetas = np.linspace(0, 2 * np.pi, k + 1)
        ect1 = np.empty((k, xpoints), dtype=float)
        for i in range(k):
            theta = thetas[i]
            direction = np.array((np.sin(theta), np.cos(theta)))
            ect_dir = cplx.compute_euler_characteristic_transform(direction)
            T = np.linspace(xinterval[0], xinterval[1], xpoints)
            ect1[i] = [ect_dir.evaluate(t) for t in T]

        image = np.zeros((ypoints, xpoints), dtype=float)
        yvalues = np.linspace(yinterval[0], yinterval[1], ypoints+1, endpoint=True)
        for i in range(xpoints):
            column = ect1[:, i]
            for j in range(ypoints):
                value = 0
                if j < ypoints-1:
                    value = len(np.where((yvalues[j] <= column) & (column < yvalues[j+1]))[0])/k
                else:
                    value = len(np.where((yvalues[j] <= column) & (column <= yvalues[j+1]))[0])/k
                image[j, i] = value
        return image
    
    def plot(self):
        plt.figure(figsize=(10, 8))
        plt.imshow(self.image, aspect='auto', extent=[self.xinterval[0], self.xinterval[1], self.yinterval[0], self.yinterval[1]], origin='lower', interpolation='none')
        plt.colorbar(label='Density')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('ECT Image Plot for '+ self.nm)
        plt.show()


def compute_ExIm(names, datafolder, k=480, xinterval=(-1.5, 1.5), xpoints=300, yinterval=(-450., 50.), ypoints=500):
    ExImage = []
    
    for nm in names: 
        file_path = os.path.join(datafolder, nm)
        with Image.open(file_path) as img:
            img_array = np.array(img)
        
        # Now, compute the SampEuler vectorization using the new array
        ect = EctImg(nm, img_array, k, xinterval, xpoints, yinterval, ypoints)
        exim = ect.compute(img_array, k, xinterval, xpoints, yinterval, ypoints)
        ExImage.append(exim)
    return ExImage

exims = compute_ExIm(names_k8, datafolder,k=360, xinterval=(-1.5, 1.5), xpoints=300, yinterval=(-60, 40), ypoints=100)
flattened_k8 = [image.flatten() for image in exims]

'''
Best parameters search for Random Forest Classifier
'''

def extract_age(name):
    return name[0]

ages_k8 = [extract_age(name) for name in names_k8]
X = np.array(flattened_k8)
y = np.array(ages_k8)
unique_ages = np.unique(y)
ypoints, xpoints = 110, 300 

# Number of trials for train-test splitting
num_trials = 50
age = unique_ages[0]
print(f"Processing age group: {age}")

# Binary classification: 1 if the sample belongs to the current age, 0 otherwise
binary_labels = np.where(y == age, 1, 0)

scores = []

# Parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],       # Number of trees in the forest
    'max_depth': [3, 5, 7, None],         # Tree depth (None means no limit)
    'min_samples_split': [2,3, 5, 10],      # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4]
}

# Perform Grid Search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=40),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    n_jobs=-1
)
grid_search.fit(X, binary_labels)

# Best parameters
best_params = grid_search.best_params_
print(f"Best parameters for age group {age}: {best_params}")


'''
SHAP Analysis with the best parameters
'''

def extract_age(name):
    return name[0]

ages_k8 = [extract_age(name) for name in names_k8]
X = np.array(flattened_k8)
y = np.array(ages_k8)
unique_ages = np.unique(y)
ypoints, xpoints = 110, 300 
num_trials = 50

for age in unique_ages:
    print(f"Processing age group: {age}")
    binary_labels = np.where(y == age, 1, 0)
    shap_values_aggregate_1 = None  
    shap_values_aggregate_0 = None  
    n_1 = 0  
    n_0 = 0  
    scores = []
    
    for trial in range(num_trials):
        X_train, X_test, y_train, y_test = train_test_split(X, binary_labels, test_size=0.2, random_state=40+trial)
        rf_classifier = RandomForestClassifier(max_depth = 3, min_samples_leaf=1, min_samples_split=10, n_estimators=200, random_state=40)
        rf_classifier.fit(X_train, y_train)
        score = rf_classifier.score(X_test, y_test)
        scores.append(score)
        
        if score > 0.8:
            explainer = shap.TreeExplainer(rf_classifier)
            shap_values = explainer.shap_values(X_test)
            shap_values_class_1 = shap_values[:, :, 1]
            shap_values_class_0 = shap_values[:, :, 0]

            # Separate SHAP values for class 1 and class 0
            shap_values_1 = shap_values_class_1[y_test == 1]
            shap_values_0 = shap_values_class_0[y_test == 0]

            # Initialize aggregate storage for class 1 and class 0
            if shap_values_aggregate_1 is None:
                shap_values_aggregate_1 = np.zeros((ypoints, xpoints))
            if shap_values_aggregate_0 is None:
                shap_values_aggregate_0 = np.zeros((ypoints, xpoints))

            # Sum SHAP values for class 1
            if len(shap_values_1) > 0:
                shap_values_mean_1 = shap_values_1.mean(axis=0)  # Average over samples classified as 1
                shap_image_1 = shap_values_mean_1.reshape(ypoints, xpoints)
                shap_values_aggregate_1 += shap_image_1
                n_1 += 1

            # Sum SHAP values for class 0
            if len(shap_values_0) > 0:
                shap_values_mean_0 = shap_values_0.mean(axis=0)  # Average over samples classified as 0
                shap_image_0 = shap_values_mean_0.reshape(ypoints, xpoints)
                shap_values_aggregate_0 += shap_image_0
                n_0 += 1

    # Average the SHAP values across all trials
    if n_1 > 0:
        shap_values_avg_1 = shap_values_aggregate_1 / n_1
    if n_0 > 0:
        shap_values_avg_0 = shap_values_aggregate_0 / n_0
        
    diff  = shap_values_avg_1 - shap_values_avg_0
    print(f'average accuracy for age {age} is {np.mean(scores)}')
    # Visualize the averaged SHAP values for class 1 (samples classified as 1)
    downsampled_importance_1 = block_reduce(shap_values_avg_1, block_size=(1, 2), func=np.mean)
    plt.figure(figsize=(10, 8))
    plt.imshow(downsampled_importance_1, aspect='auto', origin='lower', cmap='viridis', extent = [-1.5, 1.5, -60, 50])
    plt.colorbar(label='Average SHAP Value')
    plt.title(f"Average SHAP values for samples classified as {age}")
    plt.show()
    
    downsampled_importance_0 = block_reduce(shap_values_avg_0, block_size=(1, 2), func=np.mean)
    plt.figure(figsize=(10, 8))
    plt.imshow(downsampled_importance_0, aspect='auto', origin='lower', cmap='viridis', extent = [-1.5, 1.5, -60, 50])
    plt.colorbar(label='Average SHAP Value')
    plt.title(f"Average SHAP values for samples classified as not {age}")
    plt.show()

    downsampled_importance_diff = block_reduce(diff, block_size=(1, 2), func=np.mean)
    plt.figure(figsize=(10, 8))
    plt.imshow(downsampled_importance_diff, aspect='auto', origin='lower', cmap='viridis', extent = [-1.5, 1.5, -60, 50])
    plt.colorbar(label='Average SHAP Value')
    plt.show()
