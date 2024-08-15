# Fitness Tracker

This project performs data mining on exercise data and then implements a custom machine learning algorithm to predict the type of workout being done based on gyroscope and accelerometer readings.

## Project Steps

### 1. Project Setup (git clone this repo)

Download Dataset:
https://mbientlab.com/metamotions/

Add to data/raw/MetaMotion

### 2. Processing Raw Data [make_dataset.py](src/data/make_dataset.py)

In this step, we process the raw gyroscope and accelerometer data collected from barbell exercises:

Reading Data: We read and load the raw data files (CSV format) for both the accelerometer and gyroscope sensors into pandas DataFrames. Each file is parsed to extract relevant features such as the participant, exercise label, and exercise category.

Data Handling: We combine all the individual files into a single dataset for the accelerometer and gyroscope readings. We process timestamps to ensure consistent datetime indexing across the data, and we drop unnecessary columns to clean the data.

Data Merging: We merge the processed accelerometer and gyroscope datasets based on the timestamps to create a unified dataset that captures both sets of readings for each time interval.

Resampling: The combined dataset is resampled to a consistent frequency (e.g., 200ms intervals) to standardize the data. The resampled data is then split by day and further processed to remove any null values.

Exporting Processed Data: Finally, the processed and resampled dataset is saved as a pickle file for efficient storage and faster loading in subsequent steps of the project.

This processing step ensures that the raw data is cleaned, standardized, and ready for further analysis and visualization in the next stages of the project.

### 3. Data Visualization [visualize.py](src/visualization/visualize.py)

In this step, we visualize the processed accelerometer and gyroscope data to gain insights and understand patterns in the exercise data:

Loading Data: We start by loading the processed data from the previous step. The participant column is adjusted to remove the initial character for consistency.

Single Column Plotting: We plot individual sensor readings, such as acc_y for a specific set, to observe trends and identify any anomalies or patterns in the data.

Exercise-Wise Plotting: For each unique exercise label, we generate plots to visualize how the accelerometer readings change over time. This helps in identifying distinctive patterns for different exercises.

Plot Customization: We adjust the plot settings using Matplotlib’s styling options to improve the readability and presentation of the plots.

Comparative Analysis:

We compare accelerometer readings across different exercise categories (e.g., medium vs. heavy sets) and participants to see how the sensor data varies.
We also plot multiple axes (e.g., acc_x, acc_y, acc_z) together for specific exercises and participants to observe the full range of sensor data.
Automated Plot Generation: We create loops to automate the generation of plots for all combinations of exercises and participants, both for accelerometer and gyroscope data. This helps in systematically visualizing the data across the entire dataset.

Combined Plots: We create combined plots that display both accelerometer and gyroscope readings together, allowing for a comprehensive view of the data during different exercises.

Exporting Plots: Finally, all generated plots are saved as image files for documentation and further analysis, providing a visual record of the data for each exercise and participant.

This visualization step helps in understanding the structure and behavior of the exercise data, which is essential for the subsequent steps of building features and training machine learning models.

### 4. Detecting Outliers [remove_outliers.py](src/features/remove_outliers.py)

This step involves identifying and removing outliers from the accelerometer and gyroscope data using three methods: Interquartile Range (IQR), Chauvenet's Criterion, and Local Outlier Factor (LOF). The goal is to clean the data by marking and removing outliers before proceeding with further analysis or modeling.

Key Actions:

1.  Load and Prepare Data:

- The data is loaded from the processed file, and the participant labels are adjusted.
- The relevant columns for outlier detection are identified.

2. Visualize Outliers:

- Boxplots are used to visualize the distribution of data and identify potential outliers based on the data distribution for different exercises.

3. Outlier Detection Methods:

- Interquartile Range (IQR):

  - A function is implemented to mark outliers based on the IQR method. Outliers are identified as values outside 1.5 times the IQR.
  - These outliers are plotted for each relevant column.

- Chauvenet's Criterion:

  - A function is defined to mark outliers based on Chauvenet's Criterion, which considers the probability of data points under a normal distribution.
  - Histograms are used to check the normality of the data before applying the method.
  - Outliers are plotted similarly to the IQR method.

- Local Outlier Factor (LOF):
  - The LOF algorithm is applied to detect outliers based on the distance of data points from their neighbors.
  - This method is more sophisticated and accounts for the local density of data points.

4. Compare Outliers Across Methods:

- The outliers identified by the IQR and Chauvenet methods are compared for a specific exercise (e.g., bench).
- The effectiveness of each method is visually compared by plotting the results.

5. Outlier Removal:

- The Chauvenet method is chosen for outlier removal.
- A loop is implemented to apply this method across all relevant columns and exercise labels.
- Outliers are replaced with NaN values, and the number of outliers removed is logged.

6. Export Cleaned Data:

- The cleaned dataset, with outliers removed, is saved for further analysis.
- This step ensures that the dataset is free of extreme values that could skew the results in the subsequent steps.

### 5. Low-pass Filter & Principal Component Analysis [build_features.py](src/features/build_features.py)

Overlapping windows can lead to highly correlated data, which can increase the risk of overfitting in machine learning models. To address this, the script reduces the dataset size by approximately 50%, ensuring that only non-overlapping data points are retained.

- Input: The filtered and transformed data (df_freq).
- Process:
  - Drop rows with NaN values using dropna().
  - Reduce the dataset size by selecting every second row (iloc[::2]).
- Output: A reduced dataset (df_freq_reduced).

### 6. Fourier Transform and Clustering [build_features.py](src/features/build_features.py)

The purpose of clustering is to group similar data points together based on selected features, such as accelerometer data. This is done using the K-Means algorithm.

- Input: The reduced dataset (df_freq_reduced).
- Process:
  - Select the features acc_x, acc_y, and acc_z for clustering.
  - Determine the optimal number of clusters (k) by iterating over a range of values (from 2 to 9) and plotting the sum of squared distances (inertia) to find the elbow point.
  - Apply the K-Means algorithm with k=5 to cluster the data points.
  - Plot the clustered data in a 3D space.
    Compare the clusters against labeled data.
- Output: A clustered dataset (df_cluster), with an additional cluster column indicating the cluster each data point belongs to.

#### Final Output

After these steps, the resulting dataset is exported as a pickle file (03_data_features.pkl), which can be used in subsequent steps for further analysis or modeling. The script also removes the duration column before exporting, as it is no longer necessary.

### 7. Predictive Modelling [train_model.py](src/models/train_model.py)

1. Data Preparation
   Data Loading: The script reads the preprocessed dataset from a pickle file.
   Data Splitting: The dataset is split into training and test sets using train_test_split, ensuring stratification by the label.
   Feature Splitting: The script divides features into different sets, including basic features, square features, PCA features, time-domain features, and frequency-domain features.
2. Feature Selection
   Forward Feature Selection: The script uses a decision tree for forward feature selection to find the most important features based on their contribution to model accuracy.
3. Model Training and Evaluation
   Grid Search and Model Selection: It performs grid search and trains various models (neural network, random forest, k-nearest neighbors, decision tree, naive Bayes) on different feature sets. It then evaluates their accuracy on the test set.
   Visualization: The script creates bar plots to compare model performance across different feature sets.
4. Confusion Matrix Visualization
   Confusion Matrix: After selecting the best model (random forest), the script visualizes the confusion matrix to evaluate the performance on test data, first by splitting the data into training and testing sets by participant, and then by using the best model to predict and evaluate results.
5. Complex Model Evaluation
   Feedforward Neural Network: Finally, it evaluates a more complex feedforward neural network model on the selected features, again visualizing the results with a confusion matrix.

### 8. Counting Repetitions [count_repetitions.py](src/features/count_repetitions.py)

1. Load and Process Data
   Loads the processed dataset, filters out the "rest" periods, and calculates the magnitude of the accelerometer (acc_r) and gyroscope (gyr_r) readings.
2. Split Data by Exercise
   Splits the dataset into different exercises like bench press, squats, overhead press (OHP), rows, and deadlifts.
3. Visualize Data
   Visualizes the raw accelerometer and gyroscope data for a specific set to help identify patterns that can be used for rep counting.
4. Configure and Apply Low-Pass Filter
   Configures a low-pass filter to smooth the data and make the peaks (corresponding to repetitions) more apparent.
   Applies the filter to different exercise sets, tweaking parameters like the cutoff frequency and filter order for optimal results.
5. Create a Function to Count Repetitions
   Defines a count_reps function that:
   Applies the low-pass filter to the data.
   Identifies peaks in the filtered data using argrelextrema.
   Plots the filtered data and highlights the detected peaks.
   Returns the count of detected peaks (reps).
6. Create Benchmark DataFrame
   Constructs a rep_df DataFrame that stores the actual number of reps (based on the exercise category) and initializes a column for the predicted reps.
   Loops through each set, applies the count_reps function, and stores the predicted reps in the rep_df DataFrame.
7. Evaluate the Results
   Calculates the mean absolute error (MAE) between the actual and predicted reps.
   Plots a bar chart comparing the mean actual and predicted reps for each exercise category.

## Special Thanks

A special thanks to [@daveebbelaar](https://github.com/daveebbelaar) for providing this machine learning project. Using machine learning on fitness, a core aspect of many people's lives, really took the learning to a new level.
