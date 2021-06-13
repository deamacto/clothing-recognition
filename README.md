# clothing-recognition

### **Introduction**

`clothing-recognition` is a simple machine learning project in which black&white pictures of clothes are clasified to one of ten labels.
Dataset for this project is taken from `zalandoresearch/fashion-mnist` [project](https://github.com/zalandoresearch/fashion-mnist).

### **Methods**

In this project I used Gabor filters for image processing, 8 different filters are applied to each picture. Then all pictures are changed into matrices of 0 and 1 depending on 
how dark the pixels are. Those are then used as classifiers by the naive Bayes algorithm, which assigns probabilities to each classifier for each label based on the training data,
and then assigns the labels to test data based on occurences of those classifiers. The a priori probability is calculated using Beta distribution with two hyperparameters: alpha 
and beta. The naive Bayes algorithm implementation is based on second MSiD task.

### **Results**

The table below compares my naive Bayes results with those found in the benchmarks section of `fashion-mnist`.

| Name | Accuracy | Training time |
| --- | --- | --- |
| Naive Bayes | 0.637 | 0:56:01 |
| GaussianNB | 0.564 | 0:00:10 |

The resulting accuracy is higher than the example from benchmarks, but training time is incomparably longer. This can be caused by "home" implementation, pure python usage, 
lack of multithreading support, ...

### **Usage**

The required data can be downloaded from the `fashion-mnist` repository. It must be put into a data/fashion directory, which must be created first.  
Required libraries:
- numpy
- matplotlib
- scipy

To run start the main.py script. Results will be displayed in console.
