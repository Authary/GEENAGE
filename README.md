# GEENAGE
Python module for selecting important features in a supervised classification dataset and interpreting them in terms of prediction and discrimination.

Functionalities:<br/>
-Identify the features that are the most *predictive* and *discriminant* of a class<br/>
-Identify the features that are the *causes* and *effets* of a class

This module uses pyCausalFS (https://github.com/wt-hu/pyCausalFS) for the computation of causes and effects.

# LEGAL STUFF

You are free to use this module.<br/>
If you do something interesting with it, please tell Alexandre Bazin: contact [at] alexandrebazin [dot] com.

# INPUT FORMAT

At the moment, the software only accepts numerical datasets. The data should take the form of a numpy.array in which the first row contains the names of the features and the other rows are numerical values. The first column should be the class.<br/>
Example:
| "Diabetes" |	"Glucose" |	"Age" |	"SkinThickness" |	 "BloodPressure" | "Insulin" |
|---| ---| ------|--------|------|------| 
|0|	0.27  |	0.65 | 0.77 | 0.03 | 0.12	|
|1| 0.95	|	0.58 | 0.34 |	0.12 | 0.27	|
|1| 0.03	|	0.12 | 0.80	|	0.65 | 0.58 |
|1| 0.58	| 0.97 | 1 | 0.17	| 0.58 |
|0| 0.62	| 0.11 | 0.99 | 0.80 | 0.65 |

# USEFUL FUNCTIONS

*preDisc(data,file = None,depth = 1, n_perm = 200)*<br/>
INPUT: the data and, optionally, the name of a file in which to write the output, a depth (more depth equals more features) and a number of permutations for the computation of features' importance<br/>
OUTPUT: a 2D list of terms describing the importance of the selected features in terms of prediction and discrimination and the list of selected features' names


*causEffects(data,file = None, alpha = 0.1)*<br/>
INPUT: the data and, optionally, the name of a file in which to write the output and alpha, the p-alue threshold for conditional independence tests (higher alpha equals more features)<br/>
OUTPUT: a 2D list of terms describing the importance of the selected features in terms of causes and effects and the list of selected features' names


*loadData(file)*<br/>
INPUT: a filename<br/>
OUTPUT: the data in an array



# HOW TO USE

```python
data = loadData("mydata.txt")
preDisc(data,file = "output.txt",depth = 2)
```

Result:

Glucose : Discriminant Predictive_neg Predictive_pos<br/>
Pregnancies : Predictive_neg Discriminant<br/>
BloodPressure : Predictive_neg Discriminant<br/>
SkinThickness : Predictive_pos<br/>
Insulin : Discriminant Predictive_pos Predictive_neg<br/>
BMI : Discriminant Predictive_pos Predictive_neg<br/>
DiabetesPedigreeFunction : Discriminant Predictive_neg<br/>
Age : Predictive_neg Discriminant
