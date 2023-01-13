Name: Grant Yap
Email: gyap@u.rochester.edu

Method:
My autofit method took in the original first set of numbers from the csv file I named 'x' and the original
second set of numbers from the given csv file I named 'y', the number of folds for the k-fold cross validation, 
and the gamma. I belive my folds parameter was custom to determine the number of ways to divide our testing
data to determine the polynomial order with the best fit for our data.

Regularization:
I tried values 0.001 and 0.01 on higher order polynomials for smaller answers such as sets 'B' and 'C'. The value
that seemed to work the best for higher orders was 0.01. For example, when I used a polynomial order of 30 on
set 'D' with a true polynomial order of 8 with gamma being 0.01 the curve actually fit well because of the smoothing
from the regularization and penalty. But when gamma was set to 0, there was significant overfitting with the graph and curve.

Model Stability:
Set A:
Model Vector:  [-1.1701722   1.57120287]
Order:  1
RMSE:  0.11395248726471283

Set A (Autofit):
Model Vector:  [-1.1701722   1.57120287]
Order:  1
RMSE:  0.11981271825791517

Set B:
Model Vector:  [-0.76713781  0.54344342 -1.55757638]
Order:  2
RMSE:  0.09564289916299712

Set B (Autofit):
Model Vector:  [-0.76713781  0.54344342 -1.55757638]
Order:  2
RMSE:  0.09334496543565887

Set C:
Model Vector:  [-1.52486736  1.57350698  0.01159637  2.31428704 -0.44453377]
Order:  4
RMSE:  0.1033590424797149

Set C (Autofit):
Model Vector:  [-1.52486736  1.57350698  0.01159637  2.31428704 -0.44453377]
Order:  4
RMSE:  0.11571635109859424

Set D:
Model Vector:  [-1.41863341  0.62888188 -1.2055595  -0.45423599  1.73419695  0.43304265
 -0.37693555 -0.98253284 -0.39675167]
Order:  8
RMSE:  0.09111738809560468

Set D (Autofit):
Model Vector:  [-1.41618137  0.54008575 -1.29112884  0.31855615  2.19016023 -1.21009445
 -1.14221517]
Order:  6
RMSE:  0.10726164139913916

Set E:
Model Vector:  [  -0.31688315    1.11743849   -2.85872238   -5.14525384   22.7615498
   21.01792836 -112.74145748  -49.70431832  238.13954464   51.32213498
 -229.84995291  -16.68837225   82.11039284]
Order:  12
RMSE:  0.09247479527998315

Set E (Autofit):
Model Vector:  [-0.33968562  0.7799859  -1.06433488  0.68816333 -0.75222793 -7.57024805
 -0.66576923  8.00479884]
Order:  7
RMSE:  0.13725067887713194

My results are usually off by a few decimal places for the first few numbers and off even
more for later numbers on higher order polynomials. But on the first few sets, they remain
fairly consistent.

Results:
For my autofit method, it works fairly consistently for the first three sets (A, B, C). However,
on the fourth and fifth sets they perform rather poorly. For D, the results tend to range from
6-7 ordered polynomials and for E, the results tend to stick to 7-9 ordered polynomials. 

For the second set on X I usually get the ordered polynomial 6. 
For Y, I tend to get the ordered polynomial 7.
For Z, I tend to get the ordered polynomial 3.

I do believe these results are likely to be correct because of the larger data set to test on,
lower RMSE values, and consistent results.

Collaboration:
I worked solo on this specific project, but did talk a lot about ideas for the project with several students
in the class to help to come up with k-fold cross validation as a method for autofit.

Notes:
I included several extra flags to help with getting unique results such as 

--plotting -> to plot the data to see how the curve overfits or tries to 
fit with the data at a higher regularization

--folds -> change the number of folds for k-fold cross validation

--trials -> change the number of trials to loop through autofit to get a more consistent result
on smaller data sets.

A common way to run this program for me as I was on version 3.10 of python was:
(file name and interpreter call) (m) (gamma) (trainpath) (modeloutput) (--autofit) (--info) (additional flags)
"python3.10 .\polyhunt.py 12 0 "SecondSet/Z" "modelOutput.txt" --autofit"

Another note is that my info flag prints my name and email to stdout instead of the modeloutput file.

Thanks for reading,
Grant
