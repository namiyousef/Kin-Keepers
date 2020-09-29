#!/usr/bin/env python
# coding: utf-8

# # Summary of work done

# The data format has changed a number of times during the course of the project. It's important to extract the following parameters from it:
# 
# data = [date, accX, accY, accZ, gyrX, gyrY, gyrZ]
# 
# date comes in the format 'YYYY-mm-dd HH:MM:SS'
# 
# acceleration ( acc{} ) is a float and has units of $g$, this is typically in the order of $\mathcal{O}(0.1)$
# 
# gyrataion ( gyr{} ) is a float and has units of $^\circ s^{-1}$ (i.e. degrees per second), this is typically in the order of $\mathcal{O}(10)$
# 
# Note that the following notation will be used henceforth:
# 
# Acceleration: $a_i$, where $i$ indicates the degrees of freedom. The Einstein convention is used here, so, the dot product (total acceleration) for example: $a_ia_i = a_1a_1+a_2a_2+a_3a_3$ 
# 
# Gyration: $\omega_i$

# ## Data Transformation
# 
# Initially, the data did not look informative. A couple of diffrerent transformations were tried (see . Though some provided a clear separation in the data (namely: Normal Quantile Transform), the separation was not informative in terms of two distinct clusters, one representing large motion and the other representing noise. The first image below shows what would hopefully be the two clusters... the second shows what 'significant movements' vs. 'noise' roughly is.
# 
# <img style= "height:auto; width:30%;" src = "Images/NormalQuantileClusters.png">
# 
# <img style= "height:auto; width:30%;" src = "Images/NormalQuantileDBSCAN.png">
# 
# More information on this can be seen in KK_Project_3_Modelling

# ## Algorithms used
# 
# The algorithms used for this project can be found as a comment on IOT-111. Note that KKProject_4_Clustering contains the code for this. At the end, it was found that iForest provided the best visual model.

# ## Evaluation
# 
# Testing data generated by Ignacio (IOT-111 under sub-directory individual-movemens) was used to find the best parameters for the iForest algorithm. The movements in question are summarised as a comment on IOT-111. There are a total of 9 movements, each of which is either 'significant' or 'insignifcant'. Note that the significant ones will have a percentage of insignificant movements in them, just by nature of how the data was collected. However, since it is impossible to pinpoint which points are significant and which aren't without imposing a certain bias into the data, 'significant' movement datasets were deemed to contain 100% significant movements. **This is one of the reasons for the accuracy being low.**
# 
# Based on the data, the best model was as follows:
# Isolation Forest with a *contamination* of **0.054**, *max_features* of **1**, *n_estimators* of **100**. Note that larger value for *n_estimators* may actually be more accurate (however it may also overfit the data). 
# This model has an accuracy of **65.5%**.
# 
# It decreases the data by **94.6%**.
# 
# <img src="Images/FinalModel.png">
# 
# ### A couple of pointers:
# 
# 1. Note that the separation creates an L-shape as opposed to a reciprocal graph (which is what you would most likely expect - in fact a previous iteration with a smaller dataset created just that!). It might be worth looking into why this may be: it could just be that more data --> more concentration of points that are stationary, hence the L-shape is more defined. An ideal algorithm would probably create the reciprocal curve.
# 2. The algorithm is not perfect, as on the RHS you can see some points with a very high gyration that are labelled as 'insignificant', when they may actually be
# 
# ### Conclusive remarks
# 
# If you are to repeat this experiment again, ensure that you have:
# 
# a) more accurate test data
# 
# b) more data from many different people (to ensure that the model is robust)
# 
# c) It would be interesting to compare the performance of this model with a supervised learning model (IOT-121)

# # Finding thresholds
# 
# Seeing as the L-shape is *somewhat* representative of a reciprocal curve, it was decided to plot $a_ia_i$ vs. $\frac{1}{\omega_i\omega_i}$ to see if the separation boundary can be modelled using a polynomial.
# 
# The result is shown below:
# <img style="height:auto; width: 50%;" src="Images/Reciprocal.png">
# 
# 
# The model this separating boundary, thanks to the suggestion to Dr Sheehan Olver, it was decided to consider the blue data as a histogram. Having done this, an attempt was made to create a KDE plot over it, and then fit the KDE curve, but this proved futile.
# 
# Instead, the histogram creation function (note: needs some improvement) successfully reduced the number of datapoints to the extent that the bounding curve could be *approximated*. 
# 
# <img src="Images/BoundingCurve.png">
# 
# To approximate this, an arbitrary model of $p(x) = ax^4 + bx^2 + cx + d$ (where $x$ here represents $\frac{1}{\omega_i\omega_i}$) was chosen. One must not assume that this is the best model; it was chosen because it seemed to represent the curve well, and there was little time to spend on finding the best model.
# 
# The datapoints were then fit into the model using an optimiser. The resulting curve is shwon in the plot below (white line).
# 
# <img style="height:auto; width: 60%" src="Images/FinalCurve.png">
# 
# It's worth noting that due to time-constraints, the polynomial model was not tested again against the test data (to compare how it performs against the iForest model).
# 
# The movements from the arduino, $M$, could now be classified as significant (if $M=-1$) or insignificant (if $M=1$)
# 
# Where:
# 
# $$ M =\begin{cases} 
#       -1 & a_ia_i > p\left(\frac{1}{{\omega}{\omega}}\right)\\
#       1 & a_ia_i\leq p\left(\frac{1}{{\omega}{\omega}}\right) \\
#    \end{cases}
# $$
# 
# $$p(x_i) = C_ix_i$$
# 
# $$ C_i = \begin{pmatrix}
# 0.7741697399557282\\
# -0.15839741967042406\\
# 0.09528795099596377\\
# -0.004279871380772796
# \end{pmatrix} \,\, \mathrm{and} \,\,  x_i = \begin{pmatrix}
# x^4\\
# x^2\\
# x\\
# 1
# \end{pmatrix} $$
# 
# 