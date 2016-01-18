---
title: Timeseries Modeling
duration: "3:00"
creator:
    name: Arun Ahuja
    city: NYC
---

# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) 
Week # | Lesson 16

### LEARNING OBJECTIVES
*After this lesson, you will be able to:*
- Model and predict from time series data using AR, ARMA or ARIMA models
- Specifically, coding those models in `statsmodels`

### STUDENT PRE-WORK
*Before this lesson, you should already be able to:*
- Prior lesson in time-series including moving averages and autocorrelation
- Prior exposure to linear regression with discussion of coefficients and residuals
- `pip install statsmodels` (should be included with Anaconda)

### INSTRUCTOR PREP
*Before this lesson, instructors will need to:*
- `pip install statsmodels` (should be included with Anaconda)


### LESSON GUIDE
| TIMING  | TYPE  | TOPIC  |
|:-:|---|---|
| 5 min  | [Opening](#opening)  | Lesson Objectives  |
| 60 min  | [Introduction](#introduction)   | Intro: Timeseries Models |
| 45 min  | [Demo/Codealong](#demo1)  | Demo/Codealong: Timeseries Models in Statsmodels  |
| 15 min  | Break
| 50 min  | [Independent Practice](#ind-practice)  | Walmart Sales Data: Timeseries Modeling Exercise |
| 5 min  | [Conclusion](#conclusion)  |   |

---
<a name="opening"></a>
## Opening (5 min)

In the last class we focused on exploring time-series data and common statistics for time-series analysis. In this class, we will advance those techniques to show how to predict or forecast forward from time series data. If we have a sequence of values (a time series), we will use the techniques in this class to predict the a future value. For example, we may want to predict the number of sales in a future month. 

<a name="introduction"></a>
## Intro: What are (is) time series models? (60 mins)

Time series models are models that will be used to predict a future value in the time-series. Like other predictive models, we will use prior history to predict the future. Unlike previous models we will use the previous _outcome_ variables as _inputs_ for prediction.

As with previous modeling exercises we will have to evaluate different types of models to ensure we have chosen the best one. We will want to evaluate on held-out set or test data to ensure our model performs well on unseen data. Unlike previous modeling exercises, we won't be able to use cross-validation for evaluation. Since there is a time component to our data, we cannot choose training and test examples at random. We will typically train on values earlier (in time) in our data and test our values at the end of data period.

### Properties for time-series prediction
In our last class we saw a few statistics for analyzing time series. 

We looked a moving averages to evaluate the local behavior of the time series. A _moving average_ is an average of _k_ surrounding data points in time.

 ![](../lesson-16/assets/images/single_moving_avg_fit.gif)

We looked at autocorrelation to compute the relationship of the data with with prior values.

_Autocorrelation_ is how correlated a variable is with itself. Specifically, how related are variables earlier in time with variables later in time.

![](../lesson-15/assets/images/autocorrelation.gif)

We fix a _lag_, k, which is how many timepoints earlier should we use to compute the correlation. 

We can use to these values to assess how we plan to model our time-series. Typically, for a high-quality model we require some autocorrelation in our data. We can compute autocorrelation at various lag values to determine how far back in time we need to go.

Additionally, many models make an assumption of _stationarity_, assuming the mean and variance of our values is the same through out. This means that while the values (of sales, for example) may shift up and down over time, the mean value of sales is constant as well as the variance (there aren't many dramatic swings up or down). As always, these assumptions may not reasonable real-world data, but we must be aware of when we are breaking the assumptions of our model.

Often, if these assumptions don't hold we can alter our data to make them true. Two common methods are _detrending_ and _differencing_

_Detrending_ would mean to remove any major trends in our data. We could do this in many ways, but the simplest is to fit a line to the trend, and make a new series that is the difference between the line and the true series.

For example in the iphone google searches, there is a clear upward trend. If we fit a line to this data first, we can create a new series that is the difference between the true number of searches and the predicted searches. We can then fit a time-series model to this difference.

![](../lesson-16/assets/images/google-iphone.png)

A simpler but related method is _differencing_. This is very closely related to the `diff` function we saw in the last class. Instead of predicting the series (again our non-stationary series) we can predict the difference between two consecutive values. We will see that the ARIMA model incorporates this directly.

#### AR Models

Autoregressive (AR) models are those are use data from previous timepoints to predict the next. This are very similar to previous regression models, except as input we take the some previous outcome. If we are attempting to predict weekly sales, we use the sales from a previous week as input. Typically, AR models are noted AR(p), where _p_ indicates the number of previous time points to incorporate, with AR(1) being the most common. 

In an autoregressive model, similar to standard regression, we are learn regression coefficients, where the inputs or features are the previous _p_ values. Therefore, we will learn _p_ coefficients or \beta values.

If we have a time series of sales per week, \y_i, we can regress each y_i from the last _p_ values.

y_i = \intercept + \beta_1 * y_(i-1) + \beta_2 * y_(i-2) + ... + \beta_p * y_(i-p) + random_error

As with standard regression, our model assumes that each outcome variable is a linear combination of the inputs and a random error term.  For an AR(1) models we will learn a single coefficient. This coefficient will tell us the relationship between the previous value and the next. A value > 1 would indicate a growth over previous values, while a value > 0  and < 1 would mean we are decreasing over time.

Recall, _autocorrelation_ is the correlation of a value with itself. We compute correlation with values _lagged_ behind. A model with high-correlation implies that the data is highly dependent on previous values and an autoregressive model would perform well.

Autoregressive models are useful for learning falls or rises in our series. This will weight together the last few values to make a future prediction. Typically, this model type is useful for small-scale trends such as an increase in demand or change in tastes that will gradually increase or decrease the series.

#### Moving Average Models

_Moving average models_, as opposed to autoregressive models, do not take the previous outputs (or values) as inputs, but, the previous error terms. We will attempt to predict the next value based on the overall average and how off our previous predictions were.

This models is useful to handling specific or abrupt changes in a system. If we consider that autoregressive models are slowly incorporating changes in the system by combining previous values, moving average models are using our prior errors. Using these as inputs helps model sudden changes by directly incorporating the prior error. This is useful for modeling a sudden occurrence - something going out of stock effecting sales or a sudden sale or rise in popularity.

As in autoregressive models, we have an order term, _q_, and we refer to our model as _MA(q)_.  This moving average model is dependent on the last _q_ errors.

If we have a time series of sales per week, \y_i, we can regress each \y_i from the last _q_ error terms.

    y_i = \mean + \beta_1 * \error_i + ... \beta_q * \error_q

We include the \mean of the time series and that is why we call this a moving average, as we assume the model takes the mean value of series and randomly jumps around it.

In this model, we learn _q_ coefficients. In an MA(1) model, we learn one coefficient where this value indicates the impact of how our previous error impacts our next prediction.

#### ARMA Models

_ARMA_, pronounced 'R-mah', models combine the autoregressive models and moving averages. For an ARMA model, we specify two model settings `p` and `q`, which correspond to combining an AR(p) model with an MA(q) model.

An ARMA(p, q) model is simply a combination (sum) of an AR(p) and MA(q) model.

Incorporating both models allows us to mix two types of effects. Autoregressive models slowly incorporate changes in preferences, tastes and patterns. Moving average models base their prediction not on the prior value but the prior error, allowing to correct sudden changes based on random events - supply, popularity spikes, etc.

#### ARIMA Models

_ARIMA_, pronounced 'uh-ri-mah', is an AutoRegressive Integrated Moving Average model.

In this model, we learn an ARMA(p, q) to predict not the value of the series, but the difference of the two series.

Recall the pandas `diff` function. This computes the difference between two consecutive values. In an ARIMA model, we attempt to predict this difference instead of the actual values.


                        \y_t - \y_(t-1) = ARMA(p, q)

An ARIMA models has three settings and is specified ARIMA(p, d, q), _p_, is the order of the autoregressive component, _q_, is the order of the moving average component and _d_ is the degree of differencing.  In the above, we set _d = 1_ .

For a higher value of _d_, for example, d=2, the model would be:

                         diff(diff(y)) = ARMA(p, q)

We would apply the `diff` function _d_ times.

Compared to an ARMA model, ARIMA models do not rely on the underlying series being stationary. The differencing operation can _convert_ the series to one that is stationary. Instead of attempting to predict the values over time, our new series in the difference in values over time. 

Since ARIMA models automatically include differencing, we can use this on a broader set of data without assumptions of a constant mean.

**Check:** 

<a name="demo1"></a>
## Demo: Modeling in time series in statsmodels (45 mins)

To explore time series models, we will continue with the Rossmann sales data.  This dataset has sales data for sales at every Rossmann store for a 3-year period as well indicators of holidays and basic store information.

In the last class, we saw that we would plot the sales data at a particular store, as well compute the autocorrelation for the data at varying lag periods.

```python
import pandas as pd

# Load the data and set the DateTime index
data = pd.read_csv('lessons/lesson-15/assets/data/rossmann.csv', skipinitialspace=True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Filter to Store 1
store1_data = data[data.Store == 1]

# Filter to open days
store1_open_data = store1_data[store1_data.Open==1]

# Plot the sales over time
store1_open_data[['Sales']].plot()
```

In this class, we will use `statsmodels` to code AR, MA, ARMA and ARIMA models. 

`statsmodels` is a machine learning package, similar to `sckit-learn`.  While it lacks many of the features of `scikit-learn` for evaluation and production level models, it does include many more niche statistical models, including time series models. It also provides a nice summary utility to help diagnose models.

The time series models are in the `ts` package. Let's start by investigating AR models.

### AR, MA and ARMA models in Statsmodels

To explore AR and ARMA models, we will use `sm.tsa.ARMA`. Remember, an ARMA model is a combination of autoregressive and moving average models.

We can train an autoregressive model by turning off the moving average component.

```python
store1_sales_data = store1_open_data[['Sales']].astype(float)
model = sm.tsa.ARMA(store1_sales_data, (1, 0)).fit()
model.summary()
```

By passing the `(1, 0)` in the second argument, we learn an AR(1), MA(0), ARMA models (alternatively, a ARMA(1,0)) model, which is the same as an AR(1) model.

In this AR(1) model we lean an intercept value, or base sales values. Additionally, we learn a coefficient that tells how to include the last sales values. This case, we take the intercept of ~4700 and add in the previous months sales * 0.68.

We can learn an AR(2) model, which regresses each sales value on the last two, with the following.

```python
model = sm.tsa.ARMA(store1_sales_data, (2, 0)).fit()
model.summary()
```

In this case we learn two coefficients, which tell us the effect on the last two sales values on the current sales. To make a sales prediction for a future month, we would combine the last two months of sales with the weights or coefficients learned.

While this model may be able to better model the series, it may be more difficult to interpret.

To start to diagnose the model, we want to look at the residuals. Recall, that the residuals are the errors of the model. What we ideally want are randomly distributed errors that are fairly small. If the errors are large clearly that would be problematic. If the errors have a pattern, particularly over time, that we have overlooked some thing in the model or certain periods of time are different than the rest of the dataset.

We can plot the residuals as below:

```python
%matplotlib inline

model.resid.plot()
```

Here we saw large spikes at the end of each year, indicating that our model does not account for the holiday spikes. Of course, our models are only related to the last few values in the time series, not taking in to account the longer seasonal pattern.

We can include the moving average component as well.

```python
model = sm.tsa.ARMA(store1_sales_data, (2, 2)).fit()
model.summary()
```

Now we learn four coefficients, two additional one for the moving average components.


**Check:** Have the students interpret and explain the coefficients

### ARIMA models in Statsmodels

To train an ARIMA model in `statsmodels`, we can change the `ARMA` model to `ARIMA` and additional providing the differencing parameter. To start, we can see that we can train an ARMA(2,2) model by training an ARIMA(2, 0, 2) model.

```python

model = sm.tsa.ARIMA(store1_sales_data, (2, 0, 2)).fit()
model.summary()

```

We can see that this model in fact simplifies automatically to an ARMA model.  If we change the differencing parameter to 1, we train an ARIMA(2, 1, 2). This predicts the difference of the series.

```python
model = sm.tsa.ARIMA(store1_sales_data, (2, 1, 2)).fit()
model.summary()
```

From our models, we can also plot future predictions and compare them with the true series.

To compare our forecast with the true values, we can use the `plot_predict` function.

We can compare the last 50 days of true values and predictions as values:
```python
model.plot_predict(0, 50)
```

This function takes two arguments which are the start and end index of the dataframe to plot. Here, we are plotting the last 50 values.

To plot earlier values, with our predictions extended out, we do the following. This plots true values in 2014, and our predictions 200 days out from 2014.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax = store1_sales_data['2014'].plot(ax=ax)

fig = model.plot_predict(0, 200, ax=ax, plot_insample=False)
```

**Check:** Have the students alter the time period of predictions and p, d, q parameters

<a name="ind-practice"></a>
## Practice: Walmart Sales Data: Timeseries Modeling Exercise (50 mins)

For the independent practice, we will analyze the weekly sales data from Walmart over a two year period from 2010 to 2012.

The data is again separated by store and by department, but we will focus on analyzing one store for simplicity.

To setup the data

```python
import pandas as pd
import numpy as np

%matplotlib inline

data = pd.read_csv('lessons/lesson-16/assets/data/train.csv')
data.set_index('Date', inplace=True)
data.head()
```

1. Filter the dataframe to Store 1 sales and aggregate over departments to compute the total sales per store.
1. Plot the rolling_mean for `Weekly_Sales`. What general trends do you observe?
1. Compute the 1, 2, 52 autocorrelations for `Weekly_Sales` and/or create an autocorrelation plot.
1. Split the weekly sales data in a training and test set - using 75% of the data for training
1. Create an AR(1) model on the training data and compute the mean absolute error of the predictions.
1. Plot the residuals - where are their significant errors.
1. Compute and AR(2) model and an ARMA(2, 2) model - does this improve your mean absolute error on the held out set.
1. Finally, compute an ARIMA model to improve your prediction error - iterate on the p, q, and parameters comparing the model's performance.

<a name="conclusion"></a>
## Conclusion (5 mins)
- Time-series models use previous values to predict future values, also known as forecasting.
- AR and MA model are simple models on previous values or previous errors respectively.
- ARMA combines these two types of models to account for both local shifts (due to AR models) and abrupt changes (MA models)
- ARIMA models train ARMA models on differenced data to account
- Note that none of this models may perform well for data that has more random variation - for example, for something like iphone sales (or searches) which may be 'bursty', with short periods of increases. 

***

### BEFORE NEXT CLASS
|   |   |
|---|---|

### ADDITIONAL RESOURCES
- [ARIMA model overview](https://www.quantstart.com/articles/Autoregressive-Integrated-Moving-Average-ARIMA-p-d-q-Models-for-Time-Series-Analysis)
- [Time Series Analysis in Python with statsmodels](http://conference.scipy.org/proceedings/scipy2011/pdfs/statsmodels.pdf)
- [First Place Entry in Walmart Sales Prediction](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/forums/t/8125/first-place-entry)
- [Google Search Terms predict market movements](https://www.quantopian.com/posts/google-search-terms-predict-market-movements)
