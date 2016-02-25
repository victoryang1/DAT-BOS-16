---
title: Progressing in Your Data Science Career
duration: "3:00"
creator:
    name: Arun Ahuja
    city: NYC
---



# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Lesson Title
Week | Lesson 19

### LEARNING OBJECTIVES
*After this lesson, you will be able to:*

### STUDENT PRE-WORK
*Before this lesson, you should already be able to:*

### INSTRUCTOR PREP
*Before this lesson, instructors will need to:*


### LESSON GUIDE
| TIMING  | TYPE  | TOPIC  |
|:-:|---|---|
| 5 min  | [Opening](#opening)  | Topic description  |
| 60 mins  | [Real-World ML Systems](#systems)   | Real-World ML Systems |
| 20 mins  | [Exercise](#exercise)   | Group Exercise |
| 20 mins  | [Demo/Codealong](#demo)   | Pipelines in scikit-learn |
| 45 mins  | [Alternative Tools](#alternative)   | Alternative Tools  |
| 20 mins  | [Next Steps](#nextsteps)   | Next Steps |
| 10 mins  | [Conclusion](#conclusion)  | Conclusion |

---
<a name="opening"></a>
## Opening (5 mins)
In this class we focus on adapting the skills of this course in real-world problems. We will what is required to maintain and improve analyses and what steps can be taken to make your work 'production' ready. Lastly, we will focus on what other tools exist in the data science ecosystem and additional topics you may want to focus on.

***

<a name="systems"></a>
## ML Systems (60 mins)

### Integrating a Model Into a Data Product

As you move on to real-world data science projects, it's important to remember that models and analysis are typically only a single aspect of a larger goal or business objective. Typically, your model or analysis may only answer one of many questions that need to be addressed.

For example, in a system that will present recommendations, we may have many modeling components that come together. Different aspects may categorize content, extract text features, analyze popularity and which are then tied together into a final data product.

For example, in Hulu's recommendation system, they pull data from multiple sources, build user profiles and summaries, then apply a recommendation model. However, this isn't the final step, additional filters are placed to refine the model to ensure predictions are novel and don't overlap with previous content.

![](assets/images/hulu.jpg)

Organizing and managing the systems and data dependencies becomes and important part of the job.

![](assets/images/ml-systems.jpg)

Many organizations rely on data engineering teams to encode this common tasks into pipelines. **Data pipelines** are a series of automated data transformations that ensure the validity of your work for routine data maintenance tasks. Below is a description of AirBnB model building pipeline.

![](assets/images/airbnb-system.png)

### Model Maintenance

Most of this class has focused on building analysis and models, but once a final model is trained and good predictive performance is achieved that model takes on it's own life and needs to be maintained. As new data is gathered, the model is likely to be re-trained. What were previously highly predictive features may begin to lose their value, requiring you to investigate again.

Google released a paper earlier addressing this phenomenon as the 'Technical Debt' of a machine learning system: [Machine Learning: The High Interest Credit Card of Technical Debt](http://research.google.com/pubs/pub43146.html)

They focused on a few core issues of model maintenance:
- code complexity
- evolving features
- monitoring and testing

#### Code Complexity

Most of the code in this class has been written in notebooks and in a script fashion. However, as your analysis and models develops you are likely to revise and reuse parts of this code. Improving the quality of the code can go along way to ease this process.

While data scientists aren't held to the higher standards, more clarity in the code will often lead to more clarity in the analysis.

A [new Python styleguide (a set of rules for code organization) has been developed for data scientists](http://columbia-applied-data-science.github.io/pages/lowclass-python-style-guide.html)

Some of the rules are pretty straightforward:
- "Give variables, methods, and attributes descriptive names"
- "Write functions that take well-defined inputs and produce well-defined output"

While it won't be covered here, a common practice is software development is unit-testing. Unit-testing involves write short statements that _test_ a piece of code or function you have written. Typically, this test provides a few sample inputs and outputs and ensure your code can produce the same outputs.

In the Google paper, they address what most analysis systems contain which is "glue code"

__"Only a tiny fraction of the code in many machine learning systems is actually doing "machine learning" ... Without care, the resulting system for preparing data in an ML-friendly format may become a jungle of scrapes, joins, and sampling steps, often with intermediate files output. Managing these pipelines, detecting errors and recovering from failures are all difficult and costly"__

__"Ensuring that code has been tested will be vital to ensure that the results of your analysis is correct."__

__"As a real-world anecdote, in a recent cleanup effort of one important machine learning system at Google, it was found possible to rip out tens of thousands of lines of unused experimental codepaths"__

#### Evolving Features

Once your model is trained or analysis is complete, it's important to track it's performance over time. Many of the correlations found or predictive features may not be true, a few months or years in the future. For example, in our evergreen prediction example, which we will revisit today, it captures important food labels to predict recipes. However, it doesn't know about popular food trends of tomorrow. While `brownies` and `cookies` and `flour` are important features, `cronut` is not since the dataset was collected prior to that phenomenon.

Revisiting, Google's technical debt paper, groups these trouble some features into two groups, legacy features and bundled features.


__"Legacy Features: The most common is that a feature F is included in a model early in its development. As time goes on, other features are added that make F mostly or entirely redundant, but this is not detected."__

__"Bundled Features: Sometimes, a group of features is evaluated and found to be beneficial. Because of deadline pressures or similar effects, all the features in the bundle are added to the model together. This form of process can hide features that add little or no value."__

While "bundled features" implies we may add features that are repetitive, the opposite could occur as well. We may remove a variable because it is strongly correlated with an existing feature. This correlation may exist today, but not in a few years and we may want to re-examine that relationship. 

__"Machine learning systems often have a difficult time distinguishing the impact of correlated features. This may not seem like a major problem: if two features are always correlated, but only one is truly causal, it may still seem okay to ascribe credit to both and rely on their observed co-occurrence. However, if the world suddenly stops making these features co-occur, prediction behavior may change significantly."__

This last point is very important for black-box models. These models may rely on correlations from a wide-range of features, however, in doing so we can typically ignore one of two variables that are highly recorded (think of PCA or regularization, where we try to remove correlated features). In the future, if these two variables are no-longer correlated, we may need to update this. This example is common in economics - 

Another way in which features may evolve is through feedback loops. Once you've performed your analysis or built your model, it's likely you are going to take some action or make a business decision. It's important to track how this may change the data you are using for your analysis!

One example, are the many data analysis efforts to find ways to stop infections from spreading in hospital. Suppose to we analyze data related to this and find that whenever a doctor sees more than 5 patients in an hour, those patients are at a greater risk for infection. Following this analysis, the hospital enforces that doctor's _cannot_ observe that many patients in an hour. 

Now, at the end of year, if we attempt to analyze the feature of "saw > 5 patients in an hour", it simply won't exist. Through our intervention we changed the data.

**Check** Brainstorm with the class two correlated features that may not be correlated in the future.

#### Monitoring Models

Once a model is deployed and making predictions, it's important to track it's performance. This will also help diagnose features that are evolving.

__"Unit testing of individual components and end-to-end tests of running systems are valuable, but in the face of a changing world such tests are not sufficient to provide evidence that a system is working as intended. Live monitoring of system behavior in real time is critical."__

One often used monitoring technique is to always have a baseline. When you are evaluating your model, it is important to always compare to a simple baseline - something that predicts the average or the most commonly occurring thing. While monitoring a model over time, you can also evaluate this baseline model over time to ensure that your 'better' model never underperforms it.

In Google's paper [Ad Click Prediction: A View From The Trenches](http://research.google.com/pubs/pub41159.html), they describe a more complex monitoring systems, that analyzes how model performance changes and in what subcategories.

__"Evaluating the quality of our models is done most cheaply through the use of logged historical data ... Because the different metrics respond in different ways to model changes, we find that it is generally useful to evaluate model changes across a plurality of possible performance metrics. We compute metrics such as AucLoss (that is, 1 − AUC, where AUC is the standard area under the ROC and SquaredError ... We also compute these metrics on a variety of sub-slices of the data, such as breakdowns by country, query topic, and layout. ... This is important because click rates vary from country to country and from query to query, and therefore the averages change over the course of a single day. ... A small aggregate accuracy win on one metric may in fact be caused by a mix of positive and negative changes in distinct countries, or for particular query topics. This makes it critical to provide performance metrics not only on the aggregate data, but also on various slicings of the data, such as a per-country basis or a per-topic basis."__


### Ethical Considerations

Another, often overlooked, aspect of managing real-world data science projects are ethical considerations. A core aspect of any data science project is understanding the biases of the data we are studying. Data often represent the summary of a system and any biases of the system will appear in the data and any analysis built from it.

Two common examples of this are in sentencing and financial loan applications. In the first, it's common to want to find data-driven solution in criminal justice. Can we analyze what drives crime in certain cities and what actions we can take to reduce it? However, most data that is collected on this is based on how current criminal justice system works. Therefore it's difficult to separate the biases of the current system from any correlations/predictions that may be found in the model.

Similarly, in most financial loan applications, we are interested analyzing applications to identify the best borrowers - who is most likely to pay back in a timely fashion and what rate? However, most analyses are strongly regulated to ensure that they do not consider protected factors such as race and gender. However, often this information can leak in, even if we don't mean for it. This can happen with proxy features, those that are not protected factors but strongly correlated with it. This type of bias can happen often in home loan applications when zip code or neighborhood can be a proxy for race.

**Check:** Discuss with the students other areas of possible ethical issues in Data Science? How can this occur when examining health data? When examining educational records?

<a name="exercise"></a>
## Group Exercise: Data Science in an Organization  (20 mins)
- Break up the class into groups of ~4/5 students each
- For each group assign a company and 1-2 data products or analyses the company must perform
- Have the students brainstorm aspects of maintenance that must be performed. When should they re-do the study? What features may change or difficult to collect in the future?
- Have the students describe possible interventions. Will this change the data collected in the future?
- Have the student describe ethical issues that may arise in their task.


Example:
Company: HBO
Task: Build a customized home screen with recommended shows

1. Maintenance: Popular movies are consistently changing and new movies/shows are added, so the model must be consistently updated. Must be a way to track unexpected events: possibly override recommendations before re-releases or sequel releases.
2. Intervention: Displaying the movies (somewhat a given here). Recommendation systems are where feedback issues are most common: movies not recommended are more likely to fall to the bottom in viewership, making them seem less appealing. Users may not like recommendations and stop using the service altogether.
3. Ethical considerations: Should gender be used in the recommendations? Should the descriptions or images of the movies or shows be altered based on 'perceived' interests of a specific gender?

Example:
Company: Credit Card Company
Task: Identify fraudulent transactions

1. Maintenance: User spending likely changes as their income increases. Expensive purchases that may have been expensive (and fraudulent) in previous year may not be this year. 
2. Intervention: Block fraud transactions. Another common case of feedback is when the intervention is adversarial. As you find ways to block fraud transactions, they are try to hide their actions better. Suppose we identify 10 common IPs used for fraudulent transaction, if we block those IPs, the fraudsters will likely change.
3. Ethical considerations: Should we immediately block transactions when we believe they are fraudulent or let them occur and review later? Is it more important to have many false positive (incorrectly blocked purchases) or false negatives (missed fraudulent transactions)


<a name="demo"></a>
## Demo/Codealong: Pipelines in scikit-learn (30 mins)
One way to improve coding and model management is to use pipelines in `scikit-learn`. These tie to together all the steps that you may need to prepare your dataset and make your predictions. Because you will need to perform all of the exact same transformations on your evaluation data, encoding the exact steps is important.

```python
from sklearn.pipeline import Pipeline
```

Previously, we looked at building a text classification model using `CountVectorizer`

```python
import pandas as pd
import json

data = pd.read_csv("../../assets/dataset/stumbleupon.tsv", sep='\t')
data['title'] = data.boilerplate.map(lambda x: json.loads(x).get('title', ''))
data['body'] = data.boilerplate.map(lambda x: json.loads(x).get('body', ''))

titles = data['title'].fillna('')

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features = 1000, 
                             ngram_range=(1, 2), 
                             stop_words='english',
                             binary=True)

# Use `fit` to learn the vocabulary of the titles
vectorizer.fit(titles)

# Use `tranform` to generate the sample X word matrix - one column per feature (word or n-grams)
X = vectorizer.transform(titles)
```

We used this input X, matrix of all commong n-grams in the dataset, as input to a classifier. We wanted to classify how evergreen a story was based on these inputs.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty = 'l1')
y = data['label']

from sklearn.cross_validation import cross_val_score

scores = cross_val_score(model, X, y, scoring='roc_auc')
print('CV AUC {}, Average AUC {}'.format(scores, scores.mean()))
```

Often we will want to combine these steps to evaluate on some future dataset. For that incoming, future dataset, we need to make sure we perform the exact same transformations on the data. If has_brownies_in_text is column 19, we need to make sure it is column 19 when it comes to evaluation time.
Pipelines combine all of the pre-processing steps and model building into a single object.

Rather than manually evaluating the transformers and then feeding them into the models, pipelines tie these steps together. Similar to models and vectorizers in scikit-learn, they are equipped with `fit` and `predict` or `predict_proba` methods as any model would be, but they ensure the proper data transformations are performed

```python
# Split the data into a training set
training_data = data[:6000]
X_train = training_data['title'].fillna('')
y_train = training_data['label']

# These rows are rows obtained in the future, unavailable at training time
X_new = data[6000:]['title'].fillna('')

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
        ('features', vectorizer),
        ('model', model)   
    ])

# Fit the full pipeline
# This means we perform the steps laid out above
# First we fit the vectorizer, 
# and then feed the output of that into the fit function of the model

pipeline.fit(X_train, y_train)

# Here again we apply the full pipeline for predictions
# The text is transformed automatically to match the features from the pipeline
pipeline.predict_proba(X_new)
```

**Check** Add a MaxAbsScaler scaling step to the pipeline as well, this should occur after the vectorization

Additionally, we want to merge many different feature sets automatically, we can use scikit-learn's `FeatureUnion`.

While scikit-learn pipelines help with managing the data transformation from raw data to input into the model, there may be many steps before this in your data pipeline as well. These pipelines are often referred to as ETL pipelines for Extract, Transform, Load. The data is pulled or extracted from some source (say a database), transformed or manipulated to the relevant information and then used (or "loaded") into whatever system or analysis requires them.

Many data science team rely on software tools to manage these ETL pipelines. If a transformation step fails, these tools alert you, or ensure that step can be re-run. If these transformation steps need to happen daily or weekly, these tools manage that timeline.

One of the most popular Python tools for this is [Luigi](https://github.com/spotify/luigi) developed by Spotify.

An alternative is [Airflow](https://github.com/airbnb/airflow) by AirBnB.

<a name=""></a>
## Alternative Tools (45 mins)

While most of this class has focused on data science and analytics in Python there are many tools that can be used and appear often in data science roles that offer slightly different advantages and disadvantages.

### Languages
Other common languages for data science include:
- R
- Java/Scala

R is often used in data science and many of features of Python data analysis are borrowed from R. The Pandas dataframe replicates functionality of the R dataframe.

R often contains many more specialized algorithms as well. While `statsmodels` and `scikit-learn` contain most of the most popular statistical algorithms, as your problems become more specialized, you may require a more specialized tool. Typically, R has a wider variety of niche algorithms.

However, Python's advantages over R are in it's speed and it's ability to tie into other applications. R has many specialized and fast dataframe operations, but Python code can be much faster and more efficient. Also, using Python allows you to connect your analysis with other tools in Python, for example web development.

Java/Scala are popular for their link to the Hadoop ecosystem. Many larger organizations store their data in a Hadoop system and most of the adapters to move data in and out these systems are built in Java and Scala. Therefore, it is sometimes easier to interact with these systems in those languages. However, this languages lack the interactivity and easy of use that R and Python provide.

### Modeling Frameworks
While `scikit-learn` in the most popular machine learning framework in Python, there are alternatives for specialized usecases.

For example, most models in `scikit-learn` require datasets to be small enough to fit into memory. Other frameworks can work around this. One example is `xgboost` which provides efficient Random Forest implementations that may train much faster than the models in `scikit-learn`.

Similarly, a library `Vowpal Wabbit` is often used to train very large linear models, very fast This library is developed in C++, with a Python wrapper, that includes many computational tricks to operate on millions to tens of millions of datapoints.

***

<a name="nextsteps"></a>
## Next Steps (20 mins)

The core of this class has focused on statistical knowledge, and supervised and unsupervised learning. For each of these topics, there are many alternative methods to learn.

### Statistical Testing
While it is often not important to have an encyclopedic knowledge of all the possible statistical tests, it may be worth remembering a few common ones and the assumptions they make.

Additionally, have a clear sense of distributions and what they look like is important. Being able to view a histogram and summarize it by a distribution it resembles makes it easy to discuss the data.

![](assets/images/distribution.png)

[](http://blog.cloudera.com/blog/2015/12/common-probability-distributions-the-data-scientists-crib-sheet/)

### Visualization
While most of the plotting in this class was done in Python, these plots are often not the most visual appealing. Many tools exist to build plots, but a fair amount visual design knowledge may be needed to ensure the best product. Visualizing and presenting data is often the best way to transfer information from your work to the business and much more likely to be effective than an array of numbers.

To make plots interactive, tools like plot.ly or D3.js may be used. The latter is a Javascript library to construct web-based interactive plots.

[D3 Gallery](https://github.com/mbostock/d3/wiki/Gallery)

### Model Interpretability vs Accuracy
One of the constant trade-offs in data modeling is whether we are more interested in high predictive accuracy or a high degree of interpretability. We saw that linear models are simple, can perform well and offer a concise summary of the impact of various features through the coefficients. However, black-box models such as Random Forests may perform much better (in terms of predictive accuracy) without as much transparency. In real-world scenarios, you likely care more about interpretability and insight and prefer simpler models. Logistic and Linear Regression though simple are by far the most used tools and important to know well. 

When going further in data science you will see this contrast return again and again. Two methods of advanced analysis perfectly capture this divide.

_Bayesian data analysis_ is a method of analysis that requires you first write down your expectations about the interactions in your world and then attempt to learn how strong those interactions are.

For example, suppose you are analyzing the roll-out of a new educational policy. We want to measure the impact of this policy on some outcome (test scores). Similar to our current models, we need to know what else will impact test scores and build a model to learn the impact of this policy on test scores. However, we may want to force additional constraints. For example, we may want to say that we think the policy will have a related but different effect in different states. We can further write down subgroups where the effect may be different because of different reasons (different local resources, demographics, budgets, etc.) and explicitly state how these aspects related to further constrain our model.

These types of models are typically small and their main strengths are in interpretability and capturing the amount of uncertainty exists in your data. Rather than stating that X will change Y by some amount, these models give a _distribution_ or range of possible amounts and attempts to tell what will happen in the best or worst case.

These models are useful when interpretability and precisely defining interactions are of utmost importance, particularly, if we want a clear definition how right or wrong we are. We may want to say that candidate X is likely to win the election, but if we want to quantify the degree of uncertainty we have Bayesian models can be useful.

There are a few tools to build these types of models in Python, one of which is [pymc](http://pymc-devs.github.io/pymc3/). A good reference on this is [Bayesian Methods for Hackers](http://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/)

On the other side of spectrum, powerful models that offer little to no interpretable value are _deep learning_ models.

These models, also known as _neural networks_, are very powerful predictive models, however, they are complex to build and offer little to extract what patterns they have learned.

Put simply, these models attempt to operate in a staged fashion. First they perform a dimensionality reduction to extract pattern or representations of the input data. Then these representations are used for the predictive task. This is very similar to a model we saw in this class, which was using a dimensionality reduction technique followed by a classification technique. Deep learning models tie these two steps together, attempting to learn the best representation for the task. Additionally, they include many non-linear operations to capture more complex relationships in the data.

Deep learning models are particularly well suited for image or audio analysis.

Python has developed a strong collection of well-written deep learning libraries, which include:
- [Keras](http://keras.io/)
- [lasagne](http://lasagne.readthedocs.org/en/latest/)
- [Tensorflow](https://www.tensorflow.org/)


***

<a name="conclusion"></a>
## Conclusion (10 mins)
- Data science results are often incorporated into a larger final product
- Tracking that final product means maintaining models and data pipelines, understanding the possible changes overtime and managing any ethical considerations
- Alternative common languages for data science are R or Java/Scala, but Python has many advantages
- Visualization skills are vital to improve and moving beyond Python may be useful
- Advanced machine learning methods to explore include Bayesian methods and/or deep learning


***

### BEFORE NEXT CLASS
|   |   |
|---|---|


### ADDITIONAL RESOURCES
- [Building Data Pipelines with Python and Luigi](http://marcobonzanini.com/2015/10/24/building-data-pipelines-with-python-and-luigi/)
- [Data Science at Instacart](https://tech.instacart.com/data-science-at-instacart/)
- [Striking a Balance Between Interpretability and Accuracy](https://www.oreilly.com/ideas/predictive-modeling-striking-a-balance-between-accuracy-and-interpretability)
