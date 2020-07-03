#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""

Chicago crimes - crime type classifier.

"""

import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, IndexToString

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from sklearn.metrics import confusion_matrix

from pathlib import Path


# %%

findspark.init()

spark = SparkSession.builder.appName('ChicagoCrimes').getOrCreate()

# %%

files = ['Chicago_Crimes_2005_to_2007.csv', 'Chicago_Crimes_2008_to_2011.csv', 
         'Chicago_Crimes_2012_to_2017.csv']
dfTab = []
csvDir = str(Path.home()) + '/projects/chicago-crimes/'
for f in files:
    df = spark.read.csv(csvDir + f, header=True, inferSchema=True)
    dfTab.append(df)

# %%

# crimes = dfTab[2]

crimes = dfTab[0].union(dfTab[1]).union(dfTab[2])

# %%
# Drop NA.

crimes2 = crimes.dropna(how='any')

# %%
print('Dropped ', crimes.count() - crimes2.count(), ' rows with NA')

# %%

crimes3 = crimes2.drop("ID", "Case Number")
crimes3.printSchema()

# %%
# Convert date to new columns.

crimes3.select("Date").show(3, truncate=False)

tmpdf = crimes3.withColumn('date2', to_timestamp('Date', 'MM/dd/yyyy HH:mm:ss'))
tmpdf.select("date2").show(5)

# %%

crimes4 = tmpdf.withColumn('Year', year('date2')).withColumn('Month', month('date2'))\
    .withColumn('Day', dayofmonth('date2')).withColumn('Hour', hour('date2'))\
    .withColumn('Minute', minute('date2'))

crimes4 = crimes4.drop("Date", "date2", "Updated On")
crimes4.printSchema()

# %%

# Convert categorical attributes to numerical. 
categorical_columns = ['Block', 'IUCR', 'Description', 'Location Description', 
                       'FBI Code']

indexers = [StringIndexer(inputCol=column, outputCol=column+"Index") 
            for column in categorical_columns]

pipeline = Pipeline(stages=indexers)
indexed_df = pipeline.fit(crimes4).transform(crimes4)

# %%

for col in categorical_columns:
    indexed_df = indexed_df.drop(col)
crimes5 = indexed_df.drop("Location")
crimes5.printSchema()


# %%
def plotHistogram(crimes_df, targetCol):
    ptypes = crimes_df.select(targetCol).distinct().rdd.flatMap(lambda x: x).collect()
    bins, counts = crimes_df.select(targetCol).rdd.flatMap(lambda x: x).histogram(sorted(ptypes))
    df = pd.DataFrame(list(zip(bins, counts)), columns=['bin', 'frequency']).set_index('bin')

    ax = df.sort_values(by='frequency', ascending=True).plot(kind='barh', figsize=(20,10), 
                                                      title='Amount of Crimes by Primary Type')

    ax.set_xlabel("Amount of Crimes")
    ax.set_ylabel("Crime Type") 
    return df


# %%
# Show histogram of Primary Type.
df = plotHistogram(crimes5, 'Primary Type')


# %%

unwanted_classes = df[df['frequency'] < 20000]

# %%

classes_to_exclude = list(unwanted_classes.index)
print(classes_to_exclude)

# %%
from pyspark.sql.functions import when, col

tmpdf = crimes5.withColumn('PrimaryType', 
                           when(col('Primary Type').isin(classes_to_exclude), 'OTHER')
                           .otherwise(col('Primary Type')))

# Check    
# tmpdf.select('PrimaryType', 'Primary Type').filter(col('PrimaryType') == 'OTHER').show(5)

crimes6 = tmpdf.drop('Primary Type')

# %%
# Show histogram of PrimaryType.

df = plotHistogram(crimes6, 'PrimaryType')


# %%

Classes = crimes6.select('PrimaryType').distinct().rdd.flatMap(lambda x: x).collect()
print(Classes)

# %%
# Convert categorical target to numerical. 

primaryTypeIndexer = StringIndexer(inputCol='PrimaryType', outputCol="PrimaryTypeIndex")
primaryTypeIndexerModel = primaryTypeIndexer.fit(crimes6)
tmpdf = primaryTypeIndexerModel.transform(crimes6)

crimes7 = tmpdf

crimes_for_corr = tmpdf.drop("PrimaryType")

# %%
crimes_for_corr.select('PrimaryTypeIndex').show(5)


# %%
# print(crimes_for_corr.dtypes)
# Convert Longitude to double

tmpdf = crimes_for_corr.withColumn("longitude", crimes_for_corr["Longitude"].cast(DoubleType()))
crimes_for_corr = tmpdf.drop("Longitude")

# %%
# Feature selection.


def correlation_matrix(df, corr_columns, method='pearson'):
    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=corr_columns, outputCol=vector_col)
    df_vector = assembler.setHandleInvalid("skip").transform(df).select(vector_col)
    matrix = Correlation.corr(df_vector, vector_col, method)

    result = matrix.collect()[0]["pearson({})".format(vector_col)].values
    return pd.DataFrame(result.reshape(-1, len(corr_columns)), columns=corr_columns, index=corr_columns)


# %%

matrix = correlation_matrix(crimes_for_corr, crimes_for_corr.columns)

# %%

plt.figure(figsize=(20,10))
sns.heatmap(matrix, annot=True, cmap=plt.cm.Reds)
plt.show()

# %%

# Correlation with output variable
cor_target = matrix['PrimaryTypeIndex'].abs()
# Selecting highly correlated features
relevant_features = cor_target[cor_target > 0.2]
print(relevant_features)

# %%

Features = [ 'IUCRIndex', 'DescriptionIndex', 'FBI CodeIndex']
Target = 'PrimaryTypeIndex'

# %%
# Create 'features' column for RandomForest.

assembler = VectorAssembler(inputCols=Features, outputCol="features")
crimes9 = assembler.transform(crimes7)

crimes9.select(Features + ['features']).show(3)

# %%
# Split dataset to Training Set & Test Set

test_df, train_df = crimes9.randomSplit([0.2, 0.8], 3)

# %%
from pyspark.ml.classification import RandomForestClassifier

# %%
# Train a RandomForest model.

rf = RandomForestClassifier(labelCol="PrimaryTypeIndex", featuresCol="features", 
                            numTrees=70, maxBins=400, maxDepth=15, minInstancesPerNode=30)

outputConverter = IndexToString(inputCol="prediction", outputCol="predictedPrimaryType")
outputConverter.setLabels(primaryTypeIndexerModel.labels)

# %%

pipeline = Pipeline(stages=[rf, outputConverter])
model = pipeline.fit(train_df)

# %%

# Make predictions.

predictions = model.transform(test_df)

# %%

predictions.select('features', 'PrimaryType', 'predictedPrimaryType').show(5)

# %%
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# Select (prediction, true label) and compute test error

evaluator = MulticlassClassificationEvaluator(labelCol="PrimaryTypeIndex", 
                                              predictionCol="prediction", 
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print("Model accuracy: ", accuracy)

# %%

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`."""
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else'd'
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white"  if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# %%
    
class_temp = predictions.select("PrimaryType").groupBy("PrimaryType")\
    .count().sort('count', ascending=False).toPandas()
class_temp = class_temp["PrimaryType"].values.tolist()

# %%

y_true = predictions.select("PrimaryType")
y_true = y_true.toPandas()
y_pred = predictions.select("predictedPrimaryType")
y_pred = y_pred.toPandas()

# %%

cnf_matrix = confusion_matrix(y_true, y_pred, labels=class_temp)
print(cnf_matrix)

# %%
# Plot non-normalized confusion matrix

plt.figure(figsize=(20,10))
plot_confusion_matrix(cnf_matrix, classes=class_temp, 
                      title='Confusion matrix, without normalization')
plt.show()

# %%
# Plot normalized confusion matrix

plt.figure(figsize=(20,10))
plot_confusion_matrix(cnf_matrix, classes=class_temp, normalize=True,
                      title='Normalized confusion matrix')
plt.show()

# %%
