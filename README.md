# DRC_violation_prediction

## Introduction

In this work we proposed a machine learning based
algorithm to predict the design rule violations, including wire
short, via short, wire spacing and via spacing violations, based on
placement information.

Two machine learning frameworks (SGD and nueral network) were
tested on industry level data sets.

## Dataset

[ISPD2018](http://www.ispd.cc/contests/18/) and [ISPD2019](http://www.ispd.cc/contests/19/) were used as the datasets. 

## How to run

First clone the repo.

Then download the datasets. Now you have a dataset, but it's unlabeled, which means you don't know where the DRC violations are.

Download [dr. cu](https://github.com/cuhk-eda/dr-cu) as the detailed router. Route each design in your dataset, record the location of violations in each design to label the dataset.

Change the input directory in your source code.

Run `dnn_train_and_predict.py` to train the neural network, and `drcPredictionSvm.py` to train the SGD model.

## Details and results

Details of this work and comparison between this work and other works can be found in the pdf file.
