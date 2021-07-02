import numpy as np
from sklearn import svm
import os

trainFolder = "/bigdata/ethan/training_set_nets"
testFolder = "/bigdata/ethan/validation_set_nets"
vioFolder = "/bigdata/ethan/DRC_data_set"

trainLis = ["8t1", "8t3", "8t4", "8t6", "8t7", "8t9", "8t10", "9t2", "9t3", "9t7", "9t9", "9t10"]
validLis = ["8t2", "8t5", "8t8", "9t1", "9t6", "9t8"]

