from scipy.spatial.distance import cityblock, euclidean
import numpy as np
np.set_printoptions(suppress = True)
import pandas
from EER import evaluateEER
from EER_GMM import evaluateEERGMM
from sklearn.svm import OneClassSVM
from sklearn.mixture import GMM
import warnings
warnings.filterwarnings("ignore")
from abc import ABCMeta, abstractmethod

class Detector:
    
    __metaclass__ = ABCMeta
 
    def __init__(self, subjects):
        self.user_scores = []
        self.imposter_scores = []
        self.mean_vector = []
        self.subjects = subjects
        
    @abstractmethod
    def training(self):
        pass
    
    @abstractmethod
    def testing(self):
        pass
 
    def evaluate(self):
        eers = []
 
        for subject in subjects:        
            genuine_user_data = data.loc[data.subject == subject, \
                                         "H.period":"H.Return"]
            imposter_data = data.loc[data.subject != subject, :]
            
            self.train = genuine_user_data[:200]
            self.test_genuine = genuine_user_data[200:]
            self.test_imposter = imposter_data.groupby("subject"). \
                                 head(5).loc[:, "H.period":"H.Return"]
 
            self.training()
            self.testing()
            
            if isinstance(self, GMMDetector):
                eers.append(evaluateEERGMM(self.user_scores, \
                                     self.imposter_scores))
            else:
                eers.append(evaluateEER(self.user_scores, \
                                     self.imposter_scores))
        return np.mean(eers), np.std(eers)
        
class ManhattanDetector(Detector):
    
    def training(self):
        self.mean_vector = self.train.mean().values         
 
    def testing(self):
        for i in range(self.test_genuine.shape[0]):
            cur_score = cityblock(self.test_genuine.iloc[i].values, \
                                   self.mean_vector)
            self.user_scores.append(cur_score)
 
        for i in range(self.test_imposter.shape[0]):
            cur_score = cityblock(self.test_imposter.iloc[i].values, \
                                   self.mean_vector)
            self.imposter_scores.append(cur_score)
            
class ManhattanFilteredDetector(Detector):
    
    def training(self):
        self.mean_vector = self.train.mean().values
        self.std_vector = self.train.std().values
        dropping_indices = []
        for i in range(self.train.shape[0]):
            cur_score = euclidean(self.train.iloc[i].values, 
                                   self.mean_vector)
            if (cur_score > 3*self.std_vector).all() == True:
                dropping_indices.append(i)
        self.train = self.train.drop(self.train.index[dropping_indices])
        self.mean_vector = self.train.mean().values

    def testing(self):
        for i in range(self.test_genuine.shape[0]):
            cur_score = cityblock(self.test_genuine.iloc[i].values, \
                                   self.mean_vector)
            self.user_scores.append(cur_score)
 
        for i in range(self.test_imposter.shape[0]):
            cur_score = cityblock(self.test_imposter.iloc[i].values, \
                                   self.mean_vector)
            self.imposter_scores.append(cur_score)
            
class ManhattanScaledDetector(Detector):
    
    def training(self):
        self.mean_vector = self.train.mean().values
        self.mad_vector  = self.train.mad().values

    def testing(self):
        for i in range(self.test_genuine.shape[0]):
            cur_score = 0
            for j in range(len(self.mean_vector)):
                cur_score = cur_score + \
                            abs(self.test_genuine.iloc[i].values[j] - \
                            self.mean_vector[j]) / self.mad_vector[j]
            self.user_scores.append(cur_score)
 
        for i in range(self.test_imposter.shape[0]):
            cur_score = 0
            for j in range(len(self.mean_vector)):
                cur_score = cur_score + \
                            abs(self.test_imposter.iloc[i].values[j] - \
                            self.mean_vector[j]) / self.mad_vector[j]
            self.imposter_scores.append(cur_score)
            
class SVMDetector(Detector):
    
    def training(self):
        self.clf = OneClassSVM(kernel='rbf',gamma=26)
        self.clf.fit(self.train)
 
    def testing(self):
        self.user_scores = -self.clf.decision_function(self.test_genuine)
        self.imposter_scores = -self.clf.decision_function(self.test_imposter)
        self.user_scores = list(self.user_scores)
        self.imposter_scores = list(self.imposter_scores)
        
class GMMDetector(Detector):
    
     def training(self):
        self.gmm = GMM(n_components = 2, covariance_type = 'diag', 
                        verbose = False )
        self.gmm.fit(self.train)
 
     def testing(self):
        for i in range(self.test_genuine.shape[0]):
            j = self.test_genuine.iloc[i].values
            cur_score = self.gmm.score(j)
            self.user_scores.append(cur_score)
 
        for i in range(self.test_imposter.shape[0]):
            j = self.test_imposter.iloc[i].values
            cur_score = self.gmm.score(j)
            self.imposter_scores.append(cur_score)        

path = "D:\\Keystroke\\keystroke.csv" 
data = pandas.read_csv(path)
subjects = data["subject"].unique()
print "average EER for Manhattan detector:"
obj = ManhattanDetector(subjects)
print(obj.evaluate())
print "====================================================================="
print "average EER for Manhattan filtered detector:"
obj = ManhattanFilteredDetector(subjects)
print(obj.evaluate())
print "====================================================================="
print "average EER for Manhattan scaled detector:"
obj = ManhattanScaledDetector(subjects)
print(obj.evaluate())
print "====================================================================="
print "average EER for One class SVM detector:"
obj = SVMDetector(subjects)
print(obj.evaluate())
print "====================================================================="
print "average EER for GMM detector:"
obj = GMMDetector(subjects)
print(obj.evaluate())