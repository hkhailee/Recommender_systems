import logging
import lift
from tqdm.auto import tqdm
from scipy.sparse import csr_matrix, triu
import numpy as np
import pandas as pd


import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as sps

from lenskit import crossfold as xf
from lenskit.datasets import MovieLens
from lenskit import topn , batch , util
from lenskit.algorithms import Recommender, als, basic, bias, item_knn, user_knn, svd, tf
from lenskit.batch import predict, recommend

from lenskit.metrics.predict import rmse
from lenskit import topn
from binpickle import dump, load

from lenskit.algorithms.tf import BPR

import sys, getopt

# I want a logger for information
_log = logging.getLogger(__name__)

'''
one trains the algorithm and saves
it to a file with binpickle
'''
#####   loading in the data    #####
train_0 = pd.read_csv('ml-25m/train-0.csv')
train_1 = pd.read_csv('ml-25m/train-1.csv')
train_2 = pd.read_csv('ml-25m/train-2.csv')
train_3 = pd.read_csv('ml-25m/train-3.csv')
train_4 = pd.read_csv('ml-25m/train-4.csv')
train_list = [train_0, train_1, train_2, train_3, train_4]

test_0 = pd.read_csv('ml-25m/test-0.csv')
test_1 = pd.read_csv('ml-25m/test-1.csv')
test_2 = pd.read_csv('ml-25m/test-2.csv')
test_3 = pd.read_csv('ml-25m/test-3.csv')
test_4 = pd.read_csv('ml-25m/test-4.csv')
test_list = [test_0, test_1, test_2, test_3, test_4]

df = pd.DataFrame(columns=['item','score','user','rank'])
def main():
  in_alg = sys.argv[1] # instance of algorithm name 
  in_alg = in_alg.lower()

  st = None

  test_all = [pd.read_csv('test_master.csv')]
  #popular (R)
  if(in_alg == "popular"):
   
      rec_all = []
      print("algorithm chosen is popular")
     
      #loop over test partitions 
      for p in train_list:
         alg = basic.Popular() # create an instance of the algorithm object 
         alg = Recommender.adapt(alg)
         # train the algorithm 
         alg.fit(p)
         # produce recomendations
         recs = recommend(alg, p['user'].unique(),50)
         df = recs
         rec_all.append(df)
           
      # out of for
      recs = pd.concat(rec_all, ignore_index=True)
      test = pd.concat(test_all, ignore_index=True)
      
      #find ndcg and record  
      rla = topn.RecListAnalysis()
      rla.add_metric(topn.ndcg)
      metric = rla.compute(recs, test)
      # and compute overall system performance
      record = metric['ndcg'].mean()
      print('NDCG:', record)
      # export to csv for visualization 
      df = pd.DataFrame([['popular',record]], columns = ['alg name', 'ndcg'] )
      df.to_csv('ndcg.csv', mode ='a', index=False, header = False)
    
        

  #Bias (R,P)
  elif(in_alg == "bias"):
      
      print("algorithm chosen is bias")
        
      rec_all = []
      pred_all = []
      #loop over test partitions 
      for p,t in zip(train_list, test_list):
          alg = bias.Bias() # create an instance of the algorithm object
          alg = Recommender.adapt(alg)
          # train the algorithm 
          alg.fit(p)
          # produce recomendations
          recs = recommend(alg, p['user'].unique(),50)
          rec_all.append(recs)
          # produce predictions
          alg.predictor 
          preds = predict(alg, t)
          pred_all.append(preds)
                  
      # out of for
      recs = pd.concat(rec_all, ignore_index=True)
      test = pd.concat(test_all, ignore_index=True)
      preds = pd.concat(pred_all, ignore_index=True)
    
      #find the rmse and record
      rmse_return = rmse(preds['prediction'],preds['rating'])
      print('RMSE:', rmse_return)
      df = pd.DataFrame([['bias',rmse_return]], columns = ['alg name', 'rmse'] )
      df.to_csv('rmse.csv', mode ='a', index=False, header = False)
      
      #find ndcg and record  
      rla = topn.RecListAnalysis()
      rla.add_metric(topn.ndcg)
      metric = rla.compute(recs, test)
      # and compute overall system performance
      record = metric['ndcg'].mean()
      print('NDCG:', record)
      # export to csv for visualization 
      df = pd.DataFrame([['bias',record]], columns = ['alg name', 'ndcg'] )
      df.to_csv('ndcg.csv', mode ='a', index=False, header = False)

  #UserUser (R,P)
  elif(in_alg == "user-user"):
       
      print("algorithm chosen is user-user")
      rec_all = []
      pred_all = []
        
      for p,t in zip(train_list, test_list):
          alg = user_knn.UserUser(30) # create an instance of the algorithm object
          alg = Recommender.adapt(alg)
          # train the algorithm 
          alg.fit(p)
          # produce recomendations
          recs = recommend(alg, p['user'].unique(),50)
          rec_all.append(recs)
          # produce predictions
          alg.predictor 
          preds = predict(alg, t)
          pred_all.append(preds)
                  
      # out of for
      recs = pd.concat(rec_all, ignore_index=True)
      test = pd.concat(test_all, ignore_index=True)
      preds = pd.concat(pred_all, ignore_index=True)
    
      #find the rmse and record
      rmse_return = rmse(preds['prediction'],preds['rating'])
      print('RMSE:', rmse_return)
      df = pd.DataFrame([['user-user',rmse_return]], columns = ['alg name', 'rmse'] )
      df.to_csv('rmse.csv', mode ='a', index=False, header = False)
      
      #find ndcg and record  
      rla = topn.RecListAnalysis()
      rla.add_metric(topn.ndcg)
      metric = rla.compute(recs, test)
      # and compute overall system performance
      record = metric['ndcg'].mean()
      print('NDCG:', record)
      # export to csv for visualization 
      df = pd.DataFrame([['user-user',record]], columns = ['alg name', 'ndcg'] )
      df.to_csv('ndcg.csv', mode ='a', index=False, header = False)

  #ItemItem (R,P)
  elif(in_alg == "item-item"):
      print("algorithm chosen is item-item")
    
      rec_all = []
      pred_all = []
      
      for p,t in zip(train_list, test_list):
          alg = item_knn.ItemItem(20,2) # create an instance of the algorithm object
          alg = Recommender.adapt(alg)
          # train the algorithm 
          alg.fit(p)
          # produce recomendations
          recs = recommend(alg, p['user'].unique(),50)
          rec_all.append(recs)
          # produce predictions
          alg.predictor 
          preds = predict(alg, t)
          pred_all.append(preds)
        
        
                  
      # out of for
  # given a fitted recommender find similarities
      #i = 1
      #Q = alg.item_fetaures_
      #sims = Q @ Q [i, :].T
      #X = scipy.linalg.norm(Q , axis = 1)
      #X.to_csv('testing.csv', index = False) 
    
      recs = pd.concat(rec_all, ignore_index=True)
      test = pd.concat(test_all, ignore_index=True)
      preds = pd.concat(pred_all, ignore_index=True)
    
      #find the rmse and record
      rmse_return = rmse(preds['prediction'],preds['rating'])
      print('RMSE:', rmse_return)
      df = pd.DataFrame([['item-item',rmse_return]], columns = ['alg name', 'rmse'] )
      df.to_csv('rmse.csv', mode ='a', index=False, header = False)
      
      #find ndcg and record  
      rla = topn.RecListAnalysis()
      rla.add_metric(topn.ndcg)
      metric = rla.compute(recs, test)
      # and compute overall system performance
      record = metric['ndcg'].mean()
      print('NDCG:', record)
      # export to csv for visualization 
      df = pd.DataFrame([['item-item',record]], columns = ['alg name', 'ndcg'] )
      df.to_csv('ndcg.csv', mode ='a', index=False, header = False)


  #ItemItem Sum (R)
  elif(in_alg == "item-item sum"):
      print("algorithm chosen is item-item sum")
      rec_all = []
      for p in train_list:
          alg = item_knn.ItemItem(20, 2, aggregate='sum') # create an instance of the algorithm object
          alg = Recommender.adapt(alg)
          # train the algorithm 
          alg.fit(p)
          # produce recomendations
          recs = recommend(alg, p['user'].unique(),50)
          df = recs
          rec_all.append(df)
           
      # out of for
      recs = pd.concat(rec_all, ignore_index=True)
      test = pd.concat(test_all, ignore_index=True)
      
      #find ndcg and record  
      rla = topn.RecListAnalysis()
      rla.add_metric(topn.ndcg)
      metric = rla.compute(recs, test)
      # and compute overall system performance
      record = metric['ndcg'].mean()
      print('NDCG:', record)
      # export to csv for visualization 
      df = pd.DataFrame([['item-item sum',record]], columns = ['alg name', 'ndcg'] )
      df.to_csv('ndcg.csv', mode ='a', index=False, header = False)
          
      

  #ExplicitMF (R,P)
  elif(in_alg == "explicit mf"):
        
      print("algorithm chosen is explicit mf")
      rec_all = []
      pred_all = []
      for p,t in zip(train_list, test_list):
          alg = als.BiasedMF(50) # create an instance of the algorithm object
          alg = Recommender.adapt(alg)
          # train the algorithm 
          alg.fit(p)
          # produce recomendations
          recs = recommend(alg, p['user'].unique(),50)
          rec_all.append(recs)
          # produce predictions
          alg.predictor 
          preds = predict(alg, t)
          pred_all.append(preds)
                  
      # out of for
    
      #i = 1
      #Q = alg.item_fetaures_
      #sims = Q @ Q [i, :].T
      #X = scipy.linalg.norm(Q , axis = 1)
      #X.to_csv('testing.csv', index = False)
        
      recs = pd.concat(rec_all, ignore_index=True)
      test = pd.concat(test_all, ignore_index=True)
      preds = pd.concat(pred_all, ignore_index=True)
    
      #find the rmse and record
      rmse_return = rmse(preds['prediction'],preds['rating'])
      print('RMSE:', rmse_return)
      df = pd.DataFrame([['explicit mf',rmse_return]], columns = ['alg name', 'rmse'] )
      df.to_csv('rmse.csv', mode ='a', index=False, header = False)
      
      #find ndcg and record  
      rla = topn.RecListAnalysis()
      rla.add_metric(topn.ndcg)
      metric = rla.compute(recs, test)
      # and compute overall system performance
      record = metric['ndcg'].mean()
      print('NDCG:', record)
      # export to csv for visualization 
      df = pd.DataFrame([['explicit mf',record]], columns = ['alg name', 'ndcg'] )
      df.to_csv('ndcg.csv', mode ='a', index=False, header = False)

  #svd (R,P)
  elif(in_alg == "svd"):
      print("algorithm chosen is svd")
      rec_all = []
      pred_all = []
      for p,t in zip(train_list, test_list):
          alg = svd.BiasedSVD(25) # create an instance of the algorithm object
          alg = Recommender.adapt(alg)
          # train the algorithm 
          alg.fit(p)
          # produce recomendations
          recs = recommend(alg, p['user'].unique(),50)
          rec_all.append(recs)
          # produce predictions
          alg.predictor 
          preds = predict(alg, t)
          pred_all.append(preds)
                  
      # out of for
      recs = pd.concat(rec_all, ignore_index=True)
      test = pd.concat(test_all, ignore_index=True)
      preds = pd.concat(pred_all, ignore_index=True)
    
      #find the rmse and record
      rmse_return = rmse(preds['prediction'],preds['rating'])
      print('RMSE:', rmse_return)
      df = pd.DataFrame([['svd',rmse_return]], columns = ['alg name', 'rmse'] )
      df.to_csv('rmse.csv', mode ='a', index=False, header = False)
      
      #find ndcg and record  
      rla = topn.RecListAnalysis()
      rla.add_metric(topn.ndcg)
      metric = rla.compute(recs, test)
      # and compute overall system performance
      record = metric['ndcg'].mean()
      print('NDCG:', record)
      # export to csv for visualization 
      df = pd.DataFrame([['svd',record]], columns = ['alg name', 'ndcg'] )
      df.to_csv('ndcg.csv', mode ='a', index=False, header = False)

  #ImplicitMF (R)
  elif(in_alg == "implicit mf"):
      print("algorithm chosen is implicit mf")
      rec_all = []
      for p in train_list:
          alg = als.ImplicitMF(50) # create an instance of the algorithm object
          alg = Recommender.adapt(alg)
          # train the algorithm 
          alg.fit(p)
          # produce recomendations
          recs = recommend(alg, p['user'].unique(),50)
          df = recs
          rec_all.append(df)
           
      # out of for
      recs = pd.concat(rec_all, ignore_index=True)
      test = pd.concat(test_all, ignore_index=True)
      
      #find ndcg and record  
      rla = topn.RecListAnalysis()
      rla.add_metric(topn.ndcg)
      metric = rla.compute(recs, test)
      # and compute overall system performance
      record = metric['ndcg'].mean()
      print('NDCG:', record)
      # export to csv for visualization 
      df = pd.DataFrame([['implicit mf',record]], columns = ['alg name', 'ndcg'] )
      df.to_csv('ndcg.csv', mode ='a', index=False, header = False)

  #BPR (R)
  elif(in_alg == "bpr"):
      print("algorithm chosen is bpr")
      rec_all = []
      for p in train_list:
          alg = tf.BPR(50) # create an instance of the algorithm object
          alg = Recommender.adapt(alg)
          # train the algorithm 
          alg.fit(p)
          # produce recomendations
          recs = recommend(alg, p['user'].unique(),50)
          df = recs
          rec_all.append(df)
           
      # out of for
      recs = pd.concat(rec_all, ignore_index=True)
      test = pd.concat(test_all, ignore_index=True)
      
      #find ndcg and record  
      rla = topn.RecListAnalysis()
      rla.add_metric(topn.ndcg)
      metric = rla.compute(recs, test)
      # and compute overall system performance
      record = metric['ndcg'].mean()
      print('NDCG:', record)
      # export to csv for visualization 
      df = pd.DataFrame([['bpr',record]], columns = ['alg name', 'ndcg'] )
      df.to_csv('ndcg.csv', mode ='a', index=False, header = False)

  #LIFT (R) -- provided 
  elif(in_alg == "lift"):
      print("algorithm chosen is lift")
      rec_all = []
      for p in train_list:
          alg = lift.Lift() # create an instance of the algorithm object
          alg = Recommender.adapt(alg)
          # train the algorithm 
          alg.fit(p)
          # produce recomendations
          recs = recommend(alg, p['user'].unique(),50)
          df = recs
          rec_all.append(df)
           
      # out of for
      recs = pd.concat(rec_all, ignore_index=True)
      test = pd.concat(test_all, ignore_index=True)
      
      #find ndcg and record  
      rla = topn.RecListAnalysis()
      rla.add_metric(topn.ndcg)
      metric = rla.compute(recs, test)
      # and compute overall system performance
      record = metric['ndcg'].mean()
      print('NDCG:', record)
      # export to csv for visualization 
      df = pd.DataFrame([['lift',record]], columns = ['alg name', 'ndcg'] )
      df.to_csv('ndcg.csv', mode ='a', index=False, header = False)

        
        
##### no longer using #####
#def startTraining(algorithm, isRP):
#    alg = algorithm
#    isRP = isRP
#    alg = Recommender.adapt(alg)
#    #loop over test partitions 
#    for p in train_list:
#        # train the algorithm 
#        alg.fit(p)
#    #load the trained model to binpickle
#    dump(alg, 'trained.bpk')

if __name__ == "__main__":
   main()