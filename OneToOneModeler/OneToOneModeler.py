class OneToOneModeler(object):

   # One to one modeler:
   # Assume that out of the sample of length n there are k failures and m successes, however m << k
   # In order to identify the optimal model, which predicts the probability of success,
   # and overcome te problematic situation when the susccess is very rare,
   # susmpling method is chosen in order to create subsamples of the total population where m = k
   # and on this sample create and validate the model
   
   ##################### Constructor paramaters:
   # pathTotal -> the path where the sample is located
   # sep -> how the columns are sperated (e.g. comma, tab)
   # pathOutput - > the path where the models created from subsampling simulations is stored
   # numberOfSamples -> number of simulations to be performed
   # classList -> each element of this list is the reference category of a variable used, in case 
   # if the variable should be treated as numeric the element should be "numeric"
   # varList -> the names of the variables used for the model building
   # initialSeed -> the seed for the simulations
   # badVar -> the target variable (0: failure, 1: success)
   # outputFile -> the output file name
   # replace -> whether an observation during the sampling can be chosen again or not
   # trainProp -> how the validation and training samples will be split, if the values is out of 0,1 range then 
   # the split will be random based on the current seed
   
   ############### executeOneToOne parameters
   # selection -> whether a feature selection algorithm will be used
   # selectionCriteria -> the criteria on which the feature selection algorithm will be based ('bic' or 'aic')
   # direction -> the direction of the feature elimination ('Forward', 'Backward', 'Both', 'allCombinationsLogistic' )
   
   def __init__(self, pathTotal, sep, pathOutput,  numberOfSamples, classList, varList, initialSeed, badVar, outputFile, replace = True, trainProp = 0):
       
       self.pathTotal = pathTotal         
       self.pathOutput = pathOutput
       self.numberOfSamples = numberOfSamples
       self.classList = classList
       self.varList = varList
       self.initialSeed = initialSeed
       self.badVar = badVar
       self.outputFile = outputFile
       self.replace = replace
       self.sep = sep
       self.trainProp = trainProp
    
   
   def Forward(self, odject_inputForward, data_inputForward, metricForward = "aic"):
       import statsmodels.api as sm
       import statsmodels.formula.api as smf
       model_formula_full = odject_inputForward.formula
       Xs_full = model_formula_full.split("~")[1].split("+")
       Xs_optimum = ["1"]
       if( metricForward == "aic"): metric_optimum = smf.glm(model_formula_full.split("~")[0]+"~"+"+1", data = data_inputForward, family=sm.families.Binomial()).fit().aic
       else: metric_optimum = smf.glm(model_formula_full.split("~")[0]+"~"+"+1", data = data_inputForward, family=sm.families.Binomial()).fit().bic
       i = 0
       for Xs_full_i in Xs_full:
          Xs_temp = Xs_optimum
          if (i == 0): Xs_temp = [Xs_full_i]
          else :  Xs_temp = Xs_temp + [Xs_full_i]    
          if( metricForward == "aic"): metric_temp = smf.glm(model_formula_full.split("~")[0]+"~"+"+".join(Xs_temp), data = data_inputForward, family=sm.families.Binomial()).fit().aic
          else :  metric_temp = smf.glm(model_formula_full.split("~")[0]+"~"+"+".join(Xs_temp), data = data_inputForward, family=sm.families.Binomial()).fit().bic
          if (metric_temp < metric_optimum):
              metric_optimum = metric_temp
              Xs_optimum = Xs_temp
          i += 1
       return model_formula_full.split("~")[0]+"~"+"+".join(Xs_optimum)  
   
               
   def Backward(self, odject_inputBackward, data_inputBackward, metricBackward = "aic"):
       import statsmodels.api as sm
       import statsmodels.formula.api as smf
       model_formula_full = odject_inputBackward.formula
       Xs_full = model_formula_full.split("~")[1].split("+")
       Xs_optimum = Xs_full
       if( metricBackward == "aic"): metric_optimum = smf.glm(model_formula_full.split("~")[0]+"~"+"+".join(Xs_optimum), data = data_inputBackward, family=sm.families.Binomial()).fit().aic
       else: metric_optimum = smf.glm(model_formula_full.split("~")[0]+"~"+"+".join(Xs_optimum), data = data_inputBackward, family=sm.families.Binomial()).fit().bic
       for Xs_full_i in Xs_full:
          Xs_temp = Xs_optimum[:]
          Xs_temp.remove(Xs_full_i)   
          if( metricBackward == "aic"): metric_temp = smf.glm(model_formula_full.split("~")[0]+"~"+"+".join(Xs_temp), data = data_inputBackward, family=sm.families.Binomial()).fit().aic
          else :  metric_temp = smf.glm(model_formula_full.split("~")[0]+"~"+"+".join(Xs_temp), data = data_inputBackward, family=sm.families.Binomial()).fit().bic
          if (metric_temp < metric_optimum):
              metric_optimum = metric_temp
              Xs_optimum = Xs_temp
       return model_formula_full.split("~")[0]+"~"+"+".join(Xs_optimum) 
       
 
   def Both(self, odject_inputBoth, data_inputBoth, metricBoth = "aic"):
       import statsmodels.api as sm
       import statsmodels.formula.api as smf
       model_formula_full = odject_inputBoth.formula
       Xs_full = model_formula_full.split("~")[1].split("+")
       Xs_optimum = Xs_full
       Xs_deleted = []
       if( metricBoth == "aic"): metric_optimum = smf.glm(model_formula_full.split("~")[0]+"~"+"+".join(Xs_optimum), data = data_inputBoth, family=sm.families.Binomial()).fit().aic
       else: metric_optimum = smf.glm(model_formula_full.split("~")[0]+"~"+"+".join(Xs_optimum), data = data_inputBoth, family=sm.families.Binomial()).fit().bic
       for Xs_full_i in Xs_full:
          Xs_temp = Xs_optimum[:]
          Xs_temp.remove(Xs_full_i)
          Xs_deleted = Xs_deleted + [Xs_full_i]
          if( metricBoth == "aic"): metric_temp = smf.glm(model_formula_full.split("~")[0]+"~"+"+".join(Xs_temp), data = data_inputBoth, family=sm.families.Binomial()).fit().aic
          else :  metric_temp = smf.glm(model_formula_full.split("~")[0]+"~"+"+".join(Xs_temp), data = data_inputBoth, family=sm.families.Binomial()).fit().bic
          if (metric_temp < metric_optimum):
              metric_optimum = metric_temp
              Xs_optimum = Xs_temp
          if (len(Xs_deleted) > 1):
              Xs_deleted_temp = Xs_deleted[:]
              for Xs_deleted_j in Xs_deleted:
                  if (Xs_deleted_j!= Xs_full_i): 
                     Xs_temp =  Xs_optimum + [Xs_deleted_j]
                     if( metricBoth == "aic"): metric_temp = smf.glm(model_formula_full.split("~")[0]+"~"+"+".join(Xs_temp), data = data_inputBoth, family=sm.families.Binomial()).fit().aic
                     else: metric_temp = smf.glm(model_formula_full.split("~")[0]+"~"+"+".join(Xs_temp), data = data_inputBoth, family=sm.families.Binomial()).fit().bic
                     if (metric_temp < metric_optimum):
                        metric_optimum = metric_temp
                        Xs_optimum = Xs_temp 
                        Xs_deleted_temp.remove(Xs_deleted_j)
              Xs_deleted = Xs_deleted_temp
              
       return model_formula_full.split("~")[0]+"~"+"+".join(Xs_optimum)  
   

   def allCombinationsLogistic(self, odject_input, data_input, metric = "aic"):
       
       import numpy as np
       import itertools 
       import statsmodels.api as sm
       import statsmodels.formula.api as smf
       from functools import reduce
       res_initial = odject_input.fit()
       if (metric == "aic"): metric_optimum = res_initial.aic
       else : metric_optimum = res_initial.bic
       model_formula_optimum = odject_input.formula
       Xs_init = np.array(model_formula_optimum.split("~")[1].split("+"))
       Xs_comb = reduce(lambda acc, x: acc + list(itertools.combinations(Xs_init, x)), range(1, len(Xs_init) + 1), [])
       for Xs_comb_list in Xs_comb: 
          formula_temp = model_formula_optimum.split("~")[0]+"~"+"+".join(Xs_comb_list)
          if (metric == "aic"): metric_temp =  model_temp = smf.glm(formula_temp, data = data_input, family=sm.families.Binomial()).fit().aic
          else:  metric_temp = model_temp = smf.glm(formula_temp, data = data_input, family=sm.families.Binomial()).fit().bic
          if (metric_temp < metric_optimum):
            metric_optimum = metric_temp
            model_formula_optimum = model_temp.formula
       return  model_formula_optimum 
   
  
   def OneToOneSampleCreator (self, sample, badVariable, currentSeed, replacementType):
       
       import numpy as np
       import pandas as pd
       onlyPositives = sample[sample.eval(badVariable) == 1]    
       onlyPositives = onlyPositives.reset_index(drop=True)
       onlyNegatives = sample[sample.eval(badVariable) == 0]
       onlyNegatives = onlyNegatives.reset_index(drop=True)
       np.random.seed(int(currentSeed))
       onlyNegativesRandom = onlyNegatives.iloc[np.random.choice(onlyNegatives.index, onlyPositives.shape[0], replace = replacementType).tolist(),:] 
       onlyNegativesRandom = onlyNegativesRandom.reset_index(drop=True)
       sampleOnetToOne = pd.concat([onlyPositives, onlyNegativesRandom])
       sampleOnetToOne = sampleOnetToOne.reset_index(drop=True)
       return sampleOnetToOne
   
   def getSensitivity(self, sample, predictedProb, badVariable):
       
        import numpy as np
        predictedPositive = np.array([1]*sample.shape[0])
        predictedPositive[ np.array(sample.loc[:,predictedProb]) > 0.5] = 0
        bad = np.array(sample.loc[:,badVariable])
        predictedPositive = predictedPositive[bad == 0]
        bad = bad[bad == 0]       
        return sum(bad == predictedPositive)/ len(bad)
    
  
    
   def sampleSplitter(self, sample, trainProp, currentSeed):
       import numpy as np
       sample.loc[:,"aa"] = range(sample.shape[0])
       if (trainProp > 0 and trainProp < 1):
           np.random.seed(int(currentSeed))
           trainingAA = np.random.choice(range(sample.shape[0]), int(trainProp*sample.shape[0]), replace = False).tolist()
           trainingSample = sample.iloc[trainingAA,:]
           validationSample = sample.loc[~sample["aa"].isin(trainingAA)]
       else:
           np.random.seed(int(currentSeed))
           trainingAA = np.unique(np.random.choice(range(sample.shape[0]), sample.shape[0])).tolist()
           trainingSample = sample.iloc[trainingAA,:]
           validationSample = sample.loc[~sample["aa"].isin(trainingAA)]
           
       return trainingSample, validationSample
   
    
   def getDiagnostics(self, sample, badVariable, predictedProbability):
       from sklearn import metrics
       import numpy as np
       fpr, tpr, thersholds = metrics.roc_curve(sample.loc[:,badVariable], sample.loc[:,predictedProbability], pos_label = 0)
       AUC = metrics.auc(fpr,tpr)
       AR  = 2*AUC - 1
       KS =  np.max(np.abs(fpr - tpr))
       return AUC, AR, KS
   
   
    
   def executeOneToOne (self, selection = False, selectionCriteria = "aic", direction = "Forward"):
       
        import pandas as pd
        import numpy as np
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from datetime import datetime
        initialModelFormulaDep = ["C("+self.badVar+",Treatment(0))"]
        initialModelFormulaIndep = [""]
        for i in range(len(self.varList)):
            if (i == 0): 
                if (self.classList[i] == "numeric"): initialModelFormulaIndep = [self.varList[i]]
                else: initialModelFormulaIndep = ["C("+self.varList[i]+",Treatment("+str(self.classList[i])+"))"]
            else:  
                if (self.classList[i] == "numeric"): initialModelFormulaIndep = initialModelFormulaIndep + [self.varList[i]]
                else: initialModelFormulaIndep = initialModelFormulaIndep + ["C("+self.varList[i]+",Treatment("+str(self.classList[i])+"))"]
        
        initialModelFormulaIndepFinal = "+".join(initialModelFormulaIndep)
        finalInitialFormula = initialModelFormulaDep[0] + "~" + initialModelFormulaIndepFinal
        dataTotal = pd.read_csv(self.pathTotal, sep = self.sep, header = 0)
        np.random.seed(self.initialSeed)
        sampleSeeds = np.random.uniform(1,100000,self.numberOfSamples)
        i = 0
        for randomSeed in sampleSeeds:
           
           dataTrain, dataValidation = self.sampleSplitter(dataTotal, self.trainProp, randomSeed)
           
           workingSample = self.OneToOneSampleCreator(dataTrain, self.badVar, randomSeed, self.replace)
           workingSampleValidation = self.OneToOneSampleCreator(dataValidation, self.badVar, randomSeed, self.replace)

           if (not selection): sample_model = smf.glm(finalInitialFormula, data = workingSample, family=sm.families.Binomial())
           else: 
              if (direction == "Forward"): sample_model = smf.glm (self.Forward(smf.glm(finalInitialFormula, data = workingSample, family=sm.families.Binomial()), data_inputForward = workingSample, metricForward = selectionCriteria) ,data = workingSample, family=sm.families.Binomial())
              elif (direction == "Backward"): sample_model = smf.glm (self.Backward(smf.glm(finalInitialFormula, data = workingSample, family=sm.families.Binomial()), data_inputBackward = workingSample, metricBackward = selectionCriteria) ,data = workingSample, family=sm.families.Binomial())
              elif (direction == "Both"):  sample_model = smf.glm (self.Both(smf.glm(finalInitialFormula, data = workingSample, family=sm.families.Binomial()), data_inputBoth = workingSample, metricBoth = selectionCriteria) ,data = workingSample, family=sm.families.Binomial())
              elif (direction == "allCombinations"): sample_model = smf.glm (self.allCombinationsLogistic(smf.glm(finalInitialFormula, data = workingSample, family=sm.families.Binomial()), data_input = workingSample, metric = selectionCriteria) ,data = workingSample, family=sm.families.Binomial())
           res = sample_model.fit()
           
           workingSample.loc[:,"PredictedProbTrain"] = res.predict(workingSample.loc[:,self.varList]).tolist()
           workingSampleValidation.loc[:,"PredictedProbValidation"] = res.predict(workingSampleValidation.loc[:,self.varList]).tolist()
           
           sensitivityTrain = self.getSensitivity(workingSample, "PredictedProbTrain", self.badVar)
           sensitivityValidation = self.getSensitivity(workingSampleValidation, "PredictedProbValidation", self.badVar)
           
           AUC_Train, AR_Train, KS_Train = self.getDiagnostics(workingSample, self.badVar, "PredictedProbTrain" )
           AUC_Validation, AR_Validation, KS_Validation = self.getDiagnostics(workingSampleValidation, self.badVar, "PredictedProbValidation" )
 
           
           Validation_results_dict = {"sample_number": ["Sample"+str(i)], "seed": [int(randomSeed)], "AUC_Train": [AUC_Train], 
                                      "AR_Train": [AR_Train], "KS_Train": [KS_Train], "Sensitivity_Train" :[sensitivityTrain],
                                      "AUC_Validation": [AUC_Validation],"AR_Validation": [AR_Validation], "KS_Validation": [KS_Validation], 
                                      "Sensitivity_Validation" :[sensitivityValidation]}
           if (i == 0) : 
               outputResult = pd.DataFrame(res.params).T
               Validation_results = pd.DataFrame(Validation_results_dict)
           else : 
               outputResult = pd.concat([outputResult,pd.DataFrame(res.params).T], sort=False)
               Validation_results = pd.concat([Validation_results,pd.DataFrame(Validation_results_dict)], sort=False)
           i += 1
           
        outputResult =  pd.concat([Validation_results,outputResult], axis = 1, sort=False)
        now = datetime.now()
        dt_string = now.strftime("_%d_%m_%Y_%H_%M_%S")
        outputResult.to_excel(self.pathOutput+"\\"+self.outputFile+dt_string+".xlsx", index = None, header = True, sheet_name='One to One models')
        return outputResult
        
  

