setwd("S://R Studio Study//Projects//Human Resource Project")
HR_Test=read.csv("hr_test.csv",stringsAsFactors = F)
HR_Train=read.csv("hr_train.csv",stringsAsFactors = F)

HR_Train$DATA="train"
HR_Test$DATA="test"

library(dplyr)

HR_Test$left=NA

HR_all=rbind(HR_Train,HR_Test)

glimpse(HR_all)

#Data prep:

#Sales Column:

round(prop.table(table(HR_all$sales,HR_all$left),1),1)

HR_all=HR_all%>%
  mutate(Sales_Grp1=as.numeric(sales%in%c("accounting","hr","IT","marketing","product_mng",
        "RandD","sales","support","technical")),
        
         Sales_Mgmt_Grp2=as.numeric(sales=="management"))

HR_all=HR_all%>%select(-sales)

#Salary:

round(prop.table(table(HR_all$salary,HR_all$left),1),1)

HR_all=HR_all%>%
  mutate(Salary.Low.Medium=as.numeric(salary%in%c('low','medium')),
         Salary.High=as.numeric(salary=="high"))

HR_all=HR_all%>%select(-salary)

#Satisfaction Level:

boxplot(HR_all$satisfaction_level,horizontal = T)
unique(HR_all$satisfaction_level)

#last_evaluation:

unique(HR_all$last_evaluation)

boxplot(HR_all$last_evaluation,horizontal = T)

#number_project

unique(HR_all$number_project)

boxplot(HR_all$number_project,horizontal = T)

#average_montly_hours:

unique(HR_all$average_montly_hours)

boxplot(HR_all$average_montly_hours,horizontal = T)

# time_spend_company

unique(HR_all$time_spend_company)

boxplot(HR_all$time_spend_company,horizontal = T)

#Treating outlier in time_spend_company:

summary(HR_all$time_spend_company)

IQR(HR_all$time_spend_company)

Tmin=3-1.5
Tmax=4+1.5

X=HR_all$time_spend_company[HR_all$time_spend_company>Tmax|HR_all$time_spend_company<Tmin]

POS=which(HR_all$time_spend_company>Tmax|HR_all$time_spend_company<Tmin)

round(mean(HR_all$time_spend_company),2) #mean is 3.5

HR_all$time_spend_company[POS]=3.5

#Work_accident:

glimpse(HR_all)

unique((HR_all$Work_accident))

round(prop.table(table(HR_all$Work_accident,HR_all$left)),2)

HR_all$Work_accident=as.numeric(HR_all$Work_accident==1)

#Promotion in last 5 yrs:

unique(HR_all$promotion_last_5years)

HR_all$promotion_last_5years=as.numeric(HR_all$promotion_last_5years)

#left:

HR_all$left=as.numeric(HR_all$left)

glimpse(HR_all)

#Checking NAs:

sum(is.na(HR_all)) #4500 NAs, these are of Test file

#Separating HR_Test and HR_Train:

HR_Train2=HR_all%>%filter(DATA=="train")


HR_Test2=HR_all%>%filter(DATA=="test")

HR_Test2=HR_Test2%>%select(-DATA)
HR_Test2=HR_Test2%>%select(-left)

HR_Train2=HR_Train2%>%select(-DATA)


glimpse(HR_Train2)

sum(is.na(HR_Train2))



#############################################################################################################################

#Model Building Process on Train2:

for_vif=lm(left~.,data = HR_Train2)

library(car)
vif(for_vif)

A=cor(HR_Train2) ## Sales_Grp1, Sales_Mgmt_Grp2 have correlation coff = -1,
                 # Salary. Low. Medium, Salary.High have correlation coff = -1
alias(for_vif)

# Removing Sales_Mgmt_Grp2 and Salary.High to eliminate Multi-colinearity

for_vif=lm(left~.-Sales_Mgmt_Grp2-Salary.High,data = HR_Train2)

vif(for_vif) # No Multi-colinearity found

summary(for_vif)

#Converting 'left' to factors inTrain data set

glimpse(HR_all)

HR_Train2$left=as.factor(HR_Train2$left)

glimpse(HR_Train2)

#Spliting Train2 in Train3 and Test3:

set.seed(2)
s=sample(1:nrow(HR_Train2),0.80*nrow(HR_Train2))

HR_Train3=HR_Train2[s,]
HR_Test3=HR_Train2[-s,]

# Random Forest:

library(randomForest)
library(cvTools)

param=list(mtry=c(5,4,11),
           ntree=c(50,100,500,700,200),
           maxnodes=c(70,90,40,50),
           nodesize=c(5,10,15))

Expanded.Param=expand.grid(param)

s=sample(1:nrow(Expanded.Param),36)

subset2=Expanded.Param[s,]

#Finding best parameters:

mycost_auc=function(left,lefthat){
  roccurve=pROC::roc(left,lefthat)
  score=pROC::auc(roccurve)
  
  return(score)
}

myauc=0

for(i in 1:36){
  
  print(paste0("iteration",i))
  
  
  parameters=subset2[i,]
  
  k=cvTuning(randomForest,left ~ . - Sales_Mgmt_Grp2 - Salary.High,data = HR_Train3,
             tuning = parameters,
             
             
             folds=cvFolds(nrow(HR_Train3),K=10,type = 'random'),
             
             cost=mycost_auc, seed = 2,
             
             predictArgs = list(type='prob'))
  
  score.this=k$cv[,2]
  if(score.this>myauc)
    
  {
    myauc=score.this
    
    best_params=parameters
    print(best_params)
  }}

# my_auc=0.8404277
#best params = mtry-5, ntree-50 maxnodes-90 nodesize-5 # auc= 84.04%, In sample=87.55, outsample=84.28, Train2=86.81



RFModel=randomForest(left ~ . - Sales_Mgmt_Grp2 - Salary.High,data = HR_Train3,ntree=50,
                     mtry=5,maxnodes=90,nodesize=5)   

RFPredict=predict(RFModel,newdata = HR_Train3,type = 'prob')[,2]   

pROC::roc(HR_Train3$left,RFPredict) #AUC=0.8755 in sample validation

RFPredictTest=predict(RFModel,newdata = HR_Test3,type = 'prob')[,2]   

pROC::roc(HR_Test3$left,RFPredictTest) #AUC=0.8428 in out sample validation,

#Building Model on Train2:

RFModelTrain2=randomForest(left ~ . - Sales_Mgmt_Grp2 - Salary.High,data = HR_Train2,ntree=50,
                           mtry=5,maxnodes=90,nodesize=5)

RFPredictTrain2=predict(RFModelTrain2,newdata = HR_Train2,type = 'prob')[,2]

pROC::roc(HR_Train2$left,RFPredictTrain2)# AUC=86.81, 85.87 for model built on entire Train2

RFPredictTest3=predict(RFModelTrain2,newdata = HR_Test3,type = 'prob')[,2]

pROC::roc(HR_Test3$left,RFPredictTest3) #auc=0.872, 86.93

# Making Final Prediction:

Final_Pred_Test2=predict(RFModelTrain2,newdata = HR_Test2,type = 'prob')[,2]

#Writing predictions in csv file:

write.csv(Final_Pred_Test2,'Mohana_Bhowmick_P4_part2.csv',row.names = F)

## Submission has been made above using Random Forest ########################################



# GBM:




library(gbm)
library(cvTools)

paramsGBM=list(interaction.depth=c(1:4),
               n.trees=c(100,500,700,200),
               shrinkage=c(0.1,.01,.001),
               n.minobsinnode=c(5,10,15))


Expanded.DBMParam=expand.grid(paramsGBM)


G=sample(1:nrow(Expanded.DBMParam),30)

subset3=Expanded.DBMParam[G,]

#Finding best parameters:

mycost_GBMauc=function(left,lefthat){
  roccurve=pROC::roc(left,lefthat)
  score=pROC::auc(roccurve)
  
  return(score)}

myauc_GBM=0

for(i in 1:30){
  
  print(paste0("iteration",i))
  
  
  parameters=subset3[i,]
  
  k=cvTuning(gbm,left~.-Sales_Mgmt_Grp2-Salary.High,data = HR_Train3,
             
             tuning = parameters,
             
             args = list(distribution='bernoulli'),
             
             folds=cvFolds(nrow(HR_Train3),K=10,type = 'random'),
             
             cost=mycost_GBMauc, seed = 2,
             
             predictArgs = list(type='response',n.trees=parameters$n.trees)
  )
  score.this=k$cv[,2]
  if(score.this>myauc_GBM)
    
  {
    myauc_GBM=score.this
    
    best_params=parameters
    print(best_params)}}

# Best params:


GBMModel=gbm(left~.-Sales_Mgmt_Grp2-Salary.High,data = HR_Train3,n.trees = 500,
             n.minobsinnode = 5,interaction.depth = 3,shrinkage = 0.01,distribution = 'bernoulli')

GBMPredict.IN=predict.gbm(GBMModel,newdata = HR_Train3,type = 'response')

pROC::roc(HR_Train3$left,GBMPredict.IN) #auc=84.25 for in sample validation

GBMPredict.Out=predict.gbm(GBMModel,newdata = HR_Test3,type = 'response')

# Rounding off probabilities:

GBMPredict.Out=round(GBMPredict.Out,2)

pROC::roc(HR_Test3$left,GBMPredict.Out) #auc= 84.57 for out sample validation

#Building model on entire Train2

GBMModel_Train2=gbm(left~.-Sales_Mgmt_Grp2-Salary.High,data = HR_Train2,n.trees = 500,
                    n.minobsinnode = 5,interaction.depth = 3,shrinkage = 0.01,distribution = 'bernoulli')

GBMPredict.Train2=predict.gbm(GBMModel_Train2,newdata = HR_Train2,type = 'response')

pROC::roc(HR_Train2$left,GBMPredict.Train2) #auc=84.26 for in sample validation


#Stacking:

Predictions=data.frame(RF.Pred=RFPredictTest,GBM.Pred=GBMPredict.Out, left=HR_Test3$left)

# Meta-algorithm:

Meta.RF=randomForest(factor(left)~.,data = Predictions,ntree=50,
                     mtry=5,maxnodes=90,nodesize=5)

Meta.Pred=predict(Meta.RF,newdata = Predictions,type = 'prob')[,2]

pROC::roc(Predictions$left,Meta.Pred) # In sample 0.8585

