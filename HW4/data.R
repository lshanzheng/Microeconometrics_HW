##########################################################################
## How to look upon government intervention of people's plan to pregnant? #
##########################################################################

library(ISLR)
library(glmnet)
library(caret)
rm(list = ls())

dat <- read.csv("dat.csv")
#需要变成哑变量的变量
factor_var=c("X2","X7","X10","X13","X25","X29","X30","X44","X46","X47","X49","X50")
length(factor_var)
#需要处理哑变量的的变量构成的矩阵
dat1=dat[,colnames(dat)%in%factor_var]
#为了后面变成虚拟变量，自动命名重复，这里对列重命名
colnames(dat1)=c("A","B","C","D","E","F","G","H","I","J","K","L")
dim(dat1)
#先变成因子类型
dat1=sapply(dat1,as.factor)
dim(dat1)
#变成虚拟变量
dat1=as.data.frame(model.matrix(~.-1,data.frame(dat1)))
dim(dat1)

#不需要处理的矩阵
dat2=dat[,!colnames(dat)%in%factor_var]
dim(dat2)

#合并数据
dat_new=cbind(dat2,dat1)
dim(dat_new)


dat_new$Y <- factor(dat_new$Y,levels=c(0,1),labels=c("FALSE","TRUE"))
summary(dat_new)


# Split into training and test data sets
set.seed(123)
n <- nrow(dat_new)
train <- sample(n,n/2)
dat_new.tr <- dat_new[train,]
dat_new.te <- dat_new[-train,]


###################################
# Logistic Regression: Full Model #
###################################
# Fit on training data
logitfit <- glm(Y ~.,dat_new.tr,family='binomial')
summary(logitfit)

# Predict on test data
p = predict(logitfit,dat_new.te,type="response")
logitpred = as.factor(p > 0.5)
table(logitpred,dat_new.te$Y,dnn=c("predicted","true"))
logiterr <- 1-mean(logitpred==dat_new.te$Y) #misclassification error rate
logiterr

##############################
# Logistic Regression: Lasso #
##############################
# Fit on training data
x <- model.matrix(Y ~.,dat_new.tr)[,-1] #no intercept 
y <- dat_new.tr$Y
lassofit.all <- glmnet(x,y,alpha=1,family="binomial") 
plot(lassofit.all,xvar="lambda")

# Cross Validation
cv.lasso <- cv.glmnet(x,y,alpha=1,family="binomial") 
plot(cv.lasso)

# Refit the model using optimal lambda 
lambda.star <- cv.lasso$lambda.min #alternatively, use the 1se rule: lambda.star <- cv.lasso$lambda.1se
lassofit.star <- glmnet(x,y,alpha=1,lambda=lambda.star,family="binomial") 
coef(lassofit.star)

# Predict on test data
newx <- model.matrix(Y ~.,dat_new.te)[,-1] 
lassopred <- predict(lassofit.star,newx,type="class")
table(lassopred,dat_new.te$Y,dnn=c("predicted","true"))
lassoerr <- 1-mean(lassopred==dat_new.te$Y)
lassoerr

#######
# KNN #
#######
# Fit on training data
knnfit <- train(Y ~.,data=dat_new.tr,method="knn",             #require("caret")
                trControl=trainControl(method="repeatedcv",repeats=3), #use repeated CV to choose K
                preProcess=c("center","scale"),tuneLength = 50)        #center and scale predictors before KNN
plot(knnfit)

# Predict on test data
knnpred <- predict(knnfit,dat_new.te)
table(knnpred,dat_new.te$Y,dnn=c("predicted","true"))
knnerr <- 1-mean(knnpred==dat_new.te$Y)
knnerr

