adjdata<-read.csv('germany_2016_with_features_adj.csv',stringsAsFactors = FALSE,header = TRUE,sep=",")
str(adjdata)
summary(adjdata$DE_wind_generation_actual)
library(corrplot)
corrplot(cor(adjdata[,2:29]), type = "upper", order = "original", tl.col = "black", tl.srt = 45)
library(psych)
n=nrow(adjdata)
ceiling(0.8*8760)
set.seed(1234)
trainindex <- sample(1:n, 7008)
trainset <- adjdata[trainindex,]
head(trainset)
testsed <- adjdata[-trainindex,]
fit<-lm(DE_wind_generation_actual~ ., data = trainset)
summary(fit)
fit_adj=step(fit,direction="backward")
summary(fit_adj)
lm_predict_data <- predict(fit_adj, testsed[,-1])
write.csv(lm_predict_data, 'lm_predict_data.csv', row.names = F)
actual<-testsed[,1]
rmse(actual,lm_predict_data)
install.packages("caret")
install.packages("ModelMetrics",dependencies = TRUE)
library(tidyverse)
library(lattice)
library(ModelMetrics)
library(caret)
set.seed(123)
train.control <- trainControl(method = "cv", number = 10)
model <- train(DE_wind_generation_actual~ ., data = adjdata, method = "lm",
               trControl = train.control)
# Define training control
set.seed(123) 
train.control <- trainControl(method = "cv", number = 5)
# Train the model
model <- train(DE_wind_generation_actual ~., data = trainset, method = "lm",
               trControl = train.control)
# Summarize the results
print(model)