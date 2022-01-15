#Import Libraries and data

library(tidyverse)
library(ggplot2)
library(GGally)
library(dplyr)
library(plyr)
library(grid)
library(gridExtra)
library(caret)
library(e1071)
library(readr)
ks_projects_201612 <- read_csv("ks-projects-201612.csv")
View(ks_projects_201612)


#rename data set for ease
df <- ks_projects_201612

#Quick analysis of columns imported
head(df)
summary(df)

df <- df[, -c(14,15,16, 17)]  #Remove empty rows
df <- na.omit(df) #Omit N/A Values 
str(df)# Quick analysis of data set

newdf <- transform(df, goal = as.numeric(goal), #Fix Data types 
              pledged = as.numeric(pledged),
              backers = as.numeric(backers),
              deadline = as.Date(deadline),
              launched = as.Date(launched))

#Filter to only 2 values, cleans shifted data and make for easy computing
df <- dplyr::filter(df, df$state == 'successful'| df$state == 'failed') 



#Data exploration plotting
ggplot(df, aes(state)) + 
  geom_bar()

ggplot(df, aes(state)) + 
  geom_bar()

ggplot(df, aes(category)) + 
  geom_histogram()

ggplot(df, aes(currency)) + 
  geom_bar()

ggplot(df, aes(country)) + 
  geom_bar()

ggplot(newdf, aes(goal)) + 
  geom_boxplot()

ggplot(newdf, aes(pledged)) + 
  geom_boxplot()

ggplot(newdf, aes(backers)) + 
  geom_boxplot()

newdf = subset(newdf, select = -usd.pledged) #remove unneeded column

USAdf <- filter(newdf, currency == 'USD') # filter to only businesses raised money in USD
USAdf <- na.omit(USAdf) #Omit NA values 

#reassign values as 0 and 1 for easier computation
USAdf$state <- revalue(USAdf$state, c("successful"= 1))
USAdf$state <- revalue(USAdf$state, c("failed"= 0))

#set state column as numeric data type
USAdf <- transform(USAdf, state = as.numeric(state))

#further data exploration
ggplot(USAdf, aes(x=launched))+
  geom_line(stat = "count", size = 1)+
  theme_minimal()

ggplot(USAdf, aes(x=deadline))+
  geom_line(stat = "count", size = 1)+
  theme_minimal()

#rid data set of numeric outliers
USAdf <- filter(USAdf, goal < 10000 & backers < 200)

#final data exploration / outlier check
ggplot(USAdf, aes(backers)) + 
  geom_boxplot()

ggplot(USAdf, aes(goal)) + 
  geom_boxplot()

ggplot(USAdf, aes(pledged)) + 
  geom_boxplot()

# reorganising columns and get rid of uneeded columns
col_order <- c("state", "goal", "pledged", "backers", "category", "main_category",
               "currency", "deadline", "launched", "country")
USAdf <- USAdf[, col_order]
USAdf <- na.omit(USAdf)


BizClusters <- kmeans(USAdf[,2:3], 3, nstart = 20) # create k means clustering model
BizClusters #View raw data clusters 
BizClusters$cluster <- as.factor(BizClusters$cluster) # set clusters as factors

ggplot(USAdf, aes(goal, pledged, color = BizClusters$cluster)) + geom_point() # plot cluster visualisation

BizNorm <- USAdf[,2:3] # set needed columns for clustering

# set numeric values
bss <- numeric()
wss <- numeric()

# create k means clusters for the number of clusters 1 to 10, creating 10 dfferent sets of clusters
# also save the betweenss and tot.withinss for each in the bss and wss variables
for(i in 1:10){
  # For each k, calculate betweenes and tot.withinnes
  bss[i] <- kmeans(BizNorm, centers=i)$betweenss
  wss[i] <- kmeans(BizNorm, centers=i)$tot.withinss
}

# plot the betweennes from clustering
p3 <- qplot(1:10, bss, geom=c("point", "line"),
            xlab="Number of clusters", ylab="Between-cluster sum of squar
es") +
  scale_x_continuous(breaks=seq(0, 10, 1)) +
  theme_bw()

#plot withinness from clustering
p4 <- qplot(1:10, wss, geom=c("point", "line"),
            xlab="Number of clusters", ylab="Total within-cluster sum of
squares") +
  scale_x_continuous(breaks=seq(0, 10, 1)) +
  theme_bw()

#put betweennes and withiness plots next to each other 
grid.arrange(p3, p4, ncol=2)

#create new k means clustering model with efficient parameters 
BizClustersk2 <- kmeans(USAdf[,3:4], 3, nstart = 20)

# create a plot to show clustering betwwen different columns using efficient clustering model.
BizNum <- USAdf[2:4]
ggpairs(cbind(BizNum, Cluster=as.factor(BizClustersk2$cluster)),
        columns=1:3, aes(colour=Cluster, alpha=0.5),
        lower=list(continuous="points"),
        upper=list(continuous="blank"),
        axisLabels="none", switch="both") +
  theme_bw()


keepList <- c("goal", "pledged", "state", "backers") # set columns to be used in SVM
USAsvmDF <- USAdf[keepList]

#sample data for SVM model
USAsvmDF <- USAsvmDF[sample(nrow(USAsvmDF), 200), ]

#partition data for training and testing sets
intrain <- createDataPartition(y= USAsvmDF$state, p= 0.7, list = FALSE)
training <- USAsvmDF[intrain,]
testing <- USAsvmDF[-intrain,]

#make state column a factor to be computed 
training[["state"]] <- factor(training[["state"]])

#design control function
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

#build svm model 
svm_linear <- train(state ~., data = training, method = "svmLinear", trControl = trctrl, preProcess = c("center", "scale"), tuneLength = 10)

svm_linear

#pass test data to svm model
test_pred <- predict(svm_linear, newdata = testing)
test_pred

#plot confusion matrix fro test
confusionMatrix(table(test_pred, testing$state))

# create list of possible parameters 
grid <- expand.grid(C = c(0,0.01,0.05,0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2,5))

#create svm model with multiple parameters to find efficient parameters and plot
svm_Linear_Grid <- train(state ~., data = training, method = "svmLinear", trControl = trctrl, preProcess = c("center", "scale"), tuneGrid = grid,tunelength = 10)
plot(svm_Linear_Grid)

# test efficient model with testing data and plot most efficient results
test_pred_grid <- predict(svm_Linear_Grid, newdata = testing)
confusionMatrix(table(test_pred_grid, testing$state))
