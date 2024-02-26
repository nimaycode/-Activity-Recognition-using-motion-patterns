---
title: "Activity Recognition using motion patterns"
author: "Nimay Srinivasan"
output:
  html_notebook:
    toc: yes
    toc_float: yes
  html_document:
    toc: yes
    df_print: paged
---
```{r}  
library(keras)
library(dplyr)
library(caret)
```

```{r} 
rm(list=ls())

# Set working directory as needed
setwd("/Users/nimaysrinivasan/CS-422/CS-422 Homework 7")
df <- read.csv("activity-small.csv")

# Seed the PRNG
set.seed(1122)
df <- df[sample(nrow(df)), ] # Shuffle, as all of the data in the .csv file
                             # is ordered by label!  This will cause problems
                             # if we do not shuffle as the validation split
                             # may not include observations of class 3 (the
                             # class that occurs at the end).  The validation_
                             # split parameter samples from the end of the
                             # training set.

# Scale the dataset.  Copy this block of code as is and use it; we will get
# into the detail of why we scale.  We will scale our dataset so all of the
# predictors have a mean of 0 and standard deviation of 1.  Scale test and
# training splits independently!

indx <- sample(1:nrow(df), 0.20*nrow(df))
test.df  <- df[indx, ]
train.df <- df[-indx, ]

label.test <- test.df$label
test.df$label <- NULL
test.df <- as.data.frame(scale(test.df))
test.df$label <- label.test
rm(label.test)

label.train <- train.df$label
train.df$label <- NULL
train.df <- as.data.frame(scale(train.df))
train.df$label <- label.train
rm(label.train)
rm(indx)
```
# --- Your code goes below ---

```{r} 
X_train <- select(train.df, -label)
y_train <- train.df$label

y_train.ohe <- to_categorical(y_train)

X_test <- select(test.df, -label)
y_test <- test.df$label
y_test.ohe <- to_categorical(test.df$label)

fit_model <- function(x, y, s, a, n,...) {
  model <- keras_model_sequential() %>%
    layer_dense(units = n, activation=a, input_shape=c(3)) %>%
    layer_dense(units = 4, activation="softmax")
    
  model %>% 
    compile(loss = "categorical_crossentropy", 
            optimizer="adam", 
            metrics=c("accuracy"))
  
  model %>% fit(
    data.matrix(x), 
    y,
    epochs=100,
    batch_size=s,
    validation_split=0.20
  )
  return (model)
}

fit_model_additional_hidden <- function(x, y, s, a,n1, n2,...) {
  model <- keras_model_sequential() %>%
    layer_dense(units = n1, activation=a, input_shape=c(3)) %>%
    layer_dense(units = n2, activation=a, input_shape=c(3)) %>%
    layer_dense(units = 4, activation="softmax")
    
  model %>% 
    compile(loss = "categorical_crossentropy", 
            optimizer="adam", 
            metrics=c("accuracy"))
  
  model %>% fit(
    data.matrix(x), 
    y,
    epochs=100,
    batch_size=s,
    validation_split=0.20
  )
  return (model)
}

confusion_matrix <- function(X_train, y_train, model) {
  model %>% evaluate(as.matrix(X_test), y_test.ohe)
  pred.prob <- predict(model, as.matrix(X_test))
  pred.class <- apply(pred.prob, 1, function(x) which.max(x)-1)
  matrix <- confusionMatrix(as.factor(pred.class), as.factor(y_test))
  return (matrix)
}
```

```{r}
# 2.1A
# batch size 1
begin <- Sys.time()
batch_size = 1
model = NULL
model = fit_model(x = X_train, y = y_train.ohe, s=batch_size, a="relu", n=8) 
end <- Sys.time()
cat("Batch size: ")
print(batch_size)
cat("\n")
cat("Time taken to train the neural network:")
print(end - begin, digits=2)
cat("\n")
matrix <- confusion_matrix(X_train, y_train, model)
cat("Overall accuracy:\n")
matrix[3]
cat("Matrix:\n")
matrix[4]
```

```{r}
# 2.1B
bSize <- c(1,32,64,128,256)
time <- c()
matrices <- c()
for(batch_size in bSize) {
  begin <- Sys.time()
  model = NULL
  model = fit_model(x = X_train, y = y_train.ohe, s=batch_size, a="relu",n=8) 
  end <- Sys.time()
  
  time <- c(time, end-begin)
  matrix <- confusion_matrix(X_train, y_train, model)
  matrices <- c(matrices, matrix)
}

index <- rep(1:5)
matrices_index <- c(4,10,16,22,28)
for(i in index) {
  cat("Batch size: ")
  print(bSize[i])
  cat("\n")
  cat("Time taken to train the neural network:")
  print(time[i], digits=2)
  cat("\n")
  cat("Overall accuracy:\n")
  print(matrices[matrices_index[i]-1])
  cat("Matrix:\n")
  print(matrices[matrices_index[i]])
}
```

```{r}
# 2.1D
# Case 1: relu with 2 hidden layer wuth units 8 and 9
bSize <- c(256)
time <- c()
matrices <- c()
for(batch_size in bSize) {
  begin <- Sys.time()
  model = NULL
  model = fit_model_additional_hidden(x = X_train, y = y_train.ohe, s=batch_size, a="relu", n1=8, n2=9) 
  end <- Sys.time()
  
  time <- c(time, end-begin)
  matrix <- confusion_matrix(X_train, y_train, model)
  matrices <- c(matrices, matrix)
}

index <- rep(1:1)
matrices_index <- c(4)
for(i in index) {
  cat("Batch size: ")
  print(bSize[i])
  cat("\n")
  cat("Time taken to train the neural network:")
  print(time[i], digits=2)
  cat("\n")
  cat("Overall accuracy:\n")
  print(matrices[matrices_index[i]-1])
  cat("Matrix:\n")
  print(matrices[matrices_index[i]])
}

# Case 2: tanh with 2 hidden layer with unit 8 and 9
bSize <- c(256)
time <- c()
matrices <- c()
for(batch_size in bSize) {
  begin <- Sys.time()
  model = NULL
  model = fit_model_additional_hidden(x = X_train, y = y_train.ohe, s=batch_size, a="tanh", n1=9, n2=8) 
  end <- Sys.time()
  
  time <- c(time, end-begin)
  matrix <- confusion_matrix(X_train, y_train, model)
  matrices <- c(matrices, matrix)
}

index <- rep(1:1)
matrices_index <- c(4)
for(i in index) {
  cat("Batch size: ")
  print(bSize[i])
  cat("\n")
  cat("Time taken to train the neural network:")
  print(time[i], digits=2)
  cat("\n")
  cat("Overall accuracy:\n")
  print(matrices[matrices_index[i]-1])
  cat("Matrix:\n")
  print(matrices[matrices_index[i]])
}

# Case 3: relu with 1 hidden layer with unit 9
bSize <- c(256)
time <- c()
matrices <- c()
for(batch_size in bSize) {
  begin <- Sys.time()
  model = NULL
  model = fit_model(x = X_train, y = y_train.ohe, s=batch_size, a="relu", n=9) 
  end <- Sys.time()
  
  time <- c(time, end-begin)
  matrix <- confusion_matrix(X_train, y_train, model)
  matrices <- c(matrices, matrix)
}

index <- rep(1:1)
matrices_index <- c(4)
for(i in index) {
  cat("Batch size: ")
  print(bSize[i])
  cat("\n")
  cat("Time taken to train the neural network:")
  print(time[i], digits=2)
  cat("\n")
  cat("Overall accuracy:\n")
  print(matrices[matrices_index[i]-1])
  cat("Matrix:\n")
  print(matrices[matrices_index[i]])
}

# Case 4: relu with 2 hidden layers and with unit for first hidden layer is 8 and second is 10
bSize <- c(32)
time <- c()
matrices <- c()
for(batch_size in bSize) {
  begin <- Sys.time()
  model = NULL
  model = fit_model_additional_hidden(x = X_train, y = y_train.ohe, s=batch_size, a="relu", n1=9, n2=10) 
  end <- Sys.time()
  
  time <- c(time, end-begin)
  matrix <- confusion_matrix(X_train, y_train, model)
  matrices <- c(matrices, matrix)
}

index <- rep(1:1)
matrices_index <- c(4)
for(i in index) {
  cat("Batch size: ")
  print(bSize[i])
  cat("\n")
  cat("Time taken to train the neural network:")
  print(time[i], digits=2)
  cat("\n")
  cat("Overall accuracy:\n")
  print(matrices[matrices_index[i]-1])
  cat("Matrix:\n")
  print(matrices[matrices_index[i]])
}

# Case 5: tanh with 1 hidden layer
bSize <- c(32)
time <- c()
matrices <- c()
for(batch_size in bSize) {
  begin <- Sys.time()
  model = NULL
  model = fit_model(x = X_train, y = y_train.ohe, s=batch_size, a="tanh", n=8) 
  end <- Sys.time()
  
  time <- c(time, end-begin)
  matrix <- confusion_matrix(X_train, y_train, model)
  matrices <- c(matrices, matrix)
}

index <- rep(1:1)
matrices_index <- c(4)
for(i in index) {
  cat("Batch size: ")
  print(bSize[i])
  cat("\n")
  cat("Time taken to train the neural network:")
  print(time[i], digits=2)
  cat("\n")
  cat("Overall accuracy:\n")
  print(matrices[matrices_index[i]-1])
  cat("Matrix:\n")
  print(matrices[matrices_index[i]])
}
```
