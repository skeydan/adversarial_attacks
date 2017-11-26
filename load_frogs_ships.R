library(EBImage)
library(abind)


cifar10 <- dataset_cifar10()
# airplane, automobile, bird, cat, deer, dog, frog , horse, ship, truck

x_train <- cifar10$train$x/255
x_test <- cifar10$test$x/255
y_train <- cifar10$train$y
y_test <- cifar10$test$y

is_frog <- cifar10$train$y == 6
x_train_frogs <- x_train[is_frog, , , ]
dim(x_train_frogs)
y_train_frogs <- rep(0, 5000)

is_frog <- cifar10$test$y == 6
x_test_frogs <- x_test[is_frog, , , ]
dim(x_test_frogs)
y_test_frogs <- rep(0, 1000)

is_ship <- cifar10$train$y == 8
x_train_ships <- x_train[is_ship, , , ]
dim(x_train_ships)
y_train_ships <- rep(1, 5000)

is_ship <- cifar10$test$y == 8
x_test_ships <- x_test[is_ship, , , ]
dim(x_test_ships)
y_test_ships <- rep(1, 1000)

#layout(matrix(1:20, 4, 5))

#for (i in 1:20) {
#  img <- transpose(Image(data = x_train_frogs[i, , , ], colormode = "Color"))
#  display(img, method="raster", all = TRUE)
#}
#for (i in 1:20) {
#  img <- transpose(Image(data = x_train_ships[i, , , ], colormode = "Color"))
#  display(img, method="raster", all = TRUE)
#}

# combine

x_train_combined <- abind(x_train_frogs, x_train_ships, along = 1)
dim(x_train_combined)

y_train_combined <- c(y_train_frogs, y_train_ships)
length(y_train_combined)

x_test_combined <- abind(x_test_frogs, x_test_ships, along = 1)
dim(x_test_combined)

y_test_combined <- c(y_test_frogs, y_test_ships)
length(y_test_combined)


#layout(matrix(1:20, 4, 5))
#for (i in 4991:5010) {
#  img <- transpose(Image(data = x_train_combined[i, , , ], colormode = "Color"))
#  display(img, method="raster", all = TRUE)
#}
#for (i in 991:1010) {
#  img <- transpose(Image(data = x_test_combined[i, , , ], colormode = "Color"))
#  display(img, method="raster", all = TRUE)
#}

# shuffle? should be done by keras...
#indices <- sample(nrow(x_train_combined))

