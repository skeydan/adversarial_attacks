library(keras)
(K <- keras::backend())
library(tensorflow)
sess <- tf$Session()

source("load_frogs_ships.R")
source("functions.R")

model_name <- "frogs_ships_logistic.h5"
model_exists <- TRUE

# flatten for linear classifier

x_train_flat <- x_train_combined %>% apply(1, as.numeric)
dim(x_train_flat)
x_train_flat <- x_train_flat %>% t()
dim(x_train_flat)

x_test_flat <- x_test_combined %>% apply(1, as.numeric)
x_test_flat <- x_test_flat %>% t()
dim(x_test_flat)

### checks

x_train_flat[1:5000, 1:1024] %>% sum()
x_train_flat[1:5000, 1025:2048] %>% sum()
x_train_flat[1:5000, 2049:3072] %>% sum()

x_train_flat[5001:10000, 1:1024] %>% sum()
x_train_flat[5001:10000, 1025:2048] %>% sum()
x_train_flat[5001:10000, 2049:3072] %>% sum()

# train
model <- keras_model_sequential()

model %>%
  layer_dense(units = 1,
              input_shape = 3072,
              kernel_initializer = initializer_random_normal(stddev = 0.1)
             ) %>%
#  layer_batch_normalization() %>%
  layer_activation("sigmoid")

model %>% compile(optimizer = optimizer_sgd(lr = 0.001), 
                  loss = "binary_crossentropy",
                  metrics = "accuracy")
model %>% summary()

if (!model_exists) {
  model %>% fit(x = x_train_flat, y = y_train_combined, 
                validation_data = list(x_test_flat, y_test_combined),
                epochs = 500,
                batch_size = 10,
                callbacks = callback_early_stopping(patience = 2),
                shuffle = TRUE)
  model %>% save_model_hdf5(model_name)
} else {
  model <- load_model_hdf5(model_name)
}

model 
# sanity check
model %>% predict_proba(x_test_flat[1:10, ])
model %>% predict_proba(x_test_flat[1000:1010, ])
model %>% predict_proba(x_test_flat) %>% summary()


# poor frog
poor_frog <- x_train_flat[9, , drop = FALSE]
some_ship <- x_train_flat[5007, , drop = FALSE]
poor_frog_img <- x_train_flat[9, , ]
some_ship_img <- x_train_flat[5007, , ]

layout(1)
plot_as_image(poor_frog_img)
plot_as_image(some_ship_img)

model %>% predict_proba(poor_frog)
model %>% predict_proba(some_ship)

# method 1: add to image when weights positive, subtract when negative
# see http://karpathy.github.io/2015/03/30/breaking-convnets/
# x = [2, -1, 3, -2, 2, 2, 1, -4, 5, 1] // input
# w = [-1, -1, 1, -1, 1, -1, 1, 1, -1, 1] // weight vector
# If you do the dot product, you get -3. 
# Hence, probability of class 1 is 1/(1+e^(-(-3))) = 0.0474. 
# In every dimension where the weight is positive, we want to slightly increase the input (to get slightly more score).
# Conversely, in every dimension where the weight is negative, we want the input to be slightly lower (again, to get slightly more score).

# 
weights <- (model %>% get_weights())[[1]]
dim(weights)
dim(poor_frog)

hist(weights)
# want to move frog to ship class, which is > 0.5
# for negative weights, subtract from image; for positive weights, add to image

scale_factor <- 1:10
sapply(scale_factor, function(x) model %>% predict_proba(poor_frog + t(weights)/x))

adv <- poor_frog + t(weights)/5
model %>% predict_proba(adv)
model %>% predict_classes(adv)

# test rearrangements
# f9 <- x_train_combined[9, , , ]
# dim(f9)
# f9_flat <- f9
# dim(f9_flat) <- c(1, 3072)
# all(f9_flat[1, ] == poor_frog[1, ])
# model %>% predict_proba(f9_flat)
# model %>% predict_classes(f9_flat)
# adv <- f9_flat + t(weights)/7
# model %>% predict_proba(adv)
# model %>% predict_classes(adv)

adv_2d <- adv
dim(adv_2d) <- c(32,32,3)
plot_as_image(adv_2d)

# method 2: add (complete) gradient to image

# frog
target <- model %>% predict_classes(poor_frog)
target
target_variable = K$variable(target)
target_variable

loss <- metric_binary_crossentropy(model$output, target_variable)
gradients <- K$gradients(loss, model$input) # gradient with respect to input
get_grad_values <- K$`function`(list(model$input), gradients)
frog_grads <- get_grad_values(list(poor_frog))[[1]]
fivenum(frog_grads)
hist(frog_grads)

adv <- poor_frog + frog_grads
model %>% predict_proba(adv)
model %>% predict_classes(adv)

scale_factor <- seq(0.1,2 , by=0.1)
sapply(scale_factor, function(x) model %>% predict_proba(poor_frog + frog_grads * x))

# ship
input <- model$input
output <- model$output
target <- model %>% predict_classes(some_ship)
target_variable = K$variable(target)
sess$run(tf$global_variables_initializer())

loss <- metric_binary_crossentropy(model$output, target_variable)
gradients <- K$gradients(loss, model$input) # gradient with respect to input
evaluated_gradients <- sess$run(gradients,
                                feed_dict = dict(input = some_ship,
                                                 output = model %>% predict_proba(some_ship)))
ship_grads <- evaluated_gradients[[1]]
hist(ship_grads)

adv <- some_ship + ship_grads
model %>% predict_proba(adv)
model %>% predict_classes(adv)

scale_factor <- seq(1,1000 , by=100)
sapply(scale_factor, function(x) model %>% predict_proba(some_ship + ship_grads * x))


# 3
# use FGSM

grads <- frog_grads
sgn_grads <- sign(grads)
t(sgn_grads) %>% hist()


scale_factor <- seq(0.001,0.05, by=0.005)
sapply(scale_factor, function(x) model %>% predict_proba(poor_frog + sgn_grads * x))
