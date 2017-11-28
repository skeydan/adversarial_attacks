library(keras)
(K <- keras::backend())
library(tensorflow)
sess <- tf$Session()

source("load_frogs_ships.R")
source("functions.R")

model_name <- "frogs_ships_conv.h5"
model_exists <- TRUE

# train
model <- keras_model_sequential()

model %>%
  
  # Start with hidden 2D convolutional layer being fed 32x32 pixel images
  layer_conv_2d(
    filter = 32, kernel_size = c(3,3), padding = "same", 
    input_shape = c(32, 32, 3)
  ) %>%
  layer_activation("relu") %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # 2 additional hidden 2D convolutional layers
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  
  # Use max pooling once more
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(512) %>%
  layer_activation("relu") %>%
  
  # Outputs from dense layer are projected onto 10 unit output layer
  layer_dense(1) %>%
  layer_activation("sigmoid")

opt <- optimizer_rmsprop(lr = 0.0001, decay = 1e-6)

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = opt,
  metrics = "accuracy"
)

model %>% summary()

if (!model_exists) {
  model %>% fit(x = x_train_combined, y = y_train_combined, 
                validation_data = list(x_test_combined, y_test_combined),
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
model %>% predict_proba(x_test_combined[1:10, , , ])
model %>% predict_proba(x_test_combined[1000:1010, , , ])
model %>% predict_proba(x_test_combined) %>% summary()


# poor frog
poor_frog <- x_train_combined[9, , ,  ,drop = FALSE]
some_ship <- x_train_combined[5007, , ,  ,drop = FALSE]
poor_frog_img <- x_train_combined[9, , , ]
some_ship_img <- x_train_combined[5007, , , ] 

layout(1)
plot_as_image(poor_frog_img)
plot_as_image(some_ship_img)

model %>% predict_proba(poor_frog)
model %>% predict_proba(some_ship)


# method 3: FGSM

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

sgn_frog_grads <- sign(frog_grads)
sgn_frog_grads %>% hist()

adv <- poor_frog + sgn_frog_grads
model %>% predict_proba(adv)
model %>% predict_classes(adv)

scale_factor <- seq(0.1, 0.9, by=0.1)
sapply(scale_factor, function(x) model %>% predict_proba(poor_frog + sgn_frog_grads * x))

# ship
target <- model %>% predict_classes(some_ship)
target_variable = K$variable(target)

loss <- metric_binary_crossentropy(model$output, target_variable)
gradients <- K$gradients(loss, model$input) # gradient with respect to input
input <- model$input
output <- model$output
sess$run(tf$global_variables_initializer())
evaluated_gradients <- sess$run(gradients,
                                feed_dict = dict(input = some_ship,
                                                 output = model %>% predict_proba(some_ship)))
ship_grads <- evaluated_gradients[[1]]
hist(ship_grads)

sgn_ship_grads <- sign(ship_grads)
sgn_ship_grads %>% hist()

adv <- some_ship + 0.122 * sgn_ship_grads
adv <- ifelse(adv > 1, 1, adv)
adv <- ifelse(adv < 0, 0, adv)
adv[1, , , 2] %>% max()
adv[1, , , 2] %>% min()
model %>% predict_proba(adv)
model %>% predict_classes(adv)
plot_as_image(adv[1, , , ])

scale_factor <- seq(0.1,0.5, by=0.1)
sapply(scale_factor, function(x) model %>% predict_proba(some_ship + sgn_ship_grads * x))

