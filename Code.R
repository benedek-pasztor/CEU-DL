# unzip("test.zip")
# unzip("train.zip")

rm(list = ls())
library(keras)
library(tidyverse)
library(tidyr)
library(future)
library(furrr)
library(caret)


# sample(list.files("submission")

# file.copy(paste0("submission/nocactus/", list.files("submission/nocactus/", pattern = ".jpg")),
          # paste0("submission/cactus/", list.files("submission/nocactus/", pattern = ".jpg")))

# submission_cactusids <- sample(list.files("submission/", pattern = ".jpg"), 1500)
# file.copy(paste0("submission/", submission_cactusids), paste0("submission/cactus/", submission_cactusids))
# file.remove(paste0("submission/", submission_cactusids))
# 
# rest <- list.files("submission/", pattern = "jpg")
# file.copy(paste0("submission/", rest), paste0("submission/nocactus/", rest))
# file.remove(paste0("submission/", rest))


# file.copy(paste0("test/", list.files("test/", pattern = ".jpg")), 
#           paste0("submission/", list.files("test/", pattern = ".jpg")))
# file.remove(paste0("test/", list.files("test/", pattern = ".jpg")))
### take pictures to their correct dirs

## Train
# train_cactusids <- train_data$id[which(train_data$has_cactus == 1)]
# file.copy(paste0("train/", train_cactusids), paste0("train/cactus/", train_cactusids)) 
# file.remove(paste0("train/", train_cactusids))
# 
# train_nocactusids <- train_data$id[which(train_data$has_cactus == 0)]
# file.copy(paste0("train/", train_nocactusids), paste0("train/nocactus/", train_nocactusids)) 
# file.remove(paste0("train/", train_nocactusids))
# 
# 
# ## Creating test set
# test_cactus <- sample(list.files(paste0("train/cactus")), 2000)
# file.copy(paste0("train/cactus/", test_cactus),
#           paste0("test/cactus/", test_cactus))
# file.remove(paste0("train/cactus/", test_cactus))
# 
# test_nocactus <- sample(list.files(paste0("train/nocactus")), 900)
# file.copy(paste0("train/nocactus/", test_nocactus),
#           paste0("test/nocactus/", test_nocactus))
# file.remove(paste0("train/nocactus/", test_nocactus))

## Creating validation set
# validation_cactus <- sample(list.files(paste0("train/cactus")), 2000)
# file.copy(paste0("train/cactus/", validation_cactus),
#           paste0("validation/cactus/", validation_cactus))
# file.remove(paste0("train/cactus/", validation_cactus))
# 
# validation_nocactus <- sample(list.files(paste0("train/nocactus")), 900)
# file.copy(paste0("train/nocactus/", validation_nocactus),
#           paste0("validation/nocactus/", validation_nocactus))
# file.remove(paste0("train/nocactus/", validation_nocactus))


labels <- read_csv('train.csv')

train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

validation_datagen <- image_data_generator(rescale = 1/255)  

test_datagen <- image_data_generator(rescale = 1/255)  


image_size <- c(150, 150)
batch_size <- 50

train_generator <- flow_images_from_directory(
  file.path("train/"),  
  train_datagen,        
  target_size = image_size,
  batch_size = batch_size,
  class_mode = "binary"    
)

validation_generator <- flow_images_from_directory(
  file.path("validation/"),   
  validation_datagen,
  target_size = image_size,
  batch_size = batch_size,
  class_mode = "binary"
)

test_generator <- flow_images_from_directory(
  file.path("test/"),
  test_datagen,
  target_size = image_size,
  batch_size = batch_size,
  class_mode = "binary"
)

model <- keras_model_sequential()

model %>%
  
  layer_conv_2d(
    filter = 32, kernel_size = c(3,3), padding = "same", 
    input_shape = c(150, 150, 3)
  ) %>%
  layer_activation("relu") %>%
  
  layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  layer_flatten() %>%
  layer_dense(512) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(1) %>%
  layer_activation("sigmoid")

opt <- optimizer_rmsprop(lr = 0.0001, decay = 1e-6)


model %>% compile(
  loss = "binary_crossentropy",
  optimizer = opt,
  metrics = "accuracy"
)


history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 2000 / batch_size,
  epochs = 20,
  validation_data = validation_generator,
  validation_steps = 50
)


### Making submission file
subm_datagen <- image_data_generator(rescale = 1/255)


submission_generator <- flow_images_from_directory(
  file.path("submission/"),
  subm_datagen,
  target_size = image_size,
  batch_size = 1,
  class_mode = NULL,
  shuffle = F
)

predictions <- model %>% 
  predict_generator(submission_generator, steps = 4000)


##x meint más
submission <- data.frame(id = paste0(list.files(paste0("submission/", c("cactus/")), pattern = ".jpg")),
                       has_cactus = predictions)

submission <- submission %>% mutate(
  has_cactus = round(has_cactus, 0) 
)


write_csv(submission, 'submission2.csv')
