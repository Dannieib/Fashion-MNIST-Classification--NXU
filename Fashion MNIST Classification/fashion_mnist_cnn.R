library(keras)
library(tensorflow)
library(ggplot2)

#Loading and preprocess the Fashion MNIST dataset
print("Loading Fashion MNIST dataset...")
fashion_mnist <- dataset_fashion_mnist()

train_images <- fashion_mnist$train$x
train_labels <- fashion_mnist$train$y
test_images <- fashion_mnist$test$x
test_labels <- fashion_mnist$test$y

# Reshape and normalize the data
train_images <- array_reshape(train_images, c(60000, 28, 28, 1))
train_images <- train_images / 255

test_images <- array_reshape(test_images, c(10000, 28, 28, 1))
test_images <- test_images / 255

# Convert labels to categorical (one-hot encoding)
train_labels <- to_categorical(train_labels, 10)
test_labels <- to_categorical(test_labels, 10)

#Building the the CNN model
print("Building the CNN model...")
model <- keras_model_sequential() %>%
  
  # First Convolutional Layer
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Second Convolutional Layer
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Third Convolutional Layer
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  
  # Flatten Layer
  layer_flatten() %>%
  
  # Fully Connected Layer
  layer_dense(units = 64, activation = 'relu') %>%
  
  # Output Layer
  layer_dense(units = 10, activation = 'softmax')

# Compile the model
model %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

# Train the model
print("Training the model...")
history <- model %>% fit(
  train_images, train_labels,
  epochs = 10,
  batch_size = 64,
  validation_split = 0.2
)

#Evaluate the model on the test data
print("Evaluating the model...")
test_loss <- model %>% evaluate(test_images, test_labels)
print(paste("Test Accuracy:", test_loss$accuracy))

#Make predictions on two sample images
print("Making predictions...")
sample_images <- test_images[1:2, , , , drop = FALSE]
predictions <- model %>% predict(sample_images)
predicted_labels <- max.col(predictions) - 1

# Map predicted labels to class names
class_names <- c('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# Print predictions
for (i in 1:2) {
  print(paste("Image", i, "is predicted as:", class_names[predicted_labels[i]]))
}

#Visualize the sample images and predictions
print("Visualizing predictions...")
par(mfrow = c(1, 2))
for (i in 1:2) {
  image <- sample_images[i, , , 1]
  plot(as.raster(image, max = 1))
  title(main = paste("Predicted:", class_names[predicted_labels[i]]))
}