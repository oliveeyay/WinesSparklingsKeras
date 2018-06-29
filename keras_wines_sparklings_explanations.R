# Imports
library(keras)   # for working with neural nets
library(lime)    # for explaining models
library(magick)  # for preprocessing images
library(ggplot2) # for additional plotting

# Loading ImageNet VGG16 model
model <- application_vgg16(weights = "imagenet", include_top = TRUE)
model

# Loading Wines vs Sparklings Model
model2 <- load_model_hdf5(filepath = "/Users/ogoutay/Documents/Personal/git_perso/WinesSparklingsKeras/wines_sparklings_model.h5")
model2

# Loading images and ploting them with superpixels
wine_path = file.path("/Users/ogoutay/Documents/Personal/git_perso/WinesSparklingsKeras/explanation/wines/wine.jpg")
sparkling_path = file.path("/Users/ogoutay/Documents/Personal/git_perso/WinesSparklingsKeras/explanation/sparklings/sparkling.jpg")
plot_superpixels(wine_path, n_superpixels = 50, weight = 20)
plot_superpixels(sparkling_path, n_superpixels = 50, weight = 20)

# Image preparation function
image_prep <- function(x) {
  arrays <- lapply(x, function(path) {
    img <- image_load(path, target_size = c(224,224))
    x <- image_to_array(img)
    x <- array_reshape(x, c(1, dim(x)))
    x <- imagenet_preprocess_input(x)
  })
  do.call(abind::abind, c(arrays, list(along = 1)))
}

# ------------------ ImageNet Model ------------------
# Predictions on these two images
res <- predict(model, image_prep(c(wine_path, sparkling_path)))
imagenet_decode_predictions(res)

# Load labels and train explainer (from ImageNet Model)
model_labels <- readRDS(system.file('extdata', 'imagenet_labels.rds', package = 'lime'))
explainer <- lime(c(wine_path, sparkling_path), as_classifier(model, model_labels), image_prep)

# Explanation (from ImageNet Model)
explanation <- explain(c(wine_path, sparkling_path), explainer, 
                       n_labels = 2, n_features = 35,
                       n_superpixels = 35, weight = 10,
                       background = "white")
plot_image_explanation(explanation)
sparkling <- explanation[explanation$case == "sparkling.jpg",]
plot_image_explanation(sparkling)
# ------------------ End ImageNet Model ------------------

# ------------------ Wine vs Sparkling Model ------------------
# Predictions on these two images
test_datagen <- image_data_generator(rescale = 1/255)
test_image_files_path <- "/Users/ogoutay/Documents/Personal/git_perso/WinesSparklingsKeras/explanation"
test_generator = flow_images_from_directory(
  test_image_files_path,
  test_datagen,
  target_size = c(113, 270),
  class_mode = 'categorical')

predictions <- as.data.frame(predict_generator(model2, test_generator, steps = 1))
t(round(predictions, digits = 2))

# Image Preparation
image_prep2 <- function(x) {
  arrays <- lapply(x, function(path) {
    img <- image_load(path, target_size = c(113, 270))
    x <- image_to_array(img)
    x <- reticulate::array_reshape(x, c(1, dim(x)))
    x <- x / 255
  })
  do.call(abind::abind, c(arrays, list(along = 1)))
}

# Train explainer
explainer2 <- lime(c(wine_path, sparkling_path), as_classifier(model2), image_prep2)
explanation2 <- explain(c(wine_path, sparkling_path), explainer2, 
                        n_labels = 1, n_features = 20,
                        n_superpixels = 35, weight = 10,
                        background = "white")

# GGPLOT for block thresholds discovery
explanation2 %>% ggplot(aes(x = feature_weight)) + facet_wrap(~ case, scales = "free") + geom_density()

# Images explanation
plot_image_explanation(explanation2, display = 'block', threshold = 0.0022)
sparkling2 <- explanation2[explanation2$case == "sparkling.jpg",]
plot_image_explanation(sparkling2, display = 'block', threshold = -0.22)
# ------------------ End Wine vs Sparkling Model ------------------
