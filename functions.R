
plot_as_image <- function(img_array) {
  Image(data = img_array, colormode = "Color") %>%
    transpose() %>%
    display(method="raster", all = TRUE)
}
