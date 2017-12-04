# https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication

# https://medium.com/impactai/cnns-from-different-viewpoints-fab7f52d159c
# The matrix above is a weight matrix, just like the ones from traditional neural networks. However, this weight matrix has two special properties:
#   
#   The zeros shown in gray are untrainable. This means that they’ll stay zero throughout the optimization process.
# Some of the weights are equal, and while they are trainable (i.e. changeable), they must remain equal. These are called “shared weights”.
# 
# The zeros correspond to the pixels that the filter didn’t touch. Each row of the weight matrix corresponds to one application of the filter.


# kernel is 
k_ <- matrix(c(1,2,3,4), nrow = 2, byrow = TRUE)
k_

#  doubly block circulant matrix (
k <- matrix(c(1,2,0,3,4,0,0,0,0,
              0,1,2,0,3,4,0,0,0,
              0,0,0,1,2,0,3,4,0,
              0,0,0,0,1,2,0,3,4),
            nrow = 4,
            byrow = TRUE)
k
x <- 1:9
x_ <- matrix(x, nrow = 3, byrow = TRUE)
x_
k %*% x

(y1 <- 1*1 + 2*2 + 3*4 + 4*5)
(y2 <- 1*2 + 2*3 + 3*5 + 4*6)
(y3 <- 1*4 + 2*5 + 3*7 + 4*8)
(y4 <- 1*5 + 2*6 + 3*8 + 4*9)

y <- matrix(c(y1,y2,y3,y4), nrow = 2, byrow = TRUE)
y
