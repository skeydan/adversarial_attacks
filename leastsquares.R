library(lars)
library(matrix)
library(matrixcalc)
library(dplyr)

data(diabetes)
str(diabetes)
X <- diabetes$x
y <- diabetes$y
df <- data.frame(cbind(X,y))
str(df)

lm(y ~ . , data=df) %>% coefficients()

X_with_intercept <- cbind(rep(1, nrow(X)), X)
dim(X_with_intercept)

# naive with matrix inverse
# beta_hat <- solve(t(X_with_intercept) %*% X_with_intercept) %*% t(X_with_intercept) %*% y
beta_hat <- solve(t(X_with_intercept) %*% X_with_intercept, t(X_with_intercept) %*% y)
beta_hat

(y - X_with_intercept %*% beta_hat)^2 %>% mean()

system.time(
  for (i in 1:100000)
  solve(t(X_with_intercept) %*% X_with_intercept, t(X_with_intercept) %*% y))
system.time(
  for (i in 1:100000)
    solve(t(X_with_intercept) %*% X_with_intercept) %*% t(X_with_intercept) %*% y)

# lu
# steps
## A = LU
## LUx = b
## introduce y = Ux
## Ly = b  --> easy to solve because L is lower triangular; now we have y
## solve Ux = y --> --> again easy to solve because U is upper triangular
X_lu <- lu(X_with_intercept)
elu <- expand(X_lu)
all.equal(as.matrix(X_with_intercept), as.matrix(elu$P %*% elu$L %*% elu$U),
          check.attributes=FALSE)

lu_solve <- function(A, b) {
  A_lu <- expand(lu(A))
  L <- A_lu$L
  U <- A_lu$U
  P <- A_lu$P
  y <- forwardsolve(P %*% L, b)
  backsolve(U, y)
}

beta_hat <- lu_solve(X_with_intercept, y)
(y - X_with_intercept %*% beta_hat)^2 %>% mean()

# cholesky
# https://github.com/fastai/numerical-linear-algebra/blob/master/nbs/6.%20How%20to%20Implement%20Linear%20Regression.ipynb
# If $A$ has full rank, the pseudo-inverse $(A^TA)^{-1}A^T$ is a square, hermitian positive definite matrix. 
# The standard way of solving such a system is Cholesky Factorization, which finds upper-triangular R s.t. $A^TA = R^TR$.

# steps
# A'A = R'R
# A'Ax = A'b
# R'Rx = A'b
# y = Rx
# Ry = A'b ; solve for y
# Rx = y ; solve for x

cholesky_solve <- function(A, b) {
  R <- chol(t(A) %*% A)
  y <- backsolve(R, t(A) %*% b)
  backsolve(R,y)
}  
beta_hat <- cholesky_solve(X_with_intercept, y) 
(y - X_with_intercept %*% beta_hat)^2 %>% mean()

# qr
# steps
# A = QR  
# QRx = b
# Rx = Q'b  # Q inverse == Q transpose
# x = backsubstitute(R, Q'b)

qr_solve <- function(A, b) {
  A_qr <- qr(A)
  R <- qr.R(A_qr)
  Q <- qr.Q(A_qr)
  backsolve(R, t(Q) %*% b)
}

beta_hat <- qr_solve(X_with_intercept, y)
(y - X_with_intercept %*% beta_hat)^2 %>% mean()
system.time(
  for (i in 1:100000)
    qr_solve(X_with_intercept, y))
