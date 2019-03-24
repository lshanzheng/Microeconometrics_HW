#################
## Wage Profile #
#################
## Code to accompany Lecture on 
## Regression
## Jiaming Mao (jmao@xmu.edu.cn)
## https://jiamingmao.github.io

rm(list=ls())
library(ISLR) # contains the data set 'Wage'
library(gam)
cg <- read.csv("cg.csv",header = TRUE)
attach(cg)
# Polynomial Regression
# ---------------------
fit = lm(RD ~ poly(Y1101a,4,raw=T)) # degree-4 polynomial
summary(fit)

## plotting
Y1101alims = range(Y1101a) 
Y1101a.grid = seq(from=Y1101alims[1], to=Y1101alims[2]) 
preds = predict(fit, newdata=list(Y1101a=Y1101a.grid), se=TRUE) 
se.bands = cbind(preds$fit + 2*preds$se.fit, preds$fit - 2*preds$se.fit)
plot(Y1101a, RD, xlim=Y1101alims, cex=.5, col="darkgrey") 
lines(Y1101a.grid, preds$fit, lwd=2, col="darkblue") 
matlines(Y1101a.grid, se.bands, lwd=1, col="darkblue", lty=3)
title("Degree-4 Polynomial")

# Piecewise Constant Regression
# -----------------------------
# cut(x,M) divides x into M pieces of equal length and generates the corresponding dummy variables
fit = lm(RD ~ 0 + cut(Y1101a,4)) # no intercept
summary(fit)

## plotting
preds = predict(fit, newdata=list(Y1101a=Y1101a.grid), se=TRUE) 
se.bands = cbind(preds$fit + 2*preds$se.fit, preds$fit - 2*preds$se.fit)
plot(Y1101a, RD, xlim=Y1101alims, cex=.5, col="darkgrey") 
lines(Y1101a.grid, preds$fit, lwd=2, col="darkgreen") 
matlines(Y1101a.grid, se.bands, lwd=1, col="darkgreen", lty=3)
title("Piecewise Constant")

# Cubic Spline and Natural Cubic Spline
# -------------------------------------
# bs() generates B-spline basis functions with specified degrees of polynomials and knots 
fit = lm(RD ~ bs(Y1101a,knots=c(7,10,14),degree=3)) # knots at age 25,40,60
summary(fit)

# ns() fits a natural cubic spline
fit2 = lm(RD ~ ns(Y1101a,knots=c(7,10,14)))

## plotting
preds = predict(fit, newdata=list(Y1101a=Y1101a.grid), se=TRUE) 
se.bands = cbind(preds$fit + 2*preds$se.fit, preds$fit - 2*preds$se.fit)
preds2 = predict(fit2, newdata=list(Y1101a=Y1101a.grid), se=TRUE) 
se2.bands = cbind(preds2$fit + 2*preds2$se.fit, preds2$fit - 2*preds2$se.fit)
plot(Y1101a, RD, xlim=Y1101alims, cex=.5, col="darkgrey") 
lines(Y1101a.grid, preds$fit, lwd=2, col="darkblue") 
matlines(Y1101a.grid, se.bands, lwd=1, col="darkblue", lty=3)
lines(Y1101a.grid, preds2$fit, lwd=2, col="darkorange") 
matlines(Y1101a.grid, se2.bands, lwd=1, col="darkorange", lty=3)
abline(v=c(7,10,14),lty=3) 
legend("topright", col=c("darkblue","darkorange"), lwd=2, legend=c("Cubic Spline","Natural Cubic Spline"), bty="n")
title("Cubic and Natural Cubic Spline")
