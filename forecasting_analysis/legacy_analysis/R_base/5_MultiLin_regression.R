#### TO ANALYZE INFLUENCE OF ENVIRONMENTAL VARIABLES ON DAM MOVEMENTS ####
# author: Jonas Ziemer (author)
# 30.10.2023

# load
library(dplyr)
library(readr)
library(tidyr)
library(ggplot2)
library(stringr)
library(writexl)
library(rlang)
library(tidyverse)
library(ggpubr)
library(ggpmisc)
library(ggcorrplot)
library(relaimpo)
library(Metrics)

setwd("C:/Users/ni82xoj/Desktop/KI4KI/Daten/13_In_Situ_Ruhrverband/")
gloer = read_csv('Gloer_MultiReg.csv', show_col_types = FALSE)
fuerwigge = read_csv('Fuerwigge_MultiReg.csv', show_col_types = FALSE)
moehne = read_csv('Moehne_MultiReg.csv', show_col_types = FALSE)
ennepe1 = read_csv('Ennepe_MultiReg_Lot1.csv', show_col_types = FALSE)
ennepe2 = read_csv('Ennepe_MultiReg_Lot2.csv', show_col_types = FALSE)


# Read the CSV file into a data frame
# data <- read.csv("C:/Users/ni82xoj/Desktop/KI4KI/Daten/13_In_Situ_Ruhrverband/Gloer_MultiReg.csv", dec=".", sep=",")  # Replace with your file path
# data <- read.csv("C:/Users/ni82xoj/Desktop/KI4KI/Daten/13_In_Situ_Ruhrverband/Fuerwigge_MultiReg.csv", dec=".", sep=",")  # Replace with your file path
data <- read.csv("C:/Users/ni82xoj/Desktop/KI4KI/Daten/13_In_Situ_Ruhrverband/Moehne_MultiReg.csv", dec=".", sep=",")  # Replace with your file path
# data <- read.csv("C:/Users/ni82xoj/Desktop/KI4KI/Daten/13_In_Situ_Ruhrverband/Ennepe_MultiReg_Lot1.csv", dec=".", sep=",")  # Replace with your file path
# data <- read.csv("C:/Users/ni82xoj/Desktop/KI4KI/Daten/13_In_Situ_Ruhrverband/Ennepe_MultiReg_Lot2.csv", dec=".", sep=",")  # Replace with your file path

# Convert the date column to a Date object (assuming the date column is named "Date")
data$Date <- as.Date(data$Date, format = "%m/%d/%Y")

# Specify the reference date
reference_date <- as.Date("2015-01-01")

# Calculate the day count from the reference date
data$DayCount <- as.numeric(difftime(data$Date, reference_date, units = "days")) + 1
# Temp_norm <- (data$Temperature-min(data$Temperature))/(max(data$Temperature)-min(data$Temperature))

# data <- data %>% mutate(Temp_norm = (data$Temperature-min(data$Temperature))/(max(data$Temperature)-min(data$Temperature)))

# Filter the dataframe to select dates from 1/1/2015 to the year 2020
data <- data[data$Date >= as.Date("2015-01-01") & data$Date <= as.Date("2019-12-31"), ]
# data <- data[data$Date >= as.Date("2020-01-01") & data$Date <= as.Date("2020-12-31"), ]

# data <- data[data$Date >= as.Date("2013-01-01") & data$Date <= as.Date("2017-11-01"), ]
# data <- data[data$Date >= as.Date("2015-01-01") & data$Date <= as.Date("2017-11-01"), ]
# data <- data[data$Date >= as.Date("2019-01-01") & data$Date <= as.Date("2021-01-01"), ]
# data <- data[data$Date >= as.Date("2021-01-01") & data$Date <= as.Date("2023-04-19"), ]

# Delete rows with missing values
data <- na.omit(data)


# Check which columns have missing values
# columns_with_missing_values <- sapply(data, function(x) any(is.na(x)))
# Print the columns with missing values
# print(names(columns_with_missing_values[columns_with_missing_values]))
# Check the rows with missing values
# rows_with_missing_values <- data[rowSums(is.na(data)) > 0, ]
# Print the rows with missing values
# print(rows_with_missing_values)

# List of dates to be excluded
# dates_to_exclude <- as.Date(c("2021-02-08", "2021-02-09", "2021-02-10", "2021-02-11", "2021-02-12", "2021-02-13"))  # Replace with the dates you want to exclude
# Filter the dataframe to exclude the specified dates
# data <- data[!(data$Date %in% dates_to_exclude), ]

# lm1 = lm(formula = RAD_Lot ~ Waterlevel + Temperature + Seepage +  Precip + Precip_d10_mean + DayCount, data = data)
# lm1 = lm(formula = RAD_Lot ~ Waterlevel + Temperature + Precip + DayCount, data = data)
# lm1 = lm(formula = RAD_Lot ~ Waterlevel + Temperature + Precip_d10_mean + DayCount, data = data)
# lm1 = lm(formula = RAD_Lot ~ Waterlevel + Temperature + Precip + Precip_d10_mean + DayCount, data = data)
# lm1 = lm(formula = RAD_Lot ~ Waterlevel + Temp_norm + DayCount, data = data)
# lm1 = lm(formula = RAD_Lot ~ Waterlevel + Temperature + DayCount, data = data)
# lm1 = lm(formula = RAD_Lot ~ Precip, data = data)
lm1 = lm(formula = RAD_Lot ~ Waterlevel + Temperature + DayCount, data = data)

# Get the model residuals
model_residuals = lm1$residuals

# Plot the result
hist(model_residuals)
mean(abs(model_residuals))

# Reduce the data frame by excluding columns RAW_radial and Date
reduced_data <- data[, !names(data) %in% c("RAW_radial", "Date")]
# reduced_data <- gloer[, !names(gloer) %in% c("Date", "RAD_Schwimm")]
# reduced_data <- ennepe1[, !names(ennepe1) %in% c("Date", "RAW_radial")]
# reduced_data <- ennepe2[, !names(ennepe2) %in% c("Date", "RAW_radial")]

# Compute correlation at 2 decimal places
corr_matrix = round(cor(reduced_data), 2)

# Compute and show the  result
ggcorrplot(corr_matrix, hc.order = TRUE, type = "lower",
           lab = TRUE)

summary(lm1)

coefficients(lm1) # model coefficients
# confint(lm1, level=0.95) # CIs for model parameters

# fitted(lm1) # predicted values
# residuals(lm1) # residuals
anova(lm1) # anova table
vcov(lm1) # covariance matrix for model parameters
cov2cor(vcov(lm1)) # correlation matrix for model parameters
# influence(lm1) # regression diagnostics

# diagnostic plots
layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page
# layout(matrix(c(1),1,1)) # optional 4 graphs/page
plot(lm1)

# Calculate Relative Importance for Each Predictor
# calc.relimp(lm1,type=c("lmg", "last","first","pratt", "genizi", "betasq"),
#             rela=TRUE)
calc.relimp(lm1,type=c("lmg", "pratt", "genizi", "betasq"),
            rela=TRUE)

# Bootstrap Measures of Relative Importance (1000 samples)
# boot <- boot.relimp(lm1, b = 1000, type = c("lmg", "last", "first", "pratt", "genizi", "betasq"), rank = TRUE,
#                     diff = TRUE, rela = TRUE)
boot <- boot.relimp(lm1, b = 1000, type = c("lmg", "pratt", "genizi", "betasq"), rank = TRUE,
                    diff = TRUE, rela = TRUE)

booteval.relimp(boot) # print result

# Sort results
sorted_results <- booteval.relimp(boot)
# Plot the sorted results
plot(sorted_results)


################################################
coefficients(lm1) # model coefficients

# Calculate deformation for each day using the coefficients
data$Deformation <- with(data, 4.544543e-01*Waterlevel-2.393864e-01*Temperature-4.609715e-04*DayCount-9.281964e+01)
# data$Deformation <- with(data, 4.973391e-01*Waterlevel-2.484093e-01*Temperature-1.025478e-02*Precip-5.937679e-05*DayCount-1.020676e+02)


# data$Deformation <- with(data, 4.525102e-01*Waterlevel-2.406788e-01*Temperature-2.026144e-03*Precip-1.168604e-01*Precip_d10_mean-4.676761e-04*DayCount-9.213344e+01)
# data$Deformation <- with(data, 4.544543e-01*Waterlevel-2.393864e-01*Temperature-4.609715e-04*DayCount-9.281964e+01)
# data$Deformation <- with(data, 0.268545293*Waterlevel-0.270176298*Temperature-28.750451282*Seepage+0.001739155*DayCount-0.055470353*Precip+0.174920223*Precip_d10_mean-76.199675128)
# data$Deformation <- with(data, 8.763418e-01*Waterlevel-1.138036e+01*Temp_norm-3.068925e-04*DayCount-3.758614e+02)
def1 = (8.763418e-01*437.52)-(2.918040e-01*(-10.6))-(3.068925e-04*1155)-3.793047e+02
def2 = (8.763418e-01*437.52)-1.138036e+01*(0.03076923)-(3.068925e-04*1155)-3.758614e+02

mean(data$Deformation)
sd(data$Deformation)
boxplot.stats(data$Deformation)
stdev95 <- sd(data$Deformation)*1.96 # 95% Confidence
# stdev <- sd(data$RAD_Lot)

p <- ggplot(data, aes(x = Date, y = Deformation)) +
  geom_point(color="grey") +
  geom_point(data=data, aes(x=Date, y=RAD_Lot), color = "orange", size=2) +
  # geom_errorbar(aes(ymin=Deformation-stdev95, ymax=Deformation+stdev95)) +
  # geom_errorbar(aes(ymin=Deformation-rmse95, ymax=Deformation+rmse95)) +
  labs(x = "Date", y = "Deformation")
  # ggtitle("Deformation Over Time") 
  # annotate("text", x = data$Date, y = data$Deformation,
  #          label = ifelse(data$RAD_Lot < data$Deformation - stdev95 |
  #                           data$RAD_Lot > data$Deformation + stdev95, "False", ""),
  #          hjust = -0.2, vjust = 0.5)
  # annotate("text", x = data$Date, y = data$Deformation,
  #          label = ifelse(data$RAD_Lot < data$Deformation - rmse95 | 
  #                           data$RAD_Lot > data$Deformation + rmse95, "False", ""),
  #          hjust = -0.2, vjust = 0.5)

# Adjust size of axis labels and ticks
p + theme(axis.text.x = element_text(size = 13),  # Adjust size of x-axis labels
          axis.text.y = element_text(size = 13),  # Adjust size of y-axis labels
          axis.title.x = element_text(size = 15), # Adjust size of x-axis title
          axis.title.y = element_text(size = 15)) # Adjust size of y-axis title

# Streudiagramm mit Regressionslinie erstellen
plot(data$Date, data$RAD_Lot, col = "grey", xlab = "Time", ylab = "Deformation in mm", pch=4, ylim=c(-5.5,3))
lines(data$Date, data$Deformation, col = "orange", pch = 19, lwd = 3)
legend("bottomleft", legend = c("Predicted", "True"), col = c("orange", "grey"), pch = c(19, 4), cex = 0.8)

# Create outlier range for stdev
data$Outlier <- with(data, data$RAD_Lot >= data$Deformation - stdev95 & data$RAD_Lot <= data$Deformation + stdev95)
# Create outlier range for rmse 
# data$Outlier <- with(data, data$RAD_Lot >= data$Deformation - rmse95 & data$RAD_Lot <= data$Deformation + rmse95)
# Count the number of false values
count_false_values <- sum(!data$Outlier)
# Print the count of false values
print(count_false_values)


layout(matrix(c(1),1,1)) # optional 4 graphs/page
lm2 <- lm(RAD_Lot ~ Deformation, data = data)
plot(data$RAD_Lot ~ data$Deformation, xlab="MLR Deformation Prediction (mm)", ylab="Plumb Deformation (mm)")
grid(lty = 6, col = "cornsilk2")
abline(lm2, lwd=2)
# So wie Gideon
lm3 <- lm(Deformation ~ RAD_Lot, data = data)
#plot(data$Deformation ~ data$RAD_Lot, xlab="True", ylab="Predicted", pch=19, cex=2, col=c("#66CC66"))
plot(data$Deformation ~ data$RAD_Lot, xlab="True", ylab="Predicted", pch=19, cex=2, col=c("orange"))
# Adjust size of axis labels and ticks
par(cex.axis = 1.2,  # Adjust size of axis ticks
    cex.lab = 1.2)   # Adjust size of axis labels
abline(lm3, lwd=4, lty="dashed")


layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page
plot(lm2)
summary(lm2)
### Calculate RMSE
predicted_values <- predict(lm2)
# Calculate model residuals
residuals <- predicted_values - data$RAD_Lot
# Calculate data residuals
residuals <- data$Deformation - data$RAD_Lot
max(residuals)
min(residuals)

which.max(residuals)
which.min(residuals)
stdev <- sd(residuals)
plot(residuals)

boxplot(residuals)
boxplot.stats(residuals)
summary(residuals)
# Calculate RMSE
rmse <- sqrt(mean(residuals^2))
rmse
rmse95 <- rmse*1.96 # 95% Confidence

# Calculate MAE
mae(data$RAD_Lot, predicted_values)
