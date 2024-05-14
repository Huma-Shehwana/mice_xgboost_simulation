rm(list = ls())

#source("Simulation_function.R")

library(mice)     # for imputation and amputation
library(purrr)    # for functional programming
library(furrr)    # for functional futures
library(mvtnorm)  # for multivariate normal data
library(magrittr) # for pipes
library(dplyr)    # for data manipulation
library(tibble)   # for tibbles
library(mixgb) 
library(microbenchmark)
library(ggplot2)
library(tidyr)
library(future)



source("train_testplot.R")

seed <- 123
Num_ds <- 100
prop_values <- c(20, 40, 60, 80)
m <- 5
maxit <- 5

set.seed(seed) 

available_cores <- availableCores() - 1
plan(multisession, workers = available_cores)

#################################################################################
###                       Data loading                                      ####
#################################################################################

load("Data/simdata.RData")
load("Data/missing_MAR_list.RData")


#################################################################################
####################.           3. Imputation     ###############################
#################################################################################


############################     3.1 - Default    ##################################

print("starting default imputation")

impute_MAR_default <- missing_MAR_list %>%
  map(function(mat_list) { # for each percentage
    furrr::future_map(mat_list, function(mat) {  # for each dataset
      result <- system.time({
        mice_result <- mice::mice(mat, 
                                  m = m, 
                                  maxit = maxit,
                                  print = FALSE)
      })
      list(mice_result = mice_result, time_taken = result) # Combining both result and time taken
    }, .options = furrr_options(seed = TRUE))
  })


mice_results_default <- map(impute_MAR_default, ~ map(., "mice_result")) #  extract imputation results
time_default <- map(impute_MAR_default, ~ map(., "time_taken"))   # extract time taken for imputation

eval_default <- mice_results_default %>% 
  map(function(mat_list) { # For each percentage
    furrr::future_map(mat_list, function(mat) { # for each dataset
      complete(mat, "all") %>% # create a list of completed data sets
        map(~.x %$% # for every completed data set....
              lm(y ~ x + z + x:z + I(z^2)) # fit  model
        ) %>% 
        pool() %>%  # pool coefficients
        summary(conf.int = TRUE) %>% # summary of coefficients
        left_join(coef_true, by = "term") %>%
        mutate( cov = conf.low < true_vals & true_vals < conf.high, # coverage
                bias = estimate - true_vals,
                width = conf.high - conf.low) %>% # bias
        column_to_rownames("term")}, .options = furrr_options(seed = TRUE)) %>% # `term` as rownames
      Reduce("+", .) / Num_ds })

eval_default_linear <- mice_results_default %>% 
  map(function(mat_list) { # For each percentage
    furrr::future_map(mat_list, function(mat) { # for each dataset
      complete(mat, "all") %>% # create a list of completed data sets
        map(~.x %$% # for every completed data set....
              lm(y ~ x + z ) # fit  model
        ) %>% 
        pool() %>%  # pool coefficients
        summary(conf.int = TRUE) %>% # summary of coefficients
        left_join(coef_true, by = "term") %>%
        mutate( cov = conf.low < true_vals & true_vals < conf.high, # coverage
                bias = estimate - true_vals,
                width = conf.high - conf.low) %>% # bias
        column_to_rownames("term")}, .options = furrr_options(seed = TRUE)) %>% # `term` as rownames
      Reduce("+", .) / Num_ds })

save(mice_results_default, file = "Imputation/Default_imputation_nonlinear.RData") # imputation result
save(time_default, file = "Imputation/MiceOtherMethods/Default_time_nonlinear.RData") # imputation time
save(eval_default, file = "Imputation/MiceOtherMethods/Default_evaluation_nonlinear.RData") #regression results
save(eval_default_linear, file = "Imputation/MiceOtherMethods/Default_evaluation_linear.RData") #regression results

rm(mice_results_default,time_default,eval_default,impute_MAR_default)

############################     3.2 -Imputation using  RF    ##################################

print("starting RF")


impute_MAR_RF <- missing_MAR_list %>% # for each percentage
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) { # for each dataset
      result <- system.time({
        mice_result <- mice::mice(mat, 
                                  m = m,
                                  method = "rf",
                                  maxit = maxit,
                                  print = FALSE)
      })
      list(mice_result = mice_result, time_taken = result) # Combining both imputation result and time taken
    }, .options = furrr_options(seed = TRUE))
  })


mice_results_rf <- map(impute_MAR_RF, ~ map(., "mice_result"))  #  extract imputation results   
time_RF <- map(impute_MAR_RF, ~ map(., "time_taken")) #  extract time taken for imputation results



eval_RF <- mice_results_rf %>% 
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) {
      complete(mat, "all") %>% # create a list of completed data sets
        map(~.x %$% # for every completed data set....
              lm(y ~ x + z + x:z + I(z^2))
        ) %>% # fit linear model
        pool() %>%  # pool coefficients
        summary(conf.int = TRUE) %>% # summary of coefficients
        left_join(coef_true, by = "term") %>%
        mutate( cov = conf.low < true_vals & true_vals < conf.high, # coverage
                bias = estimate - true_vals,
                width = conf.high - conf.low) %>% # bias
        column_to_rownames("term")}, .options = furrr_options(seed = TRUE)) %>% # `term` as rownames
      Reduce("+", .) / Num_ds})


eval_RF_linear <- mice_results_rf %>% 
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) {
      complete(mat, "all") %>% # create a list of completed data sets
        map(~.x %$% # for every completed data set....
              lm(y ~ x + z)
        ) %>% # fit linear model
        pool() %>%  # pool coefficients
        summary(conf.int = TRUE) %>% # summary of coefficients
        left_join(coef_true, by = "term") %>%
        mutate( cov = conf.low < true_vals & true_vals < conf.high, # coverage
                bias = estimate - true_vals,
                width = conf.high - conf.low) %>% # bias
        column_to_rownames("term")}, .options = furrr_options(seed = TRUE)) %>% # `term` as rownames
      Reduce("+", .) / Num_ds})

save(mice_results_rf, file = "Imputation/MiceOtherMethods/RF_imputation_nonlinear.RData") # save imputation results
save(time_RF, file = "Imputation/MiceOtherMethods/RF_time_nonlinear.RData") # save time taken for imputation results
save(eval_RF, file = "Imputation/MiceOtherMethods/RF_evaluation_nonlinear.RData") # save regression results fitted on imputed data
save(eval_RF_linear, file = "Imputation/MiceOtherMethods/RF_evaluation_linear.RData") # save regression results fitted on imputed data

rm(mice_results_rf,time_RF,eval_RF,impute_MAR_RF)


###################################     3.3 - CART   ##############################################

print("starting CART imputation")

impute_MAR_cart <- missing_MAR_list %>% # for each missingness
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) {  # for each dataset-
      result <- system.time({
        mice_result <- mice::mice(mat, 
                                  m = m, method = "cart",
                                  maxit = maxit,
                                  print = FALSE)
      })
      list(mice_result = mice_result, time_taken = result) # Combining both result and time taken
    }, .options = furrr_options(seed = TRUE))
  })

mice_results_cart <- map(impute_MAR_cart, ~ map(., "mice_result")) # Extracting imputation results
time_CART <- map(impute_MAR_cart, ~ map(., "time_taken")) # Time taken for imputation



eval_CART <- mice_results_cart %>% 
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) {
      complete(mat, "all") %>% # create a list of completed data sets
        map(~.x %$% # for every completed data set....
              lm(y ~ x + z + x:z + I(z^2))) %>% # fit linear model
        pool() %>%  # pool coefficients
        summary(conf.int = TRUE) %>% # summary of coefficients
        left_join(coef_true, by = "term") %>%
        mutate( cov = conf.low < true_vals & true_vals < conf.high, # coverage
                bias = estimate - true_vals,
                width = conf.high - conf.low) %>% # bias
        column_to_rownames("term")}, .options = furrr_options(seed = TRUE)) %>% # `term` as rownames
      Reduce("+", .) / Num_ds})


eval_CART_linear <- mice_results_cart %>% 
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) {
      complete(mat, "all") %>% # create a list of completed data sets
        map(~.x %$% # for every completed data set....
              lm(y ~ x + z)
            ) %>% # fit linear model
        pool() %>%  # pool coefficients
        summary(conf.int = TRUE) %>% # summary of coefficients
        left_join(coef_true, by = "term") %>%
        mutate( cov = conf.low < true_vals & true_vals < conf.high, # coverage
                bias = estimate - true_vals,
                width = conf.high - conf.low) %>% # bias
        column_to_rownames("term")}, .options = furrr_options(seed = TRUE)) %>% # `term` as rownames
      Reduce("+", .) / Num_ds})

save(mice_results_cart, file = "Imputation/MiceOtherMethods/CART_imputation_nonlinear.RData") # Save imputation results
save(time_CART, file = "Imputation/MiceOtherMethods/CART_time_nonlinear.RData")  # Save time taken for imputation
save(eval_CART, file = "Imputation/MiceOtherMethods/CART_evaluation_nonlinear.RData") # save regression results
save(eval_CART_linear, file = "Imputation/MiceOtherMethods/CART_evaluation_linear.RData") # save regression results

rm(mice_results_cart,time_CART,eval_CART,impute_MAR_cart)
