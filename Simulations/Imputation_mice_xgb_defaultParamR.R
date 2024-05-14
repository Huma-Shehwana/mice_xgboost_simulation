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

load("Data/simdata.RData")
load("Data/missing_MAR_list.RData")


###############################    3.4 - XGBoost - default parameter  ##############################################


########################## match.type = "predicted" - default ############################

impute_MAR_xgb_default <- missing_MAR_list %>% # For each missingness
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) {  #For each dataset
      result <- system.time({
        mice_result <- mice::mice(mat, 
                                  m = m, method = "xgb",
                                  maxit = maxit,xgb.params=NULL, # will use default parameters
                                  print = FALSE)
      })
      list(mice_result = mice_result, time_taken = result) # Combining both result and time taken
    }, .options = furrr_options(seed = TRUE))
  })


xgb_ParamDefault_maxit5_P_imp_res <- map(impute_MAR_xgb_default, ~ map(., "mice_result"))  # imputation results
xgb_ParamDefault_maxit5_P_imp_time <- map(impute_MAR_xgb_default, ~ map(., "time_taken"))           # Time taken


xgb_ParamDefault_maxit5_P_eval <- xgb_ParamDefault_maxit5_P_imp_res %>% 
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
        column_to_rownames("term")
    }, .options = furrr_options(seed = TRUE)) %>% # `term` as rownames
      Reduce("+", .) / Num_ds})


xgb_ParamDefault_maxit5_P_eval_linear <- xgb_ParamDefault_maxit5_P_imp_res %>% 
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) {
      complete(mat, "all") %>% # create a list of completed data sets
        map(~.x %$% # for every completed data set....
              lm(y ~ x + z )
            ) %>% # fit linear model
        pool() %>%  # pool coefficients
        summary(conf.int = TRUE) %>% # summary of coefficients
        left_join(coef_true, by = "term") %>%
        mutate( cov = conf.low < true_vals & true_vals < conf.high, # coverage
                bias = estimate - true_vals,
                width = conf.high - conf.low) %>% # bias
        column_to_rownames("term")
    }, .options = furrr_options(seed = TRUE)) %>% # `term` as rownames
      Reduce("+", .) / Num_ds})

save(xgb_ParamDefault_maxit5_P_imp_res, file = "Imputation/mice_xgb_default/xgb_ParamDefault_maxit5_P_imp_res.RData")
save(xgb_ParamDefault_maxit5_P_imp_time, file = "Imputation/mice_xgb_default/xgb_ParamDefault_maxit5_P_imp_time.RData")
save(xgb_ParamDefault_maxit5_P_eval, file = "Imputation/mice_xgb_default/xgb_ParamDefault_maxit5_P_eval.RData")
save(xgb_ParamDefault_maxit5_P_eval_linear, file = "Imputation/mice_xgb_default/xgb_ParamDefault_maxit5_P_eval_linear.RData")

train_testplot(simdata, xgb_ParamDefault_maxit5_P_imp_res,prop_values, "Imputation/mice_xgb_default/TT_xgb_ParamDefault_maxit5_P_imp_res.pdf")

rm(xgb_ParamDefault_maxit5_P_imp_res,xgb_ParamDefault_maxit5_P_imp_time,xgb_ParamDefault_maxit5_P_eval,impute_MAR_xgb_default)

########################## match.type = "predicted.observed" - default param ############################

impute_MAR_xgb_default_PO <- missing_MAR_list %>% # For each missingness
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) {  #For each dataset
      result <- system.time({
        mice_result <- mice::mice(mat, 
                                  m = m, method = "xgb",
                                  maxit = maxit,xgb.params=NULL, 
                                  match.type = "predicted.observed", # will use default parameters
                                  print = FALSE)
      })
      list(mice_result = mice_result, time_taken = result) # Combining both result and time taken
    }, .options = furrr_options(seed = TRUE))
  })


xgb_ParamDefault_maxit5_PO_imp_res <- map(impute_MAR_xgb_default_PO, ~ map(., "mice_result"))  # imputation results
xgb_ParamDefault_maxit5_PO_imp_time <- map(impute_MAR_xgb_default_PO, ~ map(., "time_taken"))           # Time taken

xgb_ParamDefault_maxit5_PO_eval <- xgb_ParamDefault_maxit5_PO_imp_res %>% 
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
        column_to_rownames("term")
    }, .options = furrr_options(seed = TRUE)) %>% # `term` as rownames
      Reduce("+", .) / Num_ds})

xgb_ParamDefault_maxit5_PO_eval_linear <- xgb_ParamDefault_maxit5_PO_imp_res %>% 
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
        column_to_rownames("term")
    }, .options = furrr_options(seed = TRUE)) %>% # `term` as rownames
      Reduce("+", .) / Num_ds})

save(xgb_ParamDefault_maxit5_PO_imp_res, file = "Imputation/mice_xgb_default/xgb_ParamDefault_maxit5_PO_imp_res.RData")
save(xgb_ParamDefault_maxit5_PO_imp_time, file = "Imputation/mice_xgb_default/xgb_ParamDefault_maxit5_PO_imp_time.RData")
save(xgb_ParamDefault_maxit5_PO_eval, file = "Imputation/mice_xgb_default/xgb_ParamDefault_maxit5_PO_eval.RData")
save(xgb_ParamDefault_maxit5_PO_eval_linear, file = "Imputation/mice_xgb_default/xgb_ParamDefault_maxit5_PO_eval_linear.RData")

train_testplot(simdata, xgb_ParamDefault_maxit5_PO_imp_res,prop_values, "Imputation/mice_xgb_default/TT_xgb_ParamDefault_maxit5_PO_imp_res.pdf")

rm(xgb_ParamDefault_maxit5_PO_imp_res, xgb_ParamDefault_maxit5_PO_imp_time,xgb_ParamDefault_maxit5_PO_eval,impute_MAR_xgb_default_PO)

########################## match.type = "original.observed" - default param ############################

impute_MAR_xgb_default_OO <- missing_MAR_list %>% # For each missingness
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) {  #For each dataset
      result <- system.time({
        mice_result <- mice::mice(mat, 
                                  m = m, method = "xgb",
                                  maxit = maxit,xgb.params=NULL, match.type = "original.observed", # will use default parameters
                                  print = FALSE)
      })
      list(mice_result = mice_result, time_taken = result) # Combining both result and time taken
    }, .options = furrr_options(seed = TRUE))
  })


xgb_ParamDefault_maxit5_OO_imp_res <- map(impute_MAR_xgb_default_OO, ~ map(., "mice_result"))  # imputation results
xgb_ParamDefault_maxit5_OO_imp_time <- map(impute_MAR_xgb_default_OO, ~ map(., "time_taken"))           # Time taken

xgb_ParamDefault_maxit5_OO_eval <- xgb_ParamDefault_maxit5_OO_imp_res %>% 
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
        column_to_rownames("term")
    }, .options = furrr_options(seed = TRUE)) %>% # `term` as rownames
      Reduce("+", .) / Num_ds})

xgb_ParamDefault_maxit5_OO_eval_linear <- xgb_ParamDefault_maxit5_OO_imp_res %>% 
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
        column_to_rownames("term")
    }, .options = furrr_options(seed = TRUE)) %>% # `term` as rownames
      Reduce("+", .) / Num_ds})

save(xgb_ParamDefault_maxit5_OO_imp_res, file = "Imputation/mice_xgb_default/xgb_ParamDefault_maxit5_OO_imp_res.RData")
save(xgb_ParamDefault_maxit5_OO_imp_time, file = "Imputation/mice_xgb_default/xgb_ParamDefault_maxit5_OO_imp_time.RData")
save(xgb_ParamDefault_maxit5_OO_eval, file = "Imputation/mice_xgb_default/xgb_ParamDefault_maxit5_OO_eval.RData")
save(xgb_ParamDefault_maxit5_OO_eval_linear, file = "Imputation/mice_xgb_default/xgb_ParamDefault_maxit5_OO_eval_linear.RData")

train_testplot(simdata, xgb_ParamDefault_maxit5_OO_imp_res,prop_values, "Imputation/mice_xgb_default/TT_xgb_ParamDefault_maxit5_OO_imp_res.pdf")

rm(xgb_ParamDefault_maxit5_OO_imp_res, xgb_ParamDefault_maxit5_OO_imp_time,xgb_ParamDefault_maxit5_OO_eval, impute_MAR_xgb_default_OO)

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################


