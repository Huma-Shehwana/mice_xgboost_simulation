
rm(list = ls())

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

source("train_testplot.R")

seed <- 123
Num_ds <- 100
prop_values <- c(20,40,60,80)
m <- 5
maxit <- 5

true_vals <- c(0,3,1,3,1)

set.seed(seed) 

available_cores <- availableCores() - 1
plan(multisession, workers = available_cores)


########################################################################################
##############################     load data     #######################################
########################################################################################

load("Data/simdata.RData")
load("Data/missing_MAR_list.RData")

#missing_MAR_list <- missing_MAR_list[c(1,3)]
#missing_MAR_list <- lapply(missing_MAR_list, function(sublist) {
#  sublist[1:10]
#})

coef_true <- data.frame(
  term = c("(Intercept)", "x", "z", "x:z", "I(z^2)"),
  true_vals = c(0, 3, 1, 3, 1)
)


# Parameters
# number of trees (50,500)
# eta 0.0001, 1
#generateDesign: random

All_param_set_maxit5 <- missing_MAR_list %>%
  map(function(mat_list) { 
    furrr::future_map(mat_list, function(mat) {
      result <- system.time({
        params <- mice::xgb_param_calc_g1(mat,response = "all", select_features=NULL, iter = 50)
      })
      list(params = params, time_taken = result)
    }, .options = furrr_options(seed = TRUE))
  })

hyperParameters <- map(All_param_set_maxit5, ~ map(., "params"))  # imputation results

xgb_ParamAll_maxit5_PO_imp_res_tmp <- future_map2(missing_MAR_list, hyperParameters, 
                                                  .f = function(data_inner, params_inner) {
                                                    future_map2(data_inner, params_inner, 
                                                                .f = function(data_single, params_single) {
                                                                  result <- system.time({
                                                                    mice_result <- mice(data_single, m = m, method = "xgb", 
                                                                                        maxit = maxit,xgb.params =  params_single$parameter, 
                                                                                        match.type = "predicted.observed", print = FALSE)
                                                                  })
                                                                  list(mice_result = mice_result, time_taken = result)
                                                                }, .options = furrr_options(seed = TRUE))
                                                  }, .options = furrr_options(seed = TRUE))

mice_results <- map(xgb_ParamAll_maxit5_PO_imp_res_tmp, ~ map(., "mice_result"))  # imputation results

eval <- mice_results %>% 
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


eval_linear <- mice_results %>% 
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) {
      complete(mat, "all") %>% # create a list of completed data sets
        map(~.x %$% # for every completed data set....
              lm(y ~ x + z )) %>% # fit linear model
        pool() %>%  # pool coefficients
        summary(conf.int = TRUE) %>% # summary of coefficients
        left_join(coef_true, by = "term") %>%
        mutate( cov = conf.low < true_vals & true_vals < conf.high, # coverage
                bias = estimate - true_vals,
                width = conf.high - conf.low) %>% # bias
        column_to_rownames("term")
    }, .options = furrr_options(seed = TRUE)) %>% # `term` as rownames
      Reduce("+", .) / Num_ds})

save(hyperParameters, file = "ParameterOptimization/hyperparameters_iter50g1.RData")
save(mice_results, file = "ParameterOptimization/mice_results_iter50g1.RData")
save(eval, file = "ParameterOptimization/eval_iter50g1.RData")
save(eval_linear, file = "ParameterOptimization/eval_linear_iter50g1.RData")

train_testplot(simdata, mice_results,prop_values, "ParameterOptimization/TT_xgb_ParamDefault_maxit5_PO_imp_resg1.pdf")

rm(mice_results,eval,hyperParameters, eval_linear, xgb_ParamAll_maxit5_PO_imp_res_tmp)

######################################################################################
######################################################################################


All_param_set_maxit5_100 <- missing_MAR_list %>%
  map(function(mat_list) { 
    furrr::future_map(mat_list, function(mat) {
      result <- system.time({
        params <- xgb_param_calc_g1(mat,response = "all", select_features=NULL, iter = 100)
      })
      list(params = params, time_taken = result)
    }, .options = furrr_options(seed = TRUE))
  })

hyperParameters_100 <- map(All_param_set_maxit5_100, ~ map(., "params"))  # imputation results

xgb_ParamAll_maxit5_PO_imp_res_tmp_100 <- future_map2(missing_MAR_list, hyperParameters_100, 
                                                      .f = function(data_inner, params_inner) {
                                                        future_map2(data_inner, params_inner, 
                                                                    .f = function(data_single, params_single) {
                                                                      result <- system.time({
                                                                        mice_result <- mice(data_single, m = m, method = "xgb", 
                                                                                            maxit = maxit,xgb.params =  params_single$parameter, 
                                                                                            match.type = "predicted.observed", print = FALSE)
                                                                      })
                                                                      list(mice_result = mice_result, time_taken = result)
                                                                    }, .options = furrr_options(seed = TRUE))
                                                      }, .options = furrr_options(seed = TRUE))

mice_results_100 <- map(xgb_ParamAll_maxit5_PO_imp_res_tmp_100, ~ map(., "mice_result"))  # imputation results

eval_100 <- mice_results_100 %>% 
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


eval_linear_100 <- mice_results_100 %>% 
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) {
      complete(mat, "all") %>% # create a list of completed data sets
        map(~.x %$% # for every completed data set....
              lm(y ~ x + z )) %>% # fit linear model
        pool() %>%  # pool coefficients
        summary(conf.int = TRUE) %>% # summary of coefficients
        left_join(coef_true, by = "term") %>%
        mutate( cov = conf.low < true_vals & true_vals < conf.high, # coverage
                bias = estimate - true_vals,
                width = conf.high - conf.low) %>% # bias
        column_to_rownames("term")
    }, .options = furrr_options(seed = TRUE)) %>% # `term` as rownames
      Reduce("+", .) / Num_ds})

save(hyperParameters_100, file = "ParameterOptimization/hyperparameters_iter100g1.RData")
save(mice_results_100, file = "ParameterOptimization/mice_results_iter100g1.RData")
save(eval_100, file = "ParameterOptimization/eval_iter100g1.RData")
save(eval_linear_100, file = "ParameterOptimization/eval_linear_iter100g1.RData")

train_testplot(simdata, mice_results_100,prop_values, "ParameterOptimization/TT_xgb_AllParam_maxit5_PO_imp_res_100g1.pdf")

rm(mice_results_100,eval_100,hyperParameters_100, eval_linear_100, xgb_ParamAll_maxit5_PO_imp_res_tmp_100)


########################################################################################################
########################################################################################################


All_param_set_maxit5_150 <- missing_MAR_list %>%
  map(function(mat_list) { 
    furrr::future_map(mat_list, function(mat) {
      result <- system.time({
        params <- xgb_param_calc_g1(mat,response = "all", select_features=NULL, iter = 150)
      })
      list(params = params, time_taken = result)
    }, .options = furrr_options(seed = TRUE))
  })

hyperParameters_150 <- map(All_param_set_maxit5_150, ~ map(., "params"))  # imputation results

xgb_ParamAll_maxit5_PO_imp_res_tmp_150 <- future_map2(missing_MAR_list, hyperParameters_150, 
                                                      .f = function(data_inner, params_inner) {
                                                        future_map2(data_inner, params_inner, 
                                                                    .f = function(data_single, params_single) {
                                                                      result <- system.time({
                                                                        mice_result <- mice(data_single, m = m, method = "xgb", 
                                                                                            maxit = maxit,xgb.params =  params_single$parameter, 
                                                                                            match.type = "predicted.observed", print = FALSE)
                                                                      })
                                                                      list(mice_result = mice_result, time_taken = result)
                                                                    }, .options = furrr_options(seed = TRUE))
                                                      }, .options = furrr_options(seed = TRUE))

mice_results_150 <- map(xgb_ParamAll_maxit5_PO_imp_res_tmp_150, ~ map(., "mice_result"))  # imputation results

eval_150 <- mice_results_150 %>% 
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


eval_linear_150 <- mice_results_150 %>% 
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) {
      complete(mat, "all") %>% # create a list of completed data sets
        map(~.x %$% # for every completed data set....
              lm(y ~ x + z )) %>% # fit linear model
        pool() %>%  # pool coefficients
        summary(conf.int = TRUE) %>% # summary of coefficients
        left_join(coef_true, by = "term") %>%
        mutate( cov = conf.low < true_vals & true_vals < conf.high, # coverage
                bias = estimate - true_vals,
                width = conf.high - conf.low) %>% # bias
        column_to_rownames("term")
    }, .options = furrr_options(seed = TRUE)) %>% # `term` as rownames
      Reduce("+", .) / Num_ds})

save(hyperParameters_150, file = "ParameterOptimization/hyperparameters_iter150g1.RData")
save(mice_results_150, file = "ParameterOptimization/mice_results_iter150g1.RData")
save(eval_150, file = "ParameterOptimization/eval_iter150g1.RData")
save(eval_linear_150, file = "ParameterOptimization/eval_linear_iter150g1.RData")

train_testplot(simdata, mice_results_150,prop_values, "ParameterOptimization/TT_xgb_AllParam_maxit5_PO_imp_res_150g1.pdf")

rm(mice_results_150,eval_150,hyperParameters_150, eval_linear_150, xgb_ParamAll_maxit5_PO_imp_res_tmp_150)

##############################################################################################################


All_param_set_maxit5_200 <- missing_MAR_list %>%
  map(function(mat_list) { 
    furrr::future_map(mat_list, function(mat) {
      result <- system.time({
        params <- xgb_param_calc_g1(mat,response = "all", select_features=NULL, iter = 200)
      })
      list(params = params, time_taken = result)
    }, .options = furrr_options(seed = TRUE))
  })

hyperParameters_200 <- map(All_param_set_maxit5_200, ~ map(., "params"))  # imputation results

xgb_ParamAll_maxit5_PO_imp_res_tmp_200 <- future_map2(missing_MAR_list, hyperParameters_200, 
                                                      .f = function(data_inner, params_inner) {
                                                        future_map2(data_inner, params_inner, 
                                                                    .f = function(data_single, params_single) {
                                                                      result <- system.time({
                                                                        mice_result <- mice(data_single, m = m, method = "xgb", 
                                                                                            maxit = maxit,xgb.params =  params_single$parameter, 
                                                                                            match.type = "predicted.observed", print = FALSE)
                                                                      })
                                                                      list(mice_result = mice_result, time_taken = result)
                                                                    }, .options = furrr_options(seed = TRUE))
                                                      }, .options = furrr_options(seed = TRUE))

mice_results_200 <- map(xgb_ParamAll_maxit5_PO_imp_res_tmp_200, ~ map(., "mice_result"))  # imputation results

eval_200 <- mice_results_200 %>% 
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


eval_linear_200 <- mice_results_200 %>% 
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) {
      complete(mat, "all") %>% # create a list of completed data sets
        map(~.x %$% # for every completed data set....
              lm(y ~ x + z )) %>% # fit linear model
        pool() %>%  # pool coefficients
        summary(conf.int = TRUE) %>% # summary of coefficients
        left_join(coef_true, by = "term") %>%
        mutate( cov = conf.low < true_vals & true_vals < conf.high, # coverage
                bias = estimate - true_vals,
                width = conf.high - conf.low) %>% # bias
        column_to_rownames("term")
    }, .options = furrr_options(seed = TRUE)) %>% # `term` as rownames
      Reduce("+", .) / Num_ds})

save(hyperParameters_200, file = "ParameterOptimization/hyperparameters_iter200g1.RData")
save(mice_results_200, file = "ParameterOptimization/mice_results_iter200g1.RData")
save(eval_200, file = "ParameterOptimization/eval_iter200g1.RData")
save(eval_linear_200, file = "ParameterOptimization/eval_linear_iter200g1.RData")

train_testplot(simdata, mice_results_200,prop_values, "ParameterOptimization/TT_xgb_AllParam_maxit5_PO_imp_res_200g1.pdf")


rm(mice_results_200,eval_200,hyperParameters_200, eval_linear_200, xgb_ParamAll_maxit5_PO_imp_res_tmp_200)

