
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
prop_values <- c(20,40,60, 80)
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

#simdata <- simdata[1:10]
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

All_param_set_maxit5g2 <- missing_MAR_list %>%
  map(function(mat_list) { 
    furrr::future_map(mat_list, function(mat) {
      result <- system.time({
        params <- xgb_param_calc_g2(mat,response = "all", select_features=NULL, iter = 50)
      })
      list(params = params, time_taken = result)
    }, .options = furrr_options(seed = TRUE))
  })

hyperParametersg2 <- map(All_param_set_maxit5g2, ~ map(., "params"))  # imputation results

xgb_ParamAll_maxit5_PO_imp_res_tmpg2 <- future_map2(missing_MAR_list, hyperParametersg2, 
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

mice_resultsg2 <- map(xgb_ParamAll_maxit5_PO_imp_res_tmpg2, ~ map(., "mice_result"))  # imputation results

evalg2 <- mice_resultsg2 %>% 
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


eval_linearg2 <- mice_resultsg2 %>% 
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

save(hyperParametersg2, file = "ParameterOptimization/hyperparameters_iter50g2.RData")
save(mice_resultsg2, file = "ParameterOptimization/mice_results_iter50g2.RData")
save(evalg2, file = "ParameterOptimization/eval_iter50g2.RData")
save(eval_linearg2, file = "ParameterOptimization/eval_linear_iter50g2.RData")

train_testplot(simdata, mice_resultsg2,prop_values, "ParameterOptimization/TT_xgb_ParamDefault_maxit5_PO_imp_resg2.pdf")

rm(mice_resultsg2,evalg2,hyperParametersg2, eval_linearg2, xgb_ParamAll_maxit5_PO_imp_res_tmpg2)


######################################################################################
######################################################################################


All_param_set_maxit5_100g2 <- missing_MAR_list %>%
  map(function(mat_list) { 
    furrr::future_map(mat_list, function(mat) {
      result <- system.time({
        params <- xgb_param_calc_g2(mat,response = "all", select_features=NULL, iter = 100)
      })
      list(params = params, time_taken = result)
    }, .options = furrr_options(seed = TRUE))
  })

hyperParameters_100g2 <- map(All_param_set_maxit5_100g2, ~ map(., "params"))  # imputation results

xgb_ParamAll_maxit5_PO_imp_res_tmp_100g2 <- future_map2(missing_MAR_list, hyperParameters_100g2, 
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

mice_results_100g2 <- map(xgb_ParamAll_maxit5_PO_imp_res_tmp_100g2, ~ map(., "mice_result"))  # imputation results

eval_100g2 <- mice_results_100g2 %>% 
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


eval_linear_100g2 <- mice_results_100g2 %>% 
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

save(hyperParameters_100g2, file = "ParameterOptimization/hyperparameters_iter100g2.RData")
save(mice_results_100g2, file = "ParameterOptimization/mice_results_iter100g2.RData")
save(eval_100g2, file = "ParameterOptimization/eval_iter100g2.RData")
save(eval_linear_100g2, file = "ParameterOptimization/eval_linear_iter100g2.RData")

train_testplot(simdata, mice_results_100g2,prop_values, "ParameterOptimization/TT_xgb_AllParam_maxit5_PO_imp_res_100g2.pdf")

rm(mice_results_100g2,eval_100g2,hyperParameters_100g2, eval_linear_100g2, xgb_ParamAll_maxit5_PO_imp_res_tmp_100g2)



All_param_set_maxit5_150g2 <- missing_MAR_list %>%
  map(function(mat_list) { 
    furrr::future_map(mat_list, function(mat) {
      result <- system.time({
        params <- xgb_param_calc_g2(mat,response = "all", select_features=NULL, iter = 150)
      })
      list(params = params, time_taken = result)
    }, .options = furrr_options(seed = TRUE))
  })

hyperParameters_150g2 <- map(All_param_set_maxit5_150g2, ~ map(., "params"))  # imputation results

xgb_ParamAll_maxit5_PO_imp_res_tmp_150g2 <- future_map2(missing_MAR_list, hyperParameters_150g2, 
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

mice_results_150g2 <- map(xgb_ParamAll_maxit5_PO_imp_res_tmp_150g2, ~ map(., "mice_result"))  # imputation results

eval_150g2 <- mice_results_150g2 %>% 
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


eval_linear_150g2 <- mice_results_150g2 %>% 
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

save(hyperParameters_150g2, file = "ParameterOptimization/hyperparameters_iter150g2.RData")
save(mice_results_150g2, file = "ParameterOptimization/mice_results_iter150g2.RData")
save(eval_150g2, file = "ParameterOptimization/eval_iter150g2.RData")
save(eval_linear_150g2, file = "ParameterOptimization/eval_linear_iter150g2.RData")

train_testplot(simdata, mice_results_150g2,prop_values, "ParameterOptimization/TT_xgb_AllParam_maxit5_PO_imp_res_150g2.pdf")

rm(mice_results_150g2,eval_150g2,hyperParameters_150g2, eval_linear_150g2, xgb_ParamAll_maxit5_PO_imp_res_tmp_150g2)

##############################################################################################################


All_param_set_maxit5_200g2 <- missing_MAR_list %>%
  map(function(mat_list) { 
    furrr::future_map(mat_list, function(mat) {
      result <- system.time({
        params <- xgb_param_calc_g2(mat,response = "all", select_features=NULL, iter = 200)
      })
      list(params = params, time_taken = result)
    }, .options = furrr_options(seed = TRUE))
  })

hyperParameters_200g2 <- map(All_param_set_maxit5_200g2, ~ map(., "params"))  # imputation results

xgb_ParamAll_maxit5_PO_imp_res_tmp_200g2 <- future_map2(missing_MAR_list, hyperParameters_200g2, 
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

mice_results_200g2 <- map(xgb_ParamAll_maxit5_PO_imp_res_tmp_200g2, ~ map(., "mice_result"))  # imputation results

eval_200g2 <- mice_results_200g2 %>% 
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


eval_linear_200g2 <- mice_results_200g2 %>% 
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

save(hyperParameters_200g2, file = "ParameterOptimization/hyperparameters_iter200g2.RData")
save(mice_results_200g2, file = "ParameterOptimization/mice_results_iter200g2.RData")
save(eval_200g2, file = "ParameterOptimization/eval_iter200g2.RData")
save(eval_linear_200g2, file = "ParameterOptimization/eval_linear_iter200g2.RData")

train_testplot(simdata, mice_results_200g2,prop_values, "ParameterOptimization/TT_xgb_AllParam_maxit5_PO_imp_res_200g2.pdf")

rm(mice_results_200g2,eval_200g2,hyperParameters_200g2, eval_linear_200g2, xgb_ParamAll_maxit5_PO_imp_res_tmp_200g2)


