
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
Num_ds <- 10
prop_values <- c(20, 40, 60, 80)
m <- 5
maxit <- 5

true_vals <- c(0,3,1,3,1)

set.seed(seed) 

available_cores <- availableCores() - 1
plan(multisession, workers = available_cores)


########################################################################################
##############################     load data     #######################################
########################################################################################

load("results/simdata.RData")
load("results/missing_MAR_list.RData")

simdata <- simdata[1:10]
missing_MAR_list <- lapply(missing_MAR_list, head, n = 10)



coef_true <- data.frame(
  term = c("(Intercept)", "x", "z", "x:z", "I(z^2)"),
  true_vals = c(0, 3, 1, 3, 1)
)

###############################    3.4 - XGBoost - random parameter  ##############################################

########################## match.type = "predicted.Observed" - default ############################

##########################               iter = 50                 ############################


random_param_set_maxit5 <- missing_MAR_list %>%
  map(function(mat_list) { 
    furrr::future_map(mat_list, function(mat) {
      result <- system.time({
        params <- xgb_param_calc(mat,response = NULL, select_features=NULL, iter = 50)
      })
      list(params = params, time_taken = result)
    }, .options = furrr_options(seed = TRUE))
  })

random_param_set_parameter_maxit5 <- map(random_param_set_maxit5, ~ map(., "params"))  # imputation results
random_param_set_time_maxit5 <- map(random_param_set_maxit5, ~ map(., "time_taken"))           # Time taken

save(random_param_set_time_maxit5, file = "results_iter50/xgb_Paramrandom_maxit5_P_param_time_50iter.RData")
save(random_param_set_parameter_maxit5, file = "results_iter50/xgb_Paramrandom_maxit5_P_param_50iter.RData")


xgb_Paramrandom_maxit5_PO_imp_res_tmp <- future_map2(missing_MAR_list, random_param_set_parameter_maxit5, 
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

xgb_Paramrandom_maxit5_PO_imp_res <- map(xgb_Paramrandom_maxit5_PO_imp_res_tmp, ~ map(., "mice_result"))  # imputation results
xgb_Paramrandom_maxit5_PO_imp_time <- map(xgb_Paramrandom_maxit5_PO_imp_res_tmp, ~ map(., "time_taken"))  # Time taken

xgb_Paramrandom_maxit5_PO_imp_eval <- xgb_Paramrandom_maxit5_PO_imp_res %>% 
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


xgb_Paramrandom_maxit5_PO_imp_eval_linear <- xgb_Paramrandom_maxit5_PO_imp_res %>% 
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

save(xgb_Paramrandom_maxit5_PO_imp_res, file = "results_iter50/xgb_Paramrandom_maxit5_PO_imp_res_iter50.RData")
save(xgb_Paramrandom_maxit5_PO_imp_time, file = "results_iter50/xgb_Paramrandom_maxit5_PO_imp_time_iter50.RData")
save(xgb_Paramrandom_maxit5_PO_imp_eval, file = "results_iter50/xgb_Paramrandom_maxit5_PO_imp_eval_iter50.RData")
save(xgb_Paramrandom_maxit5_PO_imp_eval_linear, file = "results_iter50/xgb_Paramrandom_maxit5_PO_imp_eval_linear_iter50.RData")

rm(xgb_Paramrandom_maxit5_PO_imp_res, xgb_Paramrandom_maxit5_PO_imp_time,xgb_Paramrandom_maxit5_PO_imp_eval, 
   xgb_Paramrandom_maxit5_PO_imp_eval_linear, xgb_Paramrandom_maxit5_PO_imp_res_tmp,random_param_set_time_maxit5, 
   random_param_set_parameter_maxit5)

########################## iter = 100 ############################


random_param_set_maxit5_iter100 <- missing_MAR_list %>%
  map(function(mat_list) { 
    furrr::future_map(mat_list, function(mat) {
      result <- system.time({
        params <- xgb_param_calc(mat,response = NULL, select_features=NULL, iter = 100)
      })
      list(params = params, time_taken = result)
    }, .options = furrr_options(seed = TRUE))
  })

random_param_set_parameter_maxit5_iter100 <- map(random_param_set_maxit5_iter100, ~ map(., "params"))  # imputation results
random_param_set_time_maxit5_iter100 <- map(random_param_set_maxit5_iter100, ~ map(., "time_taken"))           # Time taken

save(random_param_set_time_maxit5_iter100, file = "results_iter100/xgb_Paramrandom_maxit5_P_param_time_100iter.RData")
save(random_param_set_parameter_maxit5_iter100, file = "results_iter100/xgb_Paramrandom_maxit5_P_param_100iter.RData")


xgb_Paramrandom_maxit5_PO_imp_res_tmp_iter100 <- future_map2(missing_MAR_list, random_param_set_parameter_maxit5_iter100, 
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

xgb_Paramrandom_maxit5_PO_imp_res_iter100 <- map(xgb_Paramrandom_maxit5_PO_imp_res_tmp_iter100, ~ map(., "mice_result"))  # imputation results
xgb_Paramrandom_maxit5_PO_imp_time_iter100 <- map(xgb_Paramrandom_maxit5_PO_imp_res_tmp_iter100, ~ map(., "time_taken"))  # Time taken


xgb_Paramrandom_maxit5_PO_imp_eval_iter100 <- xgb_Paramrandom_maxit5_PO_imp_res_iter100 %>% 
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


xgb_Paramrandom_maxit5_PO_imp_eval_linear_iter100 <- xgb_Paramrandom_maxit5_PO_imp_res_iter100 %>% 
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) {
      complete(mat, "all") %>% # create a list of completed data sets
        map(~.x %$% # for every completed data set....
              lm(y ~ x + z)) %>% # fit linear model
        pool() %>%  # pool coefficients
        summary(conf.int = TRUE) %>% # summary of coefficients
        left_join(coef_true, by = "term") %>%
        mutate( cov = conf.low < true_vals & true_vals < conf.high, # coverage
                bias = estimate - true_vals,
                width = conf.high - conf.low) %>% # bias
        column_to_rownames("term")
    }, .options = furrr_options(seed = TRUE)) %>% # `term` as rownames
      Reduce("+", .) / Num_ds})

save(xgb_Paramrandom_maxit5_PO_imp_res_iter100, file = "results_iter100/xgb_Paramrandom_maxit5_PO_imp_res_iter100.RData")
save(xgb_Paramrandom_maxit5_PO_imp_time_iter100, file = "results_iter100/xgb_Paramrandom_maxit5_PO_imp_time_iter100.RData")
save(xgb_Paramrandom_maxit5_PO_imp_eval_iter100, file = "results_iter100/xgb_Paramrandom_maxit5_PO_imp_eval_iter100.RData")
save(xgb_Paramrandom_maxit5_PO_imp_eval_linear_iter100, file = "results_iter100/xgb_Paramrandom_maxit5_PO_imp_eval_linear_iter100.RData")

rm(xgb_Paramrandom_maxit5_PO_imp_res_iter100, xgb_Paramrandom_maxit5_PO_imp_time_iter100,xgb_Paramrandom_maxit5_PO_imp_eval_iter100, 
   xgb_Paramrandom_maxit5_PO_imp_eval_linear_iter100, xgb_Paramrandom_maxit5_PO_imp_res_tmp_iter100,random_param_set_time_maxit5_iter100, 
   random_param_set_parameter_maxit5_iter100 )

########################## iter = 150 ############################

random_param_set_maxit5_iter150 <- missing_MAR_list %>%
  map(function(mat_list) { 
    furrr::future_map(mat_list, function(mat) {
      result <- system.time({
        params <- xgb_param_calc(mat,response = NULL, select_features=NULL, iter = 150)
      })
      list(params = params, time_taken = result)
    }, .options = furrr_options(seed = TRUE))
  })

random_param_set_parameter_maxit5_iter150 <- map(random_param_set_maxit5_iter150, ~ map(., "params"))  # imputation results
random_param_set_time_maxit5_iter150 <- map(random_param_set_maxit5_iter150, ~ map(., "time_taken"))           # Time taken

save(random_param_set_time_maxit5_iter150, file = "results_iter150/xgb_Paramrandom_maxit5_P_param_time_iter150.RData")
save(random_param_set_parameter_maxit5_iter150, file = "results_iter150/xgb_Paramrandom_maxit5_P_param_iter150.RData")


xgb_Paramrandom_maxit5_PO_imp_res_tmp_iter150 <- future_map2(missing_MAR_list, random_param_set_parameter_maxit5_iter150, 
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

xgb_Paramrandom_maxit5_PO_imp_res_iter150 <- map(xgb_Paramrandom_maxit5_PO_imp_res_tmp_iter150, ~ map(., "mice_result"))  # imputation results
xgb_Paramrandom_maxit5_PO_imp_time_iter150 <- map(xgb_Paramrandom_maxit5_PO_imp_res_tmp_iter150, ~ map(., "time_taken"))  # Time taken


xgb_Paramrandom_maxit5_PO_imp_eval_iter150 <- xgb_Paramrandom_maxit5_PO_imp_res_iter150 %>% 
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


xgb_Paramrandom_maxit5_PO_imp_eval_iter150_linear <- xgb_Paramrandom_maxit5_PO_imp_res_iter150 %>% 
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) {
      complete(mat, "all") %>% # create a list of completed data sets
        map(~.x %$% # for every completed data set....
              lm(y ~ x + z)) %>% # fit linear model
        pool() %>%  # pool coefficients
        summary(conf.int = TRUE) %>% # summary of coefficients
        left_join(coef_true, by = "term") %>%
        mutate( cov = conf.low < true_vals & true_vals < conf.high, # coverage
                bias = estimate - true_vals,
                width = conf.high - conf.low) %>% # bias
        column_to_rownames("term")
    }, .options = furrr_options(seed = TRUE)) %>% # `term` as rownames
      Reduce("+", .) / Num_ds})


save(xgb_Paramrandom_maxit5_PO_imp_res_iter150, file = "results_iter150/xgb_Paramrandom_maxit5_PO_imp_res_iter150.RData")
save(xgb_Paramrandom_maxit5_PO_imp_time_iter150, file = "results_iter150/xgb_Paramrandom_maxit5_PO_imp_time_iter150.RData")
save(xgb_Paramrandom_maxit5_PO_imp_eval_iter150, file = "results_iter150/xgb_Paramrandom_maxit5_PO_imp_eval_iter150.RData")
save(xgb_Paramrandom_maxit5_PO_imp_eval_iter150_linear, file = "results_iter150/xgb_Paramrandom_maxit5_PO_imp_eval_linear_iter150.RData")

rm(xgb_Paramrandom_maxit5_PO_imp_res_iter150, xgb_Paramrandom_maxit5_PO_imp_time_iter150,xgb_Paramrandom_maxit5_PO_imp_eval_iter150, 
   xgb_Paramrandom_maxit5_PO_imp_eval_iter150_linear, xgb_Paramrandom_maxit5_PO_imp_res_tmp_iter150,random_param_set_time_maxit5_iter150, 
   random_param_set_parameter_maxit5_iter150 )

########################## iter = 200 ############################


random_param_set_maxit5_iter200 <- missing_MAR_list %>%
  map(function(mat_list) { 
    furrr::future_map(mat_list, function(mat) {
      result <- system.time({
        params <- xgb_param_calc(mat,response = NULL, select_features=NULL, iter = 200)
      })
      list(params = params, time_taken = result)
    }, .options = furrr_options(seed = TRUE))
  })

random_param_set_parameter_maxit5_iter200 <- map(random_param_set_maxit5_iter200, ~ map(., "params"))  # imputation results
random_param_set_time_maxit5_iter200 <- map(random_param_set_maxit5_iter200, ~ map(., "time_taken"))           # Time taken

save(random_param_set_time_maxit5_iter200, file = "results_iter200/xgb_Paramrandom_maxit5_P_param_time_iter200.RData")
save(random_param_set_parameter_maxit5_iter200, file = "results_iter200/xgb_Paramrandom_maxit5_P_param_iter200.RData")


xgb_Paramrandom_maxit5_PO_imp_res_tmp_iter200 <- future_map2(missing_MAR_list, random_param_set_parameter_maxit5_iter200, 
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

xgb_Paramrandom_maxit5_PO_imp_res_iter200 <- map(xgb_Paramrandom_maxit5_PO_imp_res_tmp_iter200, ~ map(., "mice_result"))  # imputation results
xgb_Paramrandom_maxit5_PO_imp_time_iter200 <- map(xgb_Paramrandom_maxit5_PO_imp_res_tmp_iter200, ~ map(., "time_taken"))  # Time taken


xgb_Paramrandom_maxit5_PO_imp_eval_iter200 <- xgb_Paramrandom_maxit5_PO_imp_res_iter200 %>% 
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


xgb_Paramrandom_maxit5_PO_imp_eval_iter200_linear <- xgb_Paramrandom_maxit5_PO_imp_res_iter200 %>% 
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

save(xgb_Paramrandom_maxit5_PO_imp_res_iter200, file = "results_iter200/xgb_Paramrandom_maxit5_PO_imp_res_iter200.RData")
save(xgb_Paramrandom_maxit5_PO_imp_time_iter200, file = "results_iter200/xgb_Paramrandom_maxit5_PO_imp_time_iter200.RData")
save(xgb_Paramrandom_maxit5_PO_imp_eval_iter200, file = "results_iter200/xgb_Paramrandom_maxit5_PO_imp_eval_iter200.RData")
save(xgb_Paramrandom_maxit5_PO_imp_eval_iter200_linear, file = "results_iter200/xgb_Paramrandom_maxit5_PO_imp_eval_linear_iter200.RData")

rm(xgb_Paramrandom_maxit5_PO_imp_res_iter200, xgb_Paramrandom_maxit5_PO_imp_time_iter200,xgb_Paramrandom_maxit5_PO_imp_eval_iter200, 
   xgb_Paramrandom_maxit5_PO_imp_eval_iter200_linear, xgb_Paramrandom_maxit5_PO_imp_res_tmp_iter200,random_param_set_time_maxit5_iter200, 
   random_param_set_parameter_maxit5_iter200 )

########################## iter = 250 ############################


random_param_set_maxit5_iter250 <- missing_MAR_list %>%
  map(function(mat_list) { 
    furrr::future_map(mat_list, function(mat) {
      result <- system.time({
        params <- xgb_param_calc(mat,response = NULL, select_features=NULL, iter = 250)
      })
      list(params = params, time_taken = result)
    }, .options = furrr_options(seed = TRUE))
  })

random_param_set_parameter_maxit5_iter250 <- map(random_param_set_maxit5_iter250, ~ map(., "params"))  # imputation results
random_param_set_time_maxit5_iter250 <- map(random_param_set_maxit5_iter250, ~ map(., "time_taken"))           # Time taken

save(random_param_set_time_maxit5_iter250, file = "results_iter250/xgb_Paramrandom_maxit5_P_param_time_iter250.RData")
save(random_param_set_parameter_maxit5_iter250, file = "results_iter250/xgb_Paramrandom_maxit5_P_param_iter250.RData")


xgb_Paramrandom_maxit5_PO_imp_res_tmp_iter250 <- future_map2(missing_MAR_list, random_param_set_parameter_maxit5_iter250, 
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

xgb_Paramrandom_maxit5_PO_imp_res_iter250 <- map(xgb_Paramrandom_maxit5_PO_imp_res_tmp_iter250, ~ map(., "mice_result"))  # imputation results
xgb_Paramrandom_maxit5_PO_imp_time_iter250 <- map(xgb_Paramrandom_maxit5_PO_imp_res_tmp_iter250, ~ map(., "time_taken"))  # Time taken


xgb_Paramrandom_maxit5_PO_imp_eval_iter250 <- xgb_Paramrandom_maxit5_PO_imp_res_iter250 %>% 
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

xgb_Paramrandom_maxit5_PO_imp_eval_iter250_linear <- xgb_Paramrandom_maxit5_PO_imp_res_iter250 %>% 
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) {
      complete(mat, "all") %>% # create a list of completed data sets
        map(~.x %$% # for every completed data set....
              lm(y ~ x + z)) %>% # fit linear model
        pool() %>%  # pool coefficients
        summary(conf.int = TRUE) %>% # summary of coefficients
        left_join(coef_true, by = "term") %>%
        mutate( cov = conf.low < true_vals & true_vals < conf.high, # coverage
                bias = estimate - true_vals,
                width = conf.high - conf.low) %>% # bias
        column_to_rownames("term")
    }, .options = furrr_options(seed = TRUE)) %>% # `term` as rownames
      Reduce("+", .) / Num_ds})
save(xgb_Paramrandom_maxit5_PO_imp_res_iter250, file = "results_iter250/xgb_Paramrandom_maxit5_PO_imp_res_iter250.RData")
save(xgb_Paramrandom_maxit5_PO_imp_time_iter250, file = "results_iter250/xgb_Paramrandom_maxit5_PO_imp_time_iter250.RData")
save(xgb_Paramrandom_maxit5_PO_imp_eval_iter250, file = "results_iter250/xgb_Paramrandom_maxit5_PO_imp_eval_iter250.RData")
save(xgb_Paramrandom_maxit5_PO_imp_eval_iter250_linear, file = "results_iter250/xgb_Paramrandom_maxit5_PO_imp_eval_linear_iter250.RData")

rm(xgb_Paramrandom_maxit5_PO_imp_res_iter250, xgb_Paramrandom_maxit5_PO_imp_time_iter250,xgb_Paramrandom_maxit5_PO_imp_eval_iter250, 
   xgb_Paramrandom_maxit5_PO_imp_eval_iter250_linear, xgb_Paramrandom_maxit5_PO_imp_res_tmp_iter250,random_param_set_time_maxit5_iter250, 
   random_param_set_parameter_maxit5_iter250 )

########################## iter = 300 ############################

random_param_set_maxit5_iter300 <- missing_MAR_list %>%
  map(function(mat_list) { 
    furrr::future_map(mat_list, function(mat) {
      result <- system.time({
        params <- xgb_param_calc(mat,response = NULL, select_features=NULL, iter = 300)
      })
      list(params = params, time_taken = result)
    }, .options = furrr_options(seed = TRUE))
  })

random_param_set_parameter_maxit5_iter300 <- map(random_param_set_maxit5_iter300, ~ map(., "params"))  # imputation results
random_param_set_time_maxit5_iter300 <- map(random_param_set_maxit5_iter300, ~ map(., "time_taken"))           # Time taken

save(random_param_set_time_maxit5_iter300, file = "results_iter300/xgb_Paramrandom_maxit5_P_param_time_iter300.RData")
save(random_param_set_parameter_maxit5_iter300, file = "results_iter300/xgb_Paramrandom_maxit5_P_param_iter300.RData")


xgb_Paramrandom_maxit5_PO_imp_res_tmp_iter300 <- future_map2(missing_MAR_list, random_param_set_parameter_maxit5_iter300, 
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

xgb_Paramrandom_maxit5_PO_imp_res_iter300 <- map(xgb_Paramrandom_maxit5_PO_imp_res_tmp_iter300, ~ map(., "mice_result"))  # imputation results
xgb_Paramrandom_maxit5_PO_imp_time_iter300 <- map(xgb_Paramrandom_maxit5_PO_imp_res_tmp_iter300, ~ map(., "time_taken"))  # Time taken


xgb_Paramrandom_maxit5_PO_imp_eval_iter300 <- xgb_Paramrandom_maxit5_PO_imp_res_iter300 %>% 
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


xgb_Paramrandom_maxit5_PO_imp_eval_iter300_linear <- xgb_Paramrandom_maxit5_PO_imp_res_iter300 %>% 
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) {
      complete(mat, "all") %>% # create a list of completed data sets
        map(~.x %$% # for every completed data set....
              lm(y ~ x + z)) %>% # fit linear model
        pool() %>%  # pool coefficients
        summary(conf.int = TRUE) %>% # summary of coefficients
        left_join(coef_true, by = "term") %>%
        mutate( cov = conf.low < true_vals & true_vals < conf.high, # coverage
                bias = estimate - true_vals,
                width = conf.high - conf.low) %>% # bias
        column_to_rownames("term")
    }, .options = furrr_options(seed = TRUE)) %>% # `term` as rownames
      Reduce("+", .) / Num_ds})

save(xgb_Paramrandom_maxit5_PO_imp_res_iter300, file = "results_iter300/xgb_Paramrandom_maxit5_PO_imp_res_iter300.RData")
save(xgb_Paramrandom_maxit5_PO_imp_time_iter300, file = "results_iter300/xgb_Paramrandom_maxit5_PO_imp_time_iter300.RData")
save(xgb_Paramrandom_maxit5_PO_imp_eval_iter300, file = "results_iter300/xgb_Paramrandom_maxit5_PO_imp_eval_iter300.RData")
save(xgb_Paramrandom_maxit5_PO_imp_eval_iter300_linear, file = "results_iter300/xgb_Paramrandom_maxit5_PO_imp_eval_linear_iter300.RData")

rm(xgb_Paramrandom_maxit5_PO_imp_res_iter300, xgb_Paramrandom_maxit5_PO_imp_time_iter300,xgb_Paramrandom_maxit5_PO_imp_eval_iter300, 
   xgb_Paramrandom_maxit5_PO_imp_eval_iter300_linear, xgb_Paramrandom_maxit5_PO_imp_res_tmp_iter300,random_param_set_time_maxit5_iter300, 
   random_param_set_parameter_maxit5_iter300 )

