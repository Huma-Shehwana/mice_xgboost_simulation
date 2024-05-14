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
########################     Step 1: Data generation    #########################
#################################################################################

sigma <- matrix(data = c(1, 0.7, 0.7, 1),     # covariance matrix
                ncol = 2)

#Let's generate 1000 data sets with 1000 entries each. Each data set is saved as a separate element in the list.

simdata <- replicate(n = Num_ds, 
                     expr = mvtnorm::rmvnorm(n = 1000, 
                                             mean = c(4, 1), 
                                             sigma = sigma) %>% 
                       as_tibble() %>% # make into a tibble
                       rename(x = V1, z = V2) %>% # rename columns
                       mutate(y = as.numeric(3 * x +  z + 3*I(x*z) + z^2 + rnorm(1000, sd = 20))), # add y # mean 0 std of 5 or 10
                     simplify = FALSE) # keep as list of generated sets


coef_true <- data.frame(
  term = c("(Intercept)", "x", "z", "x:z", "I(z^2)"),
  true_vals = c(0, 3, 1, 3, 1)
)

#Regression

desired_order <- c("(Intercept)", "x", "z", "x:z", "I(z^2)")

simdata %>% 
  map_dfr(~.x %$% # for every simulated set in simdata....
            lm(y ~ x + z + x:z + I(z^2)) %>% # fit model
            coefficients) %>% # extract coefficients) %>%
colMeans() # add all and divide by length (= average)


# mean R2
simdata %>% 
  map(~.x %$% # for every simulated set in simdata....
        summary(lm(y ~ x + z + x:z + I(z^2))) %$% # fit model
        r.squared) %>% # extract coefficients
  unlist() %>%
  mean() # add all and divide by length (= average)



#################################################################################
#############          2. Missing data 
#################################################################################

# make data missing. For each missingness percentage, generate 1000 data sets with given percentage of missing data. 
plan(sequential)
set.seed(123)
plan(multisession, workers = available_cores)


apply_ampute <- function(simdata, prop_value) {
  simdata %>%
    furrr::future_map(function(x) {
      x %>%
        ampute(prop = prop_value / 100, mech = "MAR", type = "RIGHT") %>%
        .$amp 
    }, .options = furrr_options(seed = TRUE))
}

missing_MAR_list <- map(prop_values, ~ apply_ampute(simdata, .x))
names(missing_MAR_list) <- prop_values

# missing_MAR_list is a list of missingness percentage with a sub-list of 1000 datasets

NAs_in_data <- map(missing_MAR_list, ~ map_dbl(.x, ~ sum(is.na(.x))))

save(simdata, file = "results/simdata.RData") # imputation result
save(missing_MAR_list, file = "results/missing_MAR_list.RData") # imputation time


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


save(mice_results_default, file = "results/Default_imputation_nonlinear.RData") # imputation result
save(time_default, file = "results/Default_time_nonlinear.RData") # imputation time
save(eval_default, file = "results/Default_evaluation_nonlinear.RData") #regression results

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

save(mice_results_rf, file = "results/RF_imputation_nonlinear.RData") # save imputation results
save(time_RF, file = "results/RF_time_nonlinear.RData") # save time taken for imputation results
save(eval_RF, file = "results/RF_evaluation_nonlinear.RData") # save regression results fitted on imputed data

rm(mice_results_rf,time_RF,eval_RF,impute_MAR_RF)


###################################     3.3 - CART   ##############################################

print("starting CART")


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


save(mice_results_cart, file = "results/CART_imputation_nonlinear.RData") # Save imputation results
save(time_CART, file = "results/CART_time_nonlinear.RData")  # Save time taken for imputation
save(eval_CART, file = "results/CART_evaluation_nonlinear.RData") # save regression results

rm(mice_results_cart,time_CART,eval_CART,impute_MAR_cart)

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

save(xgb_ParamDefault_maxit5_P_imp_res, file = "results/xgb_ParamDefault_maxit5_P_imp_res.RData")
save(xgb_ParamDefault_maxit5_P_imp_time, file = "results/xgb_ParamDefault_maxit5_P_imp_time.RData")
save(xgb_ParamDefault_maxit5_P_eval, file = "results/xgb_ParamDefault_maxit5_P_eval.RData")


train_testplot(simdata, xgb_ParamDefault_maxit5_P_imp_res,prop_values, "results/TT_xgb_ParamDefault_maxit5_P_imp_res.pdf")

rm(xgb_ParamDefault_maxit5_P_imp_res,xgb_ParamDefault_maxit5_P_imp_time,xgb_ParamDefault_maxit5_P_eval,impute_MAR_xgb_default)


########################## match.type = "predicted.observed" - default param ############################

impute_MAR_xgb_default_PO <- missing_MAR_list %>% # For each missingness
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) {  #For each dataset
      result <- system.time({
        mice_result <- mice::mice(mat, 
                                  m = m, method = "xgb",
                                  maxit = maxit,xgb.params=NULL, match.type = "predicted.observed", # will use default parameters
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

save(xgb_ParamDefault_maxit5_PO_imp_res, file = "results/xgb_ParamDefault_maxit5_PO_imp_res.RData")
save(xgb_ParamDefault_maxit5_PO_imp_time, file = "results/xgb_ParamDefault_maxit5_PO_imp_time.RData")
save(xgb_ParamDefault_maxit5_PO_eval, file = "results/xgb_ParamDefault_maxit5_PO_eval.RData")

train_testplot(simdata, xgb_ParamDefault_maxit5_PO_imp_res,prop_values, "results/TT_xgb_ParamDefault_maxit5_PO_imp_res.pdf")

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

save(xgb_ParamDefault_maxit5_OO_imp_res, file = "results/xgb_ParamDefault_maxit5_OO_imp_res.RData")
save(xgb_ParamDefault_maxit5_OO_imp_time, file = "results/xgb_ParamDefault_maxit5_OO_imp_time.RData")
save(xgb_ParamDefault_maxit5_OO_eval, file = "results/xgb_ParamDefault_maxit5_OO_eval.RData")


train_testplot(simdata, xgb_ParamDefault_maxit5_OO_imp_res,prop_values, "results/TT_xgb_ParamDefault_maxit5_OO_imp_res.pdf")

rm(xgb_ParamDefault_maxit5_OO_imp_res, xgb_ParamDefault_maxit5_OO_imp_time,xgb_ParamDefault_maxit5_OO_eval, impute_MAR_xgb_default_OO)


########################    3.5 - XGBoost - Random parameter ####################################

################################################################
########  Parameter estimation
################################################################

random_param_set_maxit5 <- missing_MAR_list %>%
  map(function(mat_list) { 
    furrr::future_map(mat_list, function(mat) {
      result <- system.time({
        params <- xgb_param_calc(mat,response = NULL, select_features=NULL)
      })
      list(params = params, time_taken = result)
    }, .options = furrr_options(seed = TRUE))
  })

random_param_set_parameter_maxit5 <- map(random_param_set_maxit5, ~ map(., "params"))  # imputation results
random_param_set_time_maxit5 <- map(random_param_set_maxit5, ~ map(., "time_taken"))           # Time taken

save(random_param_set_time_maxit5, file = "results/xgb_ParamRandom_maxit5_P_param_time.RData")
save(random_param_set_parameter_maxit5, file = "results/xgb_ParamRandom_maxit5_P_param.RData")

########################## match.type = "predicted" - Random ############################

xgb_ParamRandom_maxit5_P_imp_res_tmp <- future_map2(missing_MAR_list, random_param_set_parameter_maxit5, 
                                                    .f = function(data_inner, params_inner) {
                                                      future_map2(data_inner, params_inner, 
                                                                  .f = function(data_single, params_single) {
                                                                    result <- system.time({
                                                                      mice_result <- mice(data_single, m = m, method = "xgb", maxit = maxit,
                                                                                          xgb.params =  params_single$parameter, print = FALSE)
                                                                    })
                                                                    list(mice_result = mice_result, time_taken = result)
                                                                  }, .options = furrr_options(seed = TRUE))
                                                    }, .options = furrr_options(seed = TRUE))

xgb_ParamRandom_maxit5_P_imp_res <- map(xgb_ParamRandom_maxit5_P_imp_res_tmp, ~ map(., "mice_result"))  # imputation results
xgb_ParamRandom_maxit5_P_imp_time <- map(xgb_ParamRandom_maxit5_P_imp_res_tmp, ~ map(., "time_taken"))           # Time taken


xgb_ParamRandom_maxit5_P_imp_eval <- xgb_ParamRandom_maxit5_P_imp_res %>% 
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

save(xgb_ParamRandom_maxit5_P_imp_res, file = "results/xgb_ParamRandom_maxit5_P_imp_res.RData")
save(xgb_ParamRandom_maxit5_P_imp_time, file = "results/xgb_ParamRandom_maxit5_P_imp_time.RData")
save(xgb_ParamRandom_maxit5_P_imp_eval, file = "results/xgb_ParamRandom_maxit5_P_imp_eval.RData")

train_testplot(simdata,xgb_ParamRandom_maxit5_P_imp_res, prop_values, "results/TT_xgb_ParamRandom_maxit5_P_imp_res.pdf")

rm(xgb_ParamRandom_maxit5_P_imp_res, xgb_ParamRandom_maxit5_P_imp_time,xgb_ParamRandom_maxit5_P_imp_eval, xgb_ParamRandom_maxit5_P_imp_res_tmp)


########################## match.type = "predicted.observed" - Random ############################

xgb_ParamRandom_maxit5_PO_imp_res_tmp <- future_map2(missing_MAR_list, random_param_set_parameter_maxit5, 
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

xgb_ParamRandom_maxit5_PO_imp_res <- map(xgb_ParamRandom_maxit5_PO_imp_res_tmp, ~ map(., "mice_result"))  # imputation results
xgb_ParamRandom_maxit5_PO_imp_time <- map(xgb_ParamRandom_maxit5_PO_imp_res_tmp, ~ map(., "time_taken"))  # Time taken


xgb_ParamRandom_maxit5_PO_imp_eval <- xgb_ParamRandom_maxit5_PO_imp_res %>% 
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

save(xgb_ParamRandom_maxit5_PO_imp_res, file = "results/xgb_ParamRandom_maxit5_PO_imp_res.RData")
save(xgb_ParamRandom_maxit5_PO_imp_time, file = "results/xgb_ParamRandom_maxit5_PO_imp_time.RData")
save(xgb_ParamRandom_maxit5_PO_imp_eval, file = "results/xgb_ParamRandom_maxit5_PO_imp_eval.RData")

train_testplot(simdata, xgb_ParamRandom_maxit5_PO_imp_res, prop_values, "results/TT_xgb_ParamRandom_maxit5_PO_imp_res.pdf")

rm(xgb_ParamRandom_maxit5_PO_imp_res, xgb_ParamRandom_maxit5_PO_imp_time,xgb_ParamRandom_maxit5_PO_imp_eval, xgb_ParamRandom_maxit5_PO_imp_res_tmp)

########################## match.type = "original.observed" - Random ############################

xgb_ParamRandom_maxit5_OO_imp_res_tmp <- future_map2(missing_MAR_list, random_param_set_parameter_maxit5, 
                                                     .f = function(data_inner, params_inner) {
                                                       future_map2(data_inner, params_inner, 
                                                                   .f = function(data_single, params_single) {
                                                                     result <- system.time({
                                                                       mice_result <- mice(data_single, m = m, method = "xgb",
                                                                                           maxit = maxit,xgb.params =  params_single$parameter, 
                                                                                           match.type = "original.observed", print = FALSE)
                                                                     })
                                                                     list(mice_result = mice_result, time_taken = result)
                                                                   }, .options = furrr_options(seed = TRUE))
                                                     }, .options = furrr_options(seed = TRUE))

xgb_ParamRandom_maxit5_OO_imp_res <- map(xgb_ParamRandom_maxit5_OO_imp_res_tmp, ~ map(., "mice_result"))  # imputation results
xgb_ParamRandom_maxit5_OO_imp_time <- map(xgb_ParamRandom_maxit5_OO_imp_res_tmp, ~ map(., "time_taken"))           # Time taken

xgb_ParamRandom_maxit5_OO_imp_eval <- xgb_ParamRandom_maxit5_OO_imp_res %>% 
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

save(xgb_ParamRandom_maxit5_OO_imp_res, file = "results/xgb_ParamRandom_maxit5_OO_imp_res.RData")
save(xgb_ParamRandom_maxit5_OO_imp_time, file = "results/xgb_ParamRandom_maxit5_OO_imp_time.RData")
save(xgb_ParamRandom_maxit5_OO_imp_eval, file = "results/xgb_ParamRandom_maxit5_OO_imp_eval.RData")

train_testplot(simdata, xgb_ParamRandom_maxit5_OO_imp_res,prop_values, "results/TT_xgb_ParamRandom_maxit5_OO_imp_res.pdf")


########################    3.6 - XGBoost - All parameter ####################################

################################################################
########  Parameter estimation
################################################################

all_param_set_maxit5 <- missing_MAR_list %>%
  map(function(mat_list) { 
    furrr::future_map(mat_list, function(mat) {
      result <- system.time({
        params <- xgb_param_calc(mat,response = "all", select_features=NULL)
      })
      list(params = params, time_taken = result)
    }, .options = furrr_options(seed = TRUE))
  })

all_param_set_parameter_maxit5 <- map(all_param_set_maxit5, ~ map(., "params"))  # imputation results
all_param_set_time_maxit5 <- map(all_param_set_maxit5, ~ map(., "time_taken"))           # Time taken

save(all_param_set_time_maxit5, file = "results/xgb_ParamAll_maxit5_P_param_time.RData")
save(all_param_set_parameter_maxit5, file = "results/xgb_ParamAll_maxit5_P_param.RData")

##################################################################################################################
##################################################################################################################

########################## match.type = "predicted" - All ############################

xgb_ParamAll_maxit5_P_imp_res_tmp <- future_map2(missing_MAR_list, all_param_set_parameter_maxit5, 
                                                 .f = function(data_inner, params_inner) {
                                                   future_map2(data_inner, params_inner, 
                                                               .f = function(data_single, params_single) {
                                                                 result <- system.time({
                                                                   mice_result <- mice(data_single, m = m, method = "xgb", 
                                                                                       maxit = maxit,xgb.params =  params_single$parameter, 
                                                                                       print = FALSE)
                                                                 })
                                                                 list(mice_result = mice_result, time_taken = result)
                                                               }, .options = furrr_options(seed = TRUE))
                                                 }, .options = furrr_options(seed = TRUE))

xgb_ParamAll_maxit5_P_imp_res <- map(xgb_ParamAll_maxit5_P_imp_res_tmp, ~ map(., "mice_result"))  # imputation results
xgb_ParamAll_maxit5_P_imp_time <- map(xgb_ParamAll_maxit5_P_imp_res_tmp, ~ map(., "time_taken"))           # Time taken


xgb_ParamAll_maxit5_P_imp_eval <- xgb_ParamAll_maxit5_P_imp_res %>% 
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

save(xgb_ParamAll_maxit5_P_imp_res, file = "results/xgb_ParamAll_maxit5_P_imp_res.RData")
save(xgb_ParamAll_maxit5_P_imp_time, file = "results/xgb_ParamAll_maxit5_P_imp_time.RData")
save(xgb_ParamAll_maxit5_P_imp_eval, file = "results/xgb_ParamAll_maxit5_P_imp_eval.RData")

train_testplot(simdata,xgb_ParamAll_maxit5_P_imp_res, prop_values, "results/TT_xgb_ParamAll_maxit5_P_imp_res.pdf")

rm(xgb_ParamAll_maxit5_P_imp_res, xgb_ParamAll_maxit5_P_imp_time, xgb_ParamAll_maxit5_P_imp_eval, xgb_ParamAll_maxit5_P_imp_res_tmp)

########################## match.type = "predicted.observed" - All ############################

xgb_ParamAll_maxit5_PO_imp_res_tmp <- future_map2(missing_MAR_list, all_param_set_parameter_maxit5, 
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

xgb_ParamAll_maxit5_PO_imp_res <- map(xgb_ParamAll_maxit5_PO_imp_res_tmp, ~ map(., "mice_result"))  # imputation results
xgb_ParamAll_maxit5_PO_imp_time <- map(xgb_ParamAll_maxit5_PO_imp_res_tmp, ~ map(., "time_taken"))  # Time taken


xgb_ParamAll_maxit5_PO_imp_eval <- xgb_ParamAll_maxit5_PO_imp_res %>% 
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

save(xgb_ParamAll_maxit5_PO_imp_res, file = "results/xgb_ParamAll_maxit5_PO_imp_res.RData")
save(xgb_ParamAll_maxit5_PO_imp_time, file = "results/xgb_ParamAll_maxit5_PO_imp_time.RData")
save(xgb_ParamAll_maxit5_PO_imp_eval, file = "results/xgb_ParamAll_maxit5_PO_imp_eval.RData")

train_testplot(simdata, xgb_ParamAll_maxit5_PO_imp_res, prop_values, "results/TT_xgb_ParamAll_maxit5_PO_imp_res.pdf")

rm(xgb_ParamAll_maxit5_PO_imp_res, xgb_ParamAll_maxit5_PO_imp_time, xgb_ParamAll_maxit5_PO_imp_eval, xgb_ParamAll_maxit5_PO_imp_res_tmp)


########################## match.type = "original.observed" - All ############################


xgb_ParamAll_maxit5_OO_imp_res_tmp <- future_map2(missing_MAR_list, all_param_set_parameter_maxit5, 
                                                  .f = function(data_inner, params_inner) {
                                                    future_map2(data_inner, params_inner, 
                                                                .f = function(data_single, params_single) {
                                                                  result <- system.time({
                                                                    mice_result <- mice(data_single, m = m, method = "xgb",
                                                                                        maxit = maxit,xgb.params =  params_single$parameter, 
                                                                                        match.type = "original.observed", print = FALSE)
                                                                  })
                                                                  list(mice_result = mice_result, time_taken = result)
                                                                }, .options = furrr_options(seed = TRUE))
                                                  }, .options = furrr_options(seed = TRUE))

xgb_ParamAll_maxit5_OO_imp_res <- map(xgb_ParamAll_maxit5_OO_imp_res_tmp, ~ map(., "mice_result"))  # imputation results
xgb_ParamAll_maxit5_OO_imp_time <- map(xgb_ParamAll_maxit5_OO_imp_res_tmp, ~ map(., "time_taken"))           # Time taken


xgb_ParamAll_maxit5_OO_imp_eval <- xgb_ParamAll_maxit5_OO_imp_res %>% 
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

save(xgb_ParamAll_maxit5_OO_imp_res, file = "results/xgb_ParamAll_maxit5_OO_imp_res.RData")
save(xgb_ParamAll_maxit5_OO_imp_time, file = "results/xgb_ParamAll_maxit5_OO_imp_time.RData")
save(xgb_ParamAll_maxit5_OO_imp_eval, file = "results/xgb_ParamAll_maxit5_OO_imp_eval.RData")

train_testplot(simdata, xgb_ParamAll_maxit5_OO_imp_res,prop_values, "results/TT_xgb_ParamAll_maxit5_OO_imp_res.pdf")


rm(xgb_ParamAll_maxit5_OO_imp_res_tmp, xgb_ParamAll_maxit5_OO_imp_res, xgb_ParamAll_maxit5_OO_imp_time, xgb_ParamAll_maxit5_OO_imp_eval)

###########################################################################################################
###########################################################################################################
###########################################################################################################

############################     3.7 -Imputation using  mixgb    ##################################

##################        PMM = NULL      #############################

print("starting Mixgb")

mixgb_maxit5_pmmNULL_tmp <- missing_MAR_list %>% # for each percentage
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) { # for each dataset
      result <- system.time({
        mice_result <- mixgb(mat,pmm.type = NULL, m = m, maxit = maxit,
                             print = FALSE)
      })
      list(mice_result = mice_result, time_taken = result) # Combining both imputation result and time taken
    }, .options = furrr_options(seed = TRUE))
  })


mixgb_maxit5_pmmNULL_imp <- map(mixgb_maxit5_pmmNULL_tmp, ~ map(., "mice_result"))  #  extract imputation results   
mixgb_maxit5_pmmNULL_time <- map(mixgb_maxit5_pmmNULL_tmp, ~ map(., "time_taken")) #  extract time taken for imputation results


mixgb_maxit5_pmmNULL_eval <- mixgb_maxit5_pmmNULL_imp %>% 
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) { mat %>%
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

save(mixgb_maxit5_pmmNULL_imp, file = "results/mixgb_maxit5_pmmNULL_imp.RData") # save imputation results
save(mixgb_maxit5_pmmNULL_time, file = "results/mixgb_maxit5_pmmNULL_time.RData") # save time taken for imputation results
save(mixgb_maxit5_pmmNULL_eval, file = "results/mixgb_maxit5_pmmNULL_eval.RData") # save regression results fitted on imputed data


rm(mixgb_maxit5_pmmNUL_tmp, mixgb_maxit5_pmmNUL_imp, mixgb_maxit5_pmmNUL_time, mixgb_maxit5_pmmNUL_eval)


###################### pmm = 1 ########################

mixgb_maxit5_pmm1_tmp <- missing_MAR_list %>% # for each percentage
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) { # for each dataset
      result <- system.time({
        mice_result <- mixgb(mat,pmm.type = 1, m = m, maxit = maxit,
                             print = FALSE)
      })
      list(mice_result = mice_result, time_taken = result) # Combining both imputation result and time taken
    }, .options = furrr_options(seed = TRUE))
  })


mixgb_maxit5_pmm1_imp <- map(mixgb_maxit5_pmm1_tmp, ~ map(., "mice_result"))  #  extract imputation results   
mixgb_maxit5_pmm1_time <- map(mixgb_maxit5_pmm1_tmp, ~ map(., "time_taken")) #  extract time taken for imputation results


mixgb_maxit5_pmm1_eval <- mixgb_maxit5_pmm1_imp %>% 
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) { mat %>%
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

save(mixgb_maxit5_pmm1_imp, file = "results/mixgb_maxit5_pmm1_imp.RData") # save imputation results
save(mixgb_maxit5_pmm1_time, file = "results/mixgb_maxit5_pmm1_time.RData") # save time taken for imputation results
save(mixgb_maxit5_pmm1_eval, file = "results/mixgb_maxit5_pmm1_eval.RData") # save regression results fitted on imputed data


rm(mixgb_maxit5_pmm1_tmp, mixgb_maxit5_pmm1_imp, mixgb_maxit5_pmm1_time, mixgb_maxit5_pmm1_eval)


##################### pmm = 2 ##############################

mixgb_maxit5_pmm2_tmp <- missing_MAR_list %>% # for each percentage
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) { # for each dataset
      result <- system.time({
        mice_result <- mixgb(mat,pmm.type = 2, m = m, maxit = maxit,
                             print = FALSE)
      })
      list(mice_result = mice_result, time_taken = result) # Combining both imputation result and time taken
    }, .options = furrr_options(seed = TRUE))
  })


mixgb_maxit5_pmm2_imp <- map(mixgb_maxit5_pmm2_tmp, ~ map(., "mice_result"))  #  extract imputation results   
mixgb_maxit5_pmm2_time <- map(mixgb_maxit5_pmm2_tmp, ~ map(., "time_taken")) #  extract time taken for imputation results


mixgb_maxit5_pmm2_eval <- mixgb_maxit5_pmm2_imp %>% 
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) { mat %>%
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

save(mixgb_maxit5_pmm2_imp, file = "results/mixgb_maxit5_pmm2_imp.RData") # save imputation results
save(mixgb_maxit5_pmm2_time, file = "results/mixgb_maxit5_pmm2_time.RData") # save time taken for imputation results
save(mixgb_maxit5_pmm2_eval, file = "results/mixgb_maxit5_pmm2_eval.RData") # save regression results fitted on imputed data


rm(mixgb_maxit5_pmm2_tmp, mixgb_maxit5_pmm2_imp, mixgb_maxit5_pmm2_time, mixgb_maxit5_pmm2_eval)

###########################################################################################################
###########################################################################################################
###########################################################################################################


cor_list <- map(simdata, ~ cor(.x)) %>% Reduce("+", .) / length(simdata)

cors_all <- cbind( 
  cor_withY = cor_list[c("y","x","z"),"y"],
  cor_withX =cor_list[c("y","x","z"),"x"],
  cor_withZ = cor_list[c("y","x","z"),"z"]
)


saveRDS(cors_all, "Correlations_XYZ.rds")

