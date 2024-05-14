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


###########################################################################################################
############################     3.7 -Imputation using  mixgb    ##################################

##################        PMM = NULL      #############################

print("starting Mixgb Imputation")

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

mixgb_maxit5_pmmNULL_eval_linear <- mixgb_maxit5_pmmNULL_imp %>% 
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) { mat %>%
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

save(mixgb_maxit5_pmmNULL_imp, file = "Imputation/Mixgb/mixgb_maxit5_pmmNULL_imp.RData") # save imputation results
save(mixgb_maxit5_pmmNULL_time, file = "Imputation/Mixgb/mixgb_maxit5_pmmNULL_time.RData") # save time taken for imputation results
save(mixgb_maxit5_pmmNULL_eval, file = "Imputation/Mixgb/mixgb_maxit5_pmmNULL_eval.RData") # save regression results fitted on imputed data
save(mixgb_maxit5_pmmNULL_eval_linear, file = "Imputation/Mixgb/mixgb_maxit5_pmmNULL_eval_linear.RData") # save regression results fitted on imputed data


rm(mixgb_maxit5_pmmNULL_tmp, mixgb_maxit5_pmmNULL_imp, mixgb_maxit5_pmmNULL_time, mixgb_maxit5_pmmNULL_eval,mixgb_maxit5_pmmNULL_eval_linear)


####################################################### pmm = 1 ##############################################

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

mixgb_maxit5_pmm1_eval_linear <- mixgb_maxit5_pmm1_imp %>% 
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) { mat %>%
        map(~.x %$% # for every completed data set....
              lm(y ~ x + z)) %>% # fit linear model
        pool() %>%  # pool coefficients
        summary(conf.int = TRUE) %>% # summary of coefficients
        left_join(coef_true, by = "term") %>%
        mutate( cov = conf.low < true_vals & true_vals < conf.high, # coverage
                bias = estimate - true_vals,
                width = conf.high - conf.low) %>% # bias
        column_to_rownames("term")}, .options = furrr_options(seed = TRUE)) %>% # `term` as rownames
      Reduce("+", .) / Num_ds})

save(mixgb_maxit5_pmm1_imp, file = "Imputation/Mixgb/mixgb_maxit5_pmm1_imp.RData") # save imputation results
save(mixgb_maxit5_pmm1_time, file = "Imputation/Mixgb/mixgb_maxit5_pmm1_time.RData") # save time taken for imputation results
save(mixgb_maxit5_pmm1_eval, file = "Imputation/Mixgb/mixgb_maxit5_pmm1_eval.RData") # save regression results fitted on imputed data
save(mixgb_maxit5_pmm1_eval_linear, file = "Imputation/Mixgb/mixgb_maxit5_pmm1_eval_linear.RData") # save regression results fitted on imputed data


rm(mixgb_maxit5_pmm1_tmp, mixgb_maxit5_pmm1_imp, mixgb_maxit5_pmm1_time, mixgb_maxit5_pmm1_eval,mixgb_maxit5_pmm1_eval_linear)


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


mixgb_maxit5_pmm2_eval_linear <- mixgb_maxit5_pmm2_imp %>% 
  map(function(mat_list) {
    furrr::future_map(mat_list, function(mat) { mat %>%
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

save(mixgb_maxit5_pmm2_imp, file = "Imputation/Mixgb/mixgb_maxit5_pmm2_imp.RData") # save imputation results
save(mixgb_maxit5_pmm2_time, file = "Imputation/Mixgb/mixgb_maxit5_pmm2_time.RData") # save time taken for imputation results
save(mixgb_maxit5_pmm2_eval, file = "Imputation/Mixgb/mixgb_maxit5_pmm2_eval.RData") # save regression results fitted on imputed data
save(mixgb_maxit5_pmm2_eval_linear, file = "Imputation/Mixgb/mixgb_maxit5_pmm2_eval_linear.RData") # save regression results fitted on imputed data


rm(mixgb_maxit5_pmm2_tmp, mixgb_maxit5_pmm2_imp, mixgb_maxit5_pmm2_time, mixgb_maxit5_pmm2_eval, mixgb_maxit5_pmm2_eval_linear)

###########################################################################################################
###########################################################################################################
###########################################################################################################
