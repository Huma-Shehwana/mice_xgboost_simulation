rm(list = ls())

dir <- "~/Documents/GitHub/mice_xgboost_simulation/results/"
setwd(dir)

#####################################################################################

get_stats <- function(df, var, meth){
  estimates <- df %>%
    map(~ .x %>% select({var})) %>%
    bind_cols() %>%
    {
      colnames(.) <- names(df)  # Set column names
      .  # Return the modified data frame
    } %>% 
    mutate("method" = meth) %>%  # Add 'method' column with default value
    rownames_to_column("term")
    
  return(estimates)
}

#####################################################################################

load("Default_evaluation_nonlinear.RData")
true_val <- eval_default$`20`$true
names(true_val) <- c("y", "x", "z", "z2", "x*z")
est_default <- get_stats(eval_default, "estimate", "Default")
bias_default <- get_stats(eval_default, "bias", "Default")
width_default <- get_stats(eval_default, "width", "Default")
cov_default <- get_stats(eval_default, "cov", "Default")



load("CART_evaluation_nonlinear.RData")
est_CART <- get_stats(eval_CART, "estimate", "CART")
bias_CART <- get_stats(eval_CART, "bias", "CART")
width_CART <- get_stats(eval_CART, "width", "CART")
cov_CART <- get_stats(eval_CART, "cov", "CART")


load("RF_evaluation_nonlinear.RData")
est_RF <- get_stats(eval_RF, "estimate", "RF")
bias_RF <- get_stats(eval_RF, "bias", "RF")
width_RF <- get_stats(eval_RF, "width", "RF")
cov_RF <- get_stats(eval_RF, "cov", "RF")


load("xgb_ParamDefault_maxit5_P_eval.RData")
est_xgb_default_maxit5_P <- get_stats(xgb_ParamDefault_maxit5_P_eval, "estimate", "XGB_Param.default_maxit5_Predicted")
bias_xgb_default_maxit5_P <- get_stats(xgb_ParamDefault_maxit5_P_eval, "bias", "XGB_Param.default_maxit5_Predicted")
width_xgb_default_maxit5_P <- get_stats(xgb_ParamDefault_maxit5_P_eval, "width", "XGB_Param.default_maxit5_Predicted")
cov_xgb_default_maxit5_P <- get_stats(xgb_ParamDefault_maxit5_P_eval, "cov", "XGB_Param.default_maxit5_Predicted")


load("xgb_ParamDefault_maxit5_PO_eval.RData")
est_xgb_default_maxit5_PO <- get_stats(xgb_ParamDefault_maxit5_PO_eval, "estimate", "XGB_Param.default_maxit5_Predicted.Observed")
bias_xgb_default_maxit5_PO <- get_stats(xgb_ParamDefault_maxit5_PO_eval, "bias", "XGB_Param.default_maxit5_Predicted.Observed")
width_xgb_default_maxit5_PO <- get_stats(xgb_ParamDefault_maxit5_PO_eval, "width", "XGB_Param.default_maxit5_Predicted.Observed")
cov_xgb_default_maxit5_PO <- get_stats(xgb_ParamDefault_maxit5_PO_eval, "cov", "XGB_Param.default_maxit5_Predicted.Observed")


load("xgb_ParamDefault_maxit5_OO_eval.RData")
est_xgb_default_maxit5_OO <- get_stats(xgb_ParamDefault_maxit5_OO_eval, "estimate","XGB_Param.default_maxit5_Original.Observed")
bias_xgb_default_maxit5_OO <- get_stats(xgb_ParamDefault_maxit5_OO_eval, "bias","XGB_Param.default_maxit5_Original.Observed")
width_xgb_default_maxit5_OO <- get_stats(xgb_ParamDefault_maxit5_OO_eval, "width","XGB_Param.default_maxit5_Original.Observed")
cov_xgb_default_maxit5_OO <- get_stats(xgb_ParamDefault_maxit5_OO_eval, "cov","XGB_Param.default_maxit5_Original.Observed")

load("xgb_ParamRandom_maxit5_P_imp_eval.RData")
est_xgb_Random_maxit5_P <- get_stats(xgb_ParamRandom_maxit5_P_imp_eval, "estimate", "XGB_Param.random_maxit5_Predicted")
bias_xgb_Random_maxit5_P <- get_stats(xgb_ParamRandom_maxit5_P_imp_eval, "bias", "XGB_Param.random_maxit5_Predicted")
width_xgb_Random_maxit5_P <- get_stats(xgb_ParamRandom_maxit5_P_imp_eval, "width", "XGB_Param.random_maxit5_Predicted")
cov_xgb_Random_maxit5_P <- get_stats(xgb_ParamRandom_maxit5_P_imp_eval, "cov", "XGB_Param.random_maxit5_Predicted")


load("xgb_ParamRandom_maxit5_PO_imp_eval.RData")
est_xgb_Random_maxit5_PO <- get_stats(xgb_ParamRandom_maxit5_PO_imp_eval, "estimate", "XGB_Param.random_maxit5_Predicted.Observed")
bias_xgb_Random_maxit5_PO <- get_stats(xgb_ParamRandom_maxit5_PO_imp_eval, "bias", "XGB_Param.random_maxit5_Predicted.Observed")
width_xgb_Random_maxit5_PO <- get_stats(xgb_ParamRandom_maxit5_PO_imp_eval, "width", "XGB_Param.random_maxit5_Predicted.Observed")
cov_xgb_Random_maxit5_PO <- get_stats(xgb_ParamRandom_maxit5_PO_imp_eval, "cov", "XGB_Param.random_maxit5_Predicted.Observed")

load("xgb_ParamRandom_maxit5_OO_imp_eval.RData")
est_xgb_Random_maxit5_OO <- get_stats(xgb_ParamRandom_maxit5_OO_imp_eval, "estimate", "XGB_Param.random_maxit5_Original.Observed")
bias_xgb_Random_maxit5_OO <- get_stats(xgb_ParamRandom_maxit5_OO_imp_eval, "bias", "XGB_Param.random_maxit5_Original.Observed")
width_xgb_Random_maxit5_OO <- get_stats(xgb_ParamRandom_maxit5_OO_imp_eval, "width", "XGB_Param.random_maxit5_Original.Observed")
cov_xgb_Random_maxit5_OO <- get_stats(xgb_ParamRandom_maxit5_OO_imp_eval, "cov", "XGB_Param.random_maxit5_Original.Observed")




load("xgb_ParamAll_maxit5_P_imp_eval.RData")
est_xgb_All_maxit5_P <- get_stats(xgb_ParamAll_maxit5_P_imp_eval, "estimate","XGB_Param.All_maxit5_Predicted")
bias_xgb_All_maxit5_P <- get_stats(xgb_ParamAll_maxit5_P_imp_eval, "bias","XGB_Param.All_maxit5_Predicted")
width_xgb_All_maxit5_P <- get_stats(xgb_ParamAll_maxit5_P_imp_eval, "width","XGB_Param.All_maxit5_Predicted")
cov_xgb_All_maxit5_P <- get_stats(xgb_ParamAll_maxit5_P_imp_eval, "cov","XGB_Param.All_maxit5_Predicted")


load("xgb_ParamAll_maxit5_PO_imp_eval.RData")
est_xgb_All_maxit5_PO <- get_stats(xgb_ParamAll_maxit5_PO_imp_eval, "estimate", "XGB_Param.All_maxit5_Predicted.Observed")
bias_xgb_All_maxit5_PO <- get_stats(xgb_ParamAll_maxit5_PO_imp_eval, "bias", "XGB_Param.All_maxit5_Predicted.Observed")
width_xgb_All_maxit5_PO <- get_stats(xgb_ParamAll_maxit5_PO_imp_eval, "width", "XGB_Param.All_maxit5_Predicted.Observed")
cov_xgb_All_maxit5_PO <- get_stats(xgb_ParamAll_maxit5_PO_imp_eval, "cov", "XGB_Param.All_maxit5_Predicted.Observed")


load("xgb_ParamAll_maxit5_OO_imp_eval.RData")
est_xgb_All_maxit5_OO <- get_stats(xgb_ParamAll_maxit5_OO_imp_eval, "estimate", "XGB_Param.All_maxit5_Original.Observed")
bias_xgb_All_maxit5_OO <- get_stats(xgb_ParamAll_maxit5_OO_imp_eval, "bias", "XGB_Param.All_maxit5_Original.Observed")
width_xgb_All_maxit5_OO <- get_stats(xgb_ParamAll_maxit5_OO_imp_eval, "width", "XGB_Param.All_maxit5_Original.Observed")
cov_xgb_All_maxit5_OO <- get_stats(xgb_ParamAll_maxit5_OO_imp_eval, "cov", "XGB_Param.All_maxit5_Original.Observed")


load("mixgb_maxit5_pmmNULL_eval.RData")
est_mixgb_maxit5_pmmNull <- get_stats(mixgb_maxit5_pmmNULL_eval, "estimate","Mixgb_maxit5_Pmm.Null")
bias_mixgb_maxit5_pmmNull <- get_stats(mixgb_maxit5_pmmNULL_eval, "bias","Mixgb_maxit5_Pmm.Null")
width_mixgb_maxit5_pmmNull <- get_stats(mixgb_maxit5_pmmNULL_eval, "width","Mixgb_maxit5_Pmm.Null")
cov_mixgb_maxit5_pmmNull <- get_stats(mixgb_maxit5_pmmNULL_eval, "cov","Mixgb_maxit5_Pmm.Null")


load("mixgb_maxit5_pmm1_eval.RData")
est_mixgb_maxit5_pmm1 <- get_stats(mixgb_maxit5_pmm1_eval, "estimate", "Mixgb_maxit5_Pmm.1")
bias_mixgb_maxit5_pmm1 <- get_stats(mixgb_maxit5_pmm1_eval, "bias", "Mixgb_maxit5_Pmm.1")
width_mixgb_maxit5_pmm1 <- get_stats(mixgb_maxit5_pmm1_eval, "width", "Mixgb_maxit5_Pmm.1")
cov_mixgb_maxit5_pmm1 <- get_stats(mixgb_maxit5_pmm1_eval, "cov", "Mixgb_maxit5_Pmm.1")


load("mixgb_maxit5_pmm2_eval.RData")
est_mixgb_maxit5_pmm2 <- get_stats(mixgb_maxit5_pmm2_eval, "estimate", "Mixgb_maxit5_Pmm.2")
bias_mixgb_maxit5_pmm2 <- get_stats(mixgb_maxit5_pmm2_eval, "bias", "Mixgb_maxit5_Pmm.2")
width_mixgb_maxit5_pmm2 <- get_stats(mixgb_maxit5_pmm2_eval, "width", "Mixgb_maxit5_Pmm.2")
cov_mixgb_maxit5_pmm2 <- get_stats(mixgb_maxit5_pmm2_eval, "cov", "Mixgb_maxit5_Pmm.2")




all_est <- bind_rows(est_default, est_CART, est_RF, 
          est_xgb_default_maxit5_P, est_xgb_default_maxit5_PO, est_xgb_default_maxit5_OO, 
          est_xgb_Random_maxit5_P, est_xgb_Random_maxit5_PO, est_xgb_Random_maxit5_OO, 
          est_xgb_All_maxit5_P, est_xgb_All_maxit5_PO,est_xgb_All_maxit5_OO,
          est_mixgb_maxit5_pmmNull, est_mixgb_maxit5_pmm1, est_mixgb_maxit5_pmm2)


all_bias <- bind_rows(bias_default, bias_CART, bias_RF, 
                      bias_xgb_default_maxit5_P, bias_xgb_default_maxit5_PO, bias_xgb_default_maxit5_OO, 
                      bias_xgb_Random_maxit5_P, bias_xgb_Random_maxit5_PO, bias_xgb_Random_maxit5_OO, 
                      bias_xgb_All_maxit5_P, bias_xgb_All_maxit5_PO,bias_xgb_All_maxit5_OO,
                      bias_mixgb_maxit5_pmmNull, bias_mixgb_maxit5_pmm1, bias_mixgb_maxit5_pmm2)


all_width <- bind_rows(width_default, width_CART, width_RF, 
                       width_xgb_default_maxit5_P, width_xgb_default_maxit5_PO, width_xgb_default_maxit5_OO, 
                       width_xgb_Random_maxit5_P, width_xgb_Random_maxit5_PO, width_xgb_Random_maxit5_OO, 
                       width_xgb_All_maxit5_P, width_xgb_All_maxit5_PO,width_xgb_All_maxit5_OO,
                       width_mixgb_maxit5_pmmNull, width_mixgb_maxit5_pmm1, width_mixgb_maxit5_pmm2)


all_cov <- bind_rows(cov_default, cov_CART, cov_RF, 
                     cov_xgb_default_maxit5_P, cov_xgb_default_maxit5_PO, cov_xgb_default_maxit5_OO, 
                     cov_xgb_Random_maxit5_P, cov_xgb_Random_maxit5_PO, cov_xgb_Random_maxit5_OO, 
                     cov_xgb_All_maxit5_P, cov_xgb_All_maxit5_PO,cov_xgb_All_maxit5_OO,
                     cov_mixgb_maxit5_pmmNull, cov_mixgb_maxit5_pmm1, cov_mixgb_maxit5_pmm2)


est_x <- all_est[all_est$term=="x",]
table_format_est_x <-  est_x %>%
  pivot_longer(cols = c("20", "40", "60", "80"), names_to = "variable", values_to = "value") %>%
  pivot_wider(names_from = "method", values_from = "value")


est_z <- all_est[all_est$term=="z",]
table_format_est_z <-  est_z %>%
  pivot_longer(cols = c("20", "40", "60", "80"), names_to = "variable", values_to = "value") %>%
  pivot_wider(names_from = "method", values_from = "value")


est_intercept <- all_est[all_est$term=="(Intercept)",]
table_format_intercept_z <-  est_intercept %>%
          pivot_longer(cols = c("20", "40", "60", "80"), names_to = "variable", values_to = "value") %>%
          pivot_wider(names_from = "method", values_from = "value")


est_xz <- all_est[all_est$term=="x:z",]
table_format_est_xz <-  est_xz %>%
          pivot_longer(cols = c("20", "40", "60", "80"), names_to = "variable", values_to = "value") %>%
          pivot_wider(names_from = "method", values_from = "value")

est_zsquare <- all_est[all_est$term=="I(z^2)",]
table_format_est_zsquare <-  est_zsquare %>%
  pivot_longer(cols = c("20", "40", "60", "80"), names_to = "variable", values_to = "value") %>%
  pivot_wider(names_from = "method", values_from = "value")


bias_x <- all_bias[all_bias$term=="x",]
bias_z <- all_bias[all_bias$term=="z",]
bias_intercept <- all_bias[all_bias$term=="(Intercept)",]
bias_xz <- all_bias[all_bias$term=="x:z",]
bias_zsquare <- all_bias[all_bias$term=="I(z^2)",]

width_x <- all_width[all_width$term=="x",]
width_z <- all_width[all_width$term=="z",]
width_intercept <- all_width[all_width$term=="(Intercept)",]
width_xz <- all_width[all_width$term=="x:z",]
width_zsquare <- all_width[all_width$term=="I(z^2)",]

cov_x <- all_cov[all_cov$term=="x",]
cov_z <- all_cov[all_cov$term=="z",]
cov_intercept <- all_cov[all_cov$term=="(Intercept)",]
cov_xz <- all_cov[all_cov$term=="x:z",]
cov_zsquare <- all_cov[all_cov$term=="I(z^2)",]





forest_long <- all_est_x %>%
  pivot_longer(
    cols = c(`20`, `40`, `60`, `80`),
    names_to = "missing_point",
    values_to = "width"
  )
          



forest_plot <- ggplot(forest_long, aes(y = method, x = width)) +
  geom_point() +
  geom_errorbarh(aes(xmin = width, xmax = width), height = 0.2) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  facet_wrap(~ missing_point, scales = "free_x") +
  theme_minimal() +
  labs(
    title = "Forest Plot by Time Point",
    x = "width",
    y = "Method"
  )

forest_plot


