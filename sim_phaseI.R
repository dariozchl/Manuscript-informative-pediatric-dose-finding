sim.phaseI <- function(doses, target.tox, true.tox,
                       sample.size, cohort.size, 
                       stopping.rule, escalation.rule,
                       stanmodel, stan_data, estimate_similarity_parameter=FALSE,
                       iterations, return_posterior=FALSE){ 
  
  library(rstan)
  library(doParallel)
  library(tidyverse)

  
  if (stopping.rule == "median") {
    stop.rule <- function(){median.tox[1] > 0.4}
  } else if(stopping.rule == "mean") {
    stop.rule <- function(){mean.tox[1] > 0.4}
  } else {return("error")}
  
  if (escalation.rule == "median") {
    escalation.rule <- function(){ifelse(which.min(abs(median.tox - target.tox)) > k.star, k.star+1, which.min(abs(median.tox - target.tox)))} 
  } else if(escalation.rule == "mean") {
    escalation.rule <- function(){ifelse(which.min(abs(mean.tox - target.tox)) > k.star, k.star+1, which.min(abs(mean.tox - target.tox)))}
  } else {return("error")}

    K <- length(doses)
    tox <- c(rep(0,length(doses)))
    notox <- c(rep(0,length(doses)))
    tox.cohort <- c(rep(0, length(1:(sample.size/cohort.size))))
    mean.tox <- 0 # if mean.tox starts with only one entry, the first k.star will always be the first dose level 
    k.star <- 1
    distr.tox <- c()
    q95.tox <- c()
    q05.tox <- c()
    
    
    for(cohort in 1:(sample.size/cohort.size)){
      # treat cohort of patients and count toxicities
      tox.cohort[cohort] <- rbinom(1, cohort.size, true.tox[k.star])
      tox[k.star] <- tox[k.star] + tox.cohort[cohort]
      notox[k.star] <- notox[k.star] + cohort.size - tox.cohort[cohort]
      
      # count patients treated so far
      N <- tox + notox
      
      # initialize data
      data <- c(list("tox"=tox), list("N"=N), list("doses"=doses), stan_data)
      
      # run STAN model      
      fit <- sampling(stanmodel, data = data, warmup = 1000, iter = iterations, chains = 4, cores = 1, thin = 1, refresh = 0, control=list(adapt_delta=0.95, max_treedepth=15))
      
      # calculate mean toxicity
      # mean.tox <- colMeans(rstan::extract(fit)$p)
      mean.tox <- inv_logit(mean(rstan::extract(fit)$logalpha) + exp(mean(rstan::extract(fit)$logbeta)) * log(doses/dR))
      
      # determine the dose for the following cohort
      k.star <- escalation.rule()
      
      # STOP criterion
      if (stop.rule()) k.star = 0
      if (stop.rule()) break
    }
    
    # define the MTD
    MTD <- k.star
    
    # store estimated similarity parameter if applicable
    if(estimate_similarity_parameter==TRUE){
      similarity_parameter <- rstan::extract(fit)$borrowing_parameter
      similarity_parameter_mean <- mean(similarity_parameter)
      similarity_parameter_q95 <- quantile(similarity_parameter, probs=0.95, names=FALSE)
      similarity_parameter_q05 <- quantile(similarity_parameter, probs=0.05, names=FALSE)
    } else {
      similarity_parameter <- NA
      similarity_parameter_mean <- NA
      similarity_parameter_q95 <- NA
      similarity_parameter_q05 <- NA
    }

    
    if(return_posterior==FALSE){
      return(as_tibble(data.frame("dose.level" = 1:K, "doses" = doses, "mean.tox" = round(mean.tox,3), "true.tox" = true.tox, "toxicities.per.dose" = tox, "patients.per.dose" = N,
                        "MTD.dose.level" = k.star, "sample.size" = sample.size, 
                        "similarity_parameter_mean"=similarity_parameter_mean, "similarity_parameter_q95"=similarity_parameter_q95, "similarity_parameter_q05"=similarity_parameter_q05)))
    } else {
        return(as_tibble(cbind(data.frame("dose.level" = 1:K, "doses" = doses, "mean.tox" = round(mean.tox,3), "true.tox" = true.tox, "toxicities.per.dose" = tox, "patients.per.dose" = N,
                               "MTD.dose.level" = k.star, "sample.size" = sample.size) %>%
                       pivot_wider(id_cols=c(MTD.dose.level, sample.size),names_from=dose.level,values_from=c(doses, mean.tox, true.tox, toxicities.per.dose, patients.per.dose),names_sep=""),
                     "borrowing_parameter"=similarity_parameter)))
      }
}
