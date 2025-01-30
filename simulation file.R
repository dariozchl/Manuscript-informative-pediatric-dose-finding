library(rstan)
library(tidyverse)
library(R2jags)
library(foreach)
library(doParallel)
library(MASS) # for ginv()
library(ggh4x) # to allow for free axes with facet_grid2()
library(ggpubr)


exnex_meta_model <- stan_model("ExNexMetaModel.stan")

metaModel.jags <- "model {

tau1 ~ dgamma(1,0.1)I(0.00001,) 
tau2 ~ dgamma(1,0.1)I(0.00001,) 

Tau[1,1]<-1/tau1
Tau[2,2]<-1/tau2
Tau[1,2]<-0
Tau[2,1]<-0

cov[1,1] ~ dgamma(1,0.1)
cov[2,2] ~ dgamma(1,0.1)
rho ~ dunif(0,1)
cov[1,2] <- sqrt(cov[1,1])*sqrt(cov[2,2])*rho
cov[2,1] <- cov[1,2]
Sigma[1:2,1:2] <- inverse(cov[1:2,1:2])


for (j in 1:nr_compounds){
  log.alphabetaa[((j-1)*2+1):((j-1)*2+2)] ~ dmnorm(mu_j[j,1:2],Tau[1:2,1:2])
  alphabetaa[(j-1)*2+1] <- exp(log.alphabetaa[(j-1)*2+1])
  alphabetaa[(j-1)*2+2] <- exp(log.alphabetaa[(j-1)*2+2])
  
  log.alphabetap[((j-1)*2+1):((j-1)*2+2)] ~ dmnorm(mu_j[j,1:2],Tau[1:2,1:2])
  alphabetap[(j-1)*2+1] <- exp(log.alphabetap[(j-1)*2+1])
  alphabetap[(j-1)*2+2] <- exp(log.alphabetap[(j-1)*2+2])
  
  mu_j[j,1:2] ~ dmnorm(mu,Sigma)
}
mu ~ dmnorm(mu_prior, priorSigma)

# sampling model
for (j in 1:nr_compounds){
  for (jj in 1:dl){
    logit(Pr.Toxa[(j-1)*dl+jj])<- log.alphabetaa[(j-1)*2+1]+alphabetaa[(j-1)*2+2]*log(dosevec[(j-1)*dl+jj]/dR)
    Ntoxa[(j-1)*dl+jj] ~ dbin(Pr.Toxa[(j-1)*dl+jj],Npata[(j-1)*dl+jj])
    logit(Pr.Toxp[(j-1)*dl+jj])<- log.alphabetap[(j-1)*2+1]+alphabetap[(j-1)*2+2]*log(dosevec[(j-1)*dl+jj]/dR)
    Ntoxp[(j-1)*dl+jj] ~ dbin(Pr.Toxp[(j-1)*dl+jj],Npatp[(j-1)*dl+jj])
  }
}

}"


inv_logit <- function(x) exp(x)/(1+exp(x))
dose_tox_fun <- function(doses, dR, logalpha, logbeta) {inv_logit(logalpha + exp(logbeta)*log(doses/dR))}

data_generator <- function(sample_size_ped, sample_size_adults, nr_compounds, doses, dR, 
                           similarity_scenario,
                           mu_means, mu_sd, tau_priors){
  # define data generating dose-toxicity function
  alpha_pair <- c(-2.5, -0.84)
  beta_pair <- c(0, 1.5)
  data_generating_scenario <- data.frame("similarity_scenario"=rep(similarity_scenario,nr_compounds), "nr_compounds"=nr_compounds) %>% 
    rowwise() %>% mutate(logalpha_adult = sample(x=alpha_pair, size=1), logbeta_adult = sample(x=beta_pair, size=1)) %>% 
    rowwise() %>% mutate(logalpha_ped = case_when(grepl(pattern="alpha.equal", x=similarity_scenario) ~ logalpha_adult,
                                                  grepl(pattern="alpha.diff", x=similarity_scenario) ~ alpha_pair[!alpha_pair==logalpha_adult])) %>% 
    rowwise() %>% mutate(logbeta_ped = case_when(grepl(pattern="beta.equal", x=similarity_scenario) ~ logbeta_adult,
                                                 grepl(pattern="beta.diff", x=similarity_scenario) ~ beta_pair[!beta_pair==logbeta_adult]))
  
  # create number of patients and toxicities matrices
  Npata <- matrix(rep(rep(ceiling(sample_size_adults/length(doses)),length(doses)), nr_compounds), nrow=nr_compounds, byrow=TRUE)
  Npatp <- matrix(rep(rep(ceiling(sample_size_ped/length(doses)),length(doses)), nr_compounds), nrow=nr_compounds, byrow=TRUE)
  
  Ntoxp_vector <- Ntoxa_vector <- c()
  for(j in 1:nr_compounds){
    Ntoxp_vector <- c(Ntoxp_vector, rbinom(n=length(doses), size=ceiling(sample_size_ped/length(doses)), prob=dose_tox_fun(doses=doses, dR=dR, logalpha=data_generating_scenario$logalpha_ped[j], logbeta=data_generating_scenario$logbeta_ped[j])))
    Ntoxa_vector <- c(Ntoxa_vector, rbinom(n=length(doses), size=ceiling(sample_size_adults/length(doses)), prob=dose_tox_fun(doses=doses, dR=dR, logalpha=data_generating_scenario$logalpha_adult[j], logbeta=data_generating_scenario$logbeta_adult[j])))
  }
  Ntoxp <- matrix(Ntoxp_vector, nrow=nr_compounds, byrow=TRUE)
  Ntoxa <- matrix(Ntoxa_vector, nrow=nr_compounds, byrow=TRUE)
  
  # wrap up all in data that can be passed to stan
  data_stan <- list("J"=nr_compounds, "K"=length(doses), "doses"=matrix(rep(doses,nr_compounds),nrow=nr_compounds,byrow=TRUE), "dR"=rep(dR,nr_compounds), 
                    "Y_P"=Ntoxp, "Y_A"=Ntoxa, "N_P"=Npatp, "N_A"=Npata,  mu_means=mu_means, mu_sd=mu_sd, tau_priors=tau_priors)
  return(data_stan)
}


simulator_fun <- function(sample_size_ped, sample_size_adults, nr_compounds, doses, dR, 
                          similarity_scenario,
                          mu_means=c(-0.84,0), mu_sd=1.5, tau_priors=c(1,1/10)){
  
  ### generate data
  data_stan <- data_generator(sample_size_ped=sample_size_ped, sample_size_adults=sample_size_adults, nr_compounds=nr_compounds, doses=doses, dR=dR, 
                              similarity_scenario=similarity_scenario,
                              mu_means=mu_means, mu_sd=mu_sd, tau_priors=tau_priors)
  
  warmup <- ifelse(nr_compounds>10, 500, 1000)
  iter <- ifelse(nr_compounds>10, 1000, 2000)
  ### Stan part
  # fit <- sampling(meta_model, data = data_stan, warmup = warmup, iter = iter, chains = 4, cores = 1, control=list(adapt_delta=0.95, max_treedepth=15), refresh=500)
  # est <- summary(fit, pars=c("tau_alpha", "tau_beta", "similarity_parameter_alpha", "similarity_parameter_beta"))$summary
  
  ### ExNex model
  fit2 <- sampling(exnex_meta_model, data = data_stan, warmup = warmup, iter = iter, chains = 4, cores = 1, control=list(adapt_delta=0.95, max_treedepth=15), refresh=500)
  estExNex <- summary(fit2, pars=c("tau_alpha", "tau_beta", "similarity_parameter_alpha", "similarity_parameter_beta"))$summary
  
  ### JAGS part
  mu_prior=mu_means #mu=matrix(rep(mu_means,nr_compounds),byrow=TRUE,ncol=2); 
  priorSigma=ginv(matrix(c(mu_sd^2,0,0,mu_sd^2),2,2))
  dl=length(doses); dR=dR; dosevec=rep(doses,nr_compounds); nr_compounds=data_stan$J 
  Ntoxa=as.vector(t(data_stan$Y_A)); Ntoxp=as.vector(t(data_stan$Y_P)); Npata=as.vector(t(data_stan$N_A)); Npatp=as.vector(t(data_stan$N_P))
  jags <- list("nr_compounds","dl","Ntoxa","Ntoxp","Npata","Npatp","mu_prior","dosevec","dR","priorSigma")
  metaModel.jags.params<-c("cov[1,1]", "cov[2,2]", "tau1", "tau2")
  metaModel.jags.fit <- jags(data = jags, inits = NULL,parameters.to.save = metaModel.jags.params,n.chains = 4, n.iter = 10000,n.burnin = 2000,model.file = textConnection(metaModel.jags))
  posts<-as.data.frame(as.matrix(as.mcmc(metaModel.jags.fit)))
  term1<-(posts$tau1)/rowSums(posts %>% dplyr::select(`cov[1,1]`))
  term2<-(posts$tau2)/rowSums(posts %>% dplyr::select(`cov[2,2]`))
  similarity_parameter_alpha <- 1/(term1+1)
  similarity_parameter_beta <- 1/(term2+1)
  
  return(rbind(#as.data.frame(est[,c("2.5%","50%","97.5%","mean")]) %>% rownames_to_column(var="param") %>% add_column("model"="stan_metaModel"), 
    as.data.frame(estExNex[,c("2.5%","50%","97.5%","mean")]) %>% rownames_to_column(var="param") %>% add_column("model"="stan_ExNexModel"), 
    as.data.frame(rbind("tau_alpha"=c(quantile(posts$tau1, probs=c(0.025,0.5,0.975)),"mean"=mean(posts$tau1)), 
                        "tau_beta"=c(quantile(posts$tau2, probs=c(0.025,0.5,0.975)),"mean"=mean(posts$tau2)),
                        "similarity_parameter_alpha"=c(quantile(similarity_parameter_alpha, probs=c(0.025,0.5,0.975)),"mean"=mean(similarity_parameter_alpha)), 
                        "similarity_parameter_beta"=c(quantile(similarity_parameter_beta, probs=c(0.025,0.5,0.975)),"mean"=mean(similarity_parameter_beta)))) %>% 
      rownames_to_column(var="param") %>% add_column("model"="jags")) %>% 
      add_column("similarity_scenario"=similarity_scenario,
                 "sample_size_adults"=sample_size_adults, "sample_size_ped"=sample_size_ped,
                 "nr_compounds"=nr_compounds))
}


####################################################################################################################################

### Simulation Part 1 - estimating the similarity parameters from other compounds

# because of the big computational effort, this simulation is split up in sets of 100 simulation runs, 
# but will be repeated 10 times --> 1,000 simulation runs in total.  
nsim=1e2 # 100 simulation runs took approximately 12-14 hours when distributed to 15 threads on an Intel i7-11700k CPU

sample_size_adults <- c(30, 100)
sample_size_ped <- c(30, 100)
nr_compounds <- c(3, 7, 20)
similarity_scenarios <- c("alpha.equal_beta.equal", "alpha.diff_beta.equal", "alpha.equal_beta.diff", "alpha.diff_beta.diff")

expand.grid.df <- function(...) Reduce(function(...) merge(..., by=NULL), list(...)) # expand.grid for data.frames
scenarios <- expand.grid.df(data.frame("sample_size_ped"=sample_size_ped,"sample_size_adults"=sample_size_adults), data.frame("nr_compounds"=nr_compounds), data.frame("similarity_scenarios"=similarity_scenarios)) 
scenarios <- scenarios %>% filter(sample_size_ped == sample_size_adults)

results <- data.frame()

scenarios.sims <- expand.grid.df(scenarios %>% rownames_to_column("scenarioID"), data.frame("simID"=1:nsim))

for(repetition in 1:10){
  
  ncores <- detectCores()-1; 
  cl <- makeCluster(ncores); registerDoParallel(cl);
  start.time <- Sys.time()
  
  scenario.start.time <- Sys.time()
  results <- foreach(sim=1:nrow(scenarios.sims), .combine=rbind, .packages=c("tidyverse", "rstan", "R2jags", "MASS")) %dopar% {
    simulator_fun(sample_size_ped=scenarios.sims$sample_size_ped[sim], 
                  sample_size_adults=scenarios.sims$sample_size_adults[sim], 
                  nr_compounds=scenarios.sims$nr_compounds[sim], 
                  similarity_scenario=scenarios.sims$similarity_scenarios[sim], doses=c(1, 3, 10, 15, 20), dR=10)
  }
  
  
  # writeLines(paste0("scenario: ", scenario, "\nrun time for this scenario: ", 
  #                   round(as.numeric(difftime(Sys.time(), scenario.start.time, units="min")),2), " Minutes\nrun time in total: ", 
  #                   round(as.numeric(difftime(Sys.time(), start.time, units="min")),2), " Minutes"))
  
  Sys.time() - start.time
  stopCluster(cl) # end parallel computing
  
  saveRDS(results, file=paste("meta_analysis_results", repetition*100, ".Rds", sep = ""))
}



results <- bind_rows(
  readRDS("meta_analysis_results100.Rds"),
  readRDS("meta_analysis_results200.Rds"),
  readRDS("meta_analysis_results300.Rds"),
  readRDS("meta_analysis_results400.Rds"),
  readRDS("meta_analysis_results500.Rds"),
  readRDS("meta_analysis_results600.Rds"),
  readRDS("meta_analysis_results700.Rds"),
  readRDS("meta_analysis_results800.Rds"),
  readRDS("meta_analysis_results900.Rds"),
  readRDS("meta_analysis_results1000.Rds")
)


results <- results %>% mutate(sample_size_adults_ped=interaction(sample_size_adults, sample_size_ped))




##########################################################

# Simulation Part 2
# perform new pediatric trial with informative priors


## compile Stan models
BLRM_fixed <- stan_model("BLRM_fixed.stan")
BLRM_mixture_fixed <- stan_model("BLRM_mixture.stan")
BLRM_mixture_estimate <- stan_model("BLRM_mixture_estimated.stan")
BLRM_inflated_fixed <- stan_model("BLRM_inflated_var.stan")
BLRM_inflated_estimated <- stan_model("BLRM_inflated_var_estimated.stan")


## generate adult data from which to be borrowed

nsim=1e3
sample_size_adults=40
doses=c(1, 3, 10, 15, 20)
dR=10

dose_tox_scenarios <- expand.grid("alpha_adults"=c(-2.5, -0.84), "beta_adults"=c(0, 1.5), "alpha_ped"=c(-2.5, -0.84), "beta_ped"=c(0, 1.5)) %>% 
  rowwise() %>% 
  mutate(adult_tox=list(dose_tox_fun(doses=doses, dR=dR, logalpha=alpha_adults, logbeta=beta_adults)),
         ped_tox=list(dose_tox_fun(doses=doses, dR=dR, logalpha=alpha_ped, logbeta=beta_ped)),
         dose = list(1:5)) %>% 
  unnest(cols = c(adult_tox, ped_tox, dose)) %>% 
  rowwise() %>% 
  mutate(NpatA=ceiling(sample_size_adults/length(doses)), 
         NtoxA=list(rbinom(n=nsim, size=NpatA, prob=adult_tox)), adult_simID=list(1:nsim)) %>% 
  unnest(cols = c(NtoxA, adult_simID)) %>% 
  pivot_wider(names_from=dose, values_from=c(adult_tox, ped_tox, NpatA, NtoxA), names_prefix="dose") %>% 
  mutate(similarity_scenario = case_when(alpha_adults==alpha_ped & beta_adults==beta_ped ~ "alpha.equal_beta.equal",
                                         alpha_adults!=alpha_ped & beta_adults==beta_ped ~ "alpha.diff_beta.equal",
                                         alpha_adults==alpha_ped & beta_adults!=beta_ped ~ "alpha.equal_beta.diff",
                                         alpha_adults!=alpha_ped & beta_adults!=beta_ped ~ "alpha.diff_beta.diff"))


## transform adult data into a prior
# fit BLRM to the adult data

mu <- c(-0.84, 1)
Sigma <- matrix(c(2^2, 0, 0, 1.5^2), ncol=2)

informative_mu <- informative_Sigma <- list()

ncores <- detectCores()-1; 
cl <- makeCluster(ncores); registerDoParallel(cl);

start.time <- Sys.time()

prior_from_adults <- foreach(i=1:nrow(dose_tox_scenarios), .combine=rbind, .packages=c("tidyverse", "rstan", "R2jags", "MASS")) %dopar% {
  
  current_scenario = dose_tox_scenarios[i,]
  data <- list("doses"=doses, "dR"=dR, "K"=length(doses), 
               "tox"=current_scenario %>% dplyr::select(starts_with("NtoxA")) %>% unlist() %>% as.numeric(), 
               "N"=current_scenario %>% dplyr::select(starts_with("NpatA")) %>% unlist() %>% as.numeric(), 
               "mu" = mu, "Sigma" = Sigma)
  
  fit <- sampling(BLRM_fixed, data = data, warmup = 2000, iter = 5000, chains = 4, cores = 1, thin = 1, refresh = 0)
  
  # Derive informative priors. 
  posterior <- as_tibble(cbind(rstan::extract(fit)$logalphabeta[,1], rstan::extract(fit)$logalphabeta[,2]))
  
  # derive covariance matrix for informative prior
  return(tibble("informative_mu" = list(c(mean(posterior$V1), mean(posterior$V2))),
                "informative_Sigma" = list(posterior %>% cov())))
}

stopCluster(cl) # end parallel computing

Sys.time() - start.time

informative_mu <- prior_from_adults$informative_mu
informative_Sigma <- prior_from_adults$informative_Sigma



## run the pediatric trial with different model specifications across several scenarios

source("sim_phaseI.R")

# similarity parameter for inflated models need to be transformed because sigma^2/x is quite informative if x=0.5
# sigma^2/x^2 did not do the job because variance decreases too fast with x
# sigma^2*100*0.01^x, i.e., transforming the similarity parameter to 1/(100*0.01^x) seems to perform better, see e.g.:
# curve(0.3*100*0.01^x, from=0.1, to=1, xlab="x", ylab="y")
# if x=0, then the factor becomes 100
# if x=1, then the factor becomes 1
# and between 0 and 1, most values of x correspond to large factors. See the mapping: curve(1/(100*0.01^x), from=0.1, to=1, xlab="x", ylab="y")


sim_results <- list()

for(i in 1:10){
  ncores <- detectCores()-1; 
  cl <- makeCluster(ncores); registerDoParallel(cl);
  
  start.time <- Sys.time()
  
  sim_results[[i]] <- foreach(sim = ((i-1)*nrow(dose_tox_scenarios)/10+1): (i*nrow(dose_tox_scenarios)/10), .combine=rbind, .packages=c("tidyverse", "rstan", "R2jags", "MASS")) %dopar% {
    
    # fixed BLRM with weakly informative priors 
    fixed_weak <- sim.phaseI(doses=doses, target.tox=0.3, true.tox=dose_tox_scenarios[sim,] %>% dplyr::select(starts_with("ped_tox")) %>% unlist() %>% as.numeric(), 
                             estimate_similarity_parameter=FALSE,
                             sample.size=12, cohort.size=2, stopping.rule="mean", escalation.rule="mean",
                             stanmodel=BLRM_fixed,  
                             stan_data = list("dR"=dR, "K"=length(doses),
                                              "mu" = informative_mu[[sim]], "Sigma" = Sigma),
                             iterations=3000, return_posterior=FALSE)
    
    # fixed BLRM with strongly informative priors 
    fixed_strong <- sim.phaseI(doses=doses, target.tox=0.3, true.tox=dose_tox_scenarios[sim,] %>% dplyr::select(starts_with("ped_tox")) %>% unlist() %>% as.numeric(), 
                               estimate_similarity_parameter=FALSE,
                               sample.size=12, cohort.size=2, stopping.rule="mean", escalation.rule="mean",
                               stanmodel=BLRM_fixed,  
                               stan_data = list("dR"=dR, "K"=length(doses),
                                                "mu" = informative_mu[[sim]], "Sigma" = informative_Sigma[[sim]]),
                               iterations=3000, return_posterior=FALSE)
    
    # mixture BLRM with fixed weights of 0.5
    mixture_0.5 <- sim.phaseI(doses=doses, target.tox=0.3, true.tox=dose_tox_scenarios[sim,] %>% dplyr::select(starts_with("ped_tox")) %>% unlist() %>% as.numeric(), 
                              estimate_similarity_parameter=FALSE,
                              sample.size=12, cohort.size=2, stopping.rule="mean", escalation.rule="mean",
                              stanmodel=BLRM_mixture_fixed,  
                              stan_data = list("dR"=dR, "K"=length(doses),
                                               "mu_1" = informative_mu[[sim]], "mu_2" = informative_mu[[sim]], 
                                               "Sigma_1" = informative_Sigma[[sim]], "Sigma_2" = Sigma, similarity_parameter_alpha=0.5, similarity_parameter_beta=0.5),
                              iterations=3000, return_posterior=FALSE)
    
    # inflated variance BLRM with fixed weight of 0.5
    inflated_0.5 <- sim.phaseI(doses=doses, target.tox=0.3, true.tox=dose_tox_scenarios[sim,] %>% dplyr::select(starts_with("ped_tox")) %>% unlist() %>% as.numeric(),
                               estimate_similarity_parameter=FALSE,
                               sample.size=12, cohort.size=2, stopping.rule="mean", escalation.rule="mean",
                               stanmodel=BLRM_inflated_fixed,
                               stan_data = list("dR"=dR, "K"=length(doses),
                                                "mu" = informative_mu[[sim]],
                                                "sigma" = c(sqrt(informative_Sigma[[sim]][1,1]), sqrt(informative_Sigma[[sim]][2,2])),
                                                "rho" = informative_Sigma[[sim]][2,1] / (sqrt(informative_Sigma[[sim]][1,1]) * sqrt(informative_Sigma[[sim]][2,2])),
                                                "similarity_parameter_alpha"=0.5, "similarity_parameter_beta"=0.5),
                               iterations=3000, return_posterior=FALSE)
    
    
    ############################################################################################################  
    # sample size 100, 20 compounds
    ############################################################################################################
    results_subset <- results %>% 
      filter(grepl(pattern="similarity", x=param) & 
               model=="jags" & similarity_scenario==dose_tox_scenarios$similarity_scenario[sim] & 
               sample_size_adults==100 & nr_compounds==20) %>% 
      pivot_wider(names_from=param, values_from=c(`2.5%`, `50%`, `97.5%`, mean)) %>% unnest() 
    
    # mixture BLRM with weights from MetaModel
    mixture_metaModel_N100_20comp <- sim.phaseI(
      doses=doses, target.tox=0.3, true.tox=dose_tox_scenarios[sim,] %>% dplyr::select(starts_with("ped_tox")) %>% unlist() %>% as.numeric(), 
      estimate_similarity_parameter=FALSE,
      sample.size=12, cohort.size=2, stopping.rule="mean", escalation.rule="mean",
      stanmodel=BLRM_mixture_fixed,  
      stan_data = list("dR"=dR, "K"=length(doses),
                       "mu_1" = informative_mu[[sim]], "mu_2" = informative_mu[[sim]], 
                       "Sigma_1" = informative_Sigma[[sim]], "Sigma_2" = Sigma, 
                       similarity_parameter_alpha=sample(x=results_subset$mean_similarity_parameter_alpha,size=1), 
                       similarity_parameter_beta=sample(x=results_subset$mean_similarity_parameter_beta,size=1)),
      iterations=3000, return_posterior=FALSE)
    
    # inflated BLRM with weights from MetaModel
    inflated_metaModel_N100_20comp <- sim.phaseI(
      doses=doses, target.tox=0.3, true.tox=dose_tox_scenarios[sim,] %>% dplyr::select(starts_with("ped_tox")) %>% unlist() %>% as.numeric(),
      estimate_similarity_parameter=FALSE,
      sample.size=12, cohort.size=2, stopping.rule="mean", escalation.rule="mean",
      stanmodel=BLRM_inflated_fixed,
      stan_data = list("dR"=dR, "K"=length(doses),
                       "mu" = informative_mu[[sim]],
                       "sigma" = c(sqrt(informative_Sigma[[sim]][1,1]), sqrt(informative_Sigma[[sim]][2,2])),
                       "rho" = informative_Sigma[[sim]][2,1] / (sqrt(informative_Sigma[[sim]][1,1]) * sqrt(informative_Sigma[[sim]][2,2])),
                       similarity_parameter_alpha=sample(x=results_subset$mean_similarity_parameter_alpha,size=1), 
                       similarity_parameter_beta=sample(x=results_subset$mean_similarity_parameter_beta,size=1)),
      iterations=3000, return_posterior=FALSE)
    
    results_subset <- results %>% 
      filter(grepl(pattern="similarity", x=param) & 
               model=="stan_ExNexModel" & similarity_scenario==dose_tox_scenarios$similarity_scenario[sim] & 
               sample_size_adults==100 & nr_compounds==20) %>% 
      pivot_wider(names_from=param, values_from=c(`2.5%`, `50%`, `97.5%`, mean)) %>% unnest() 
    
    # mixture BLRM with weights from ExNexModel
    mixture_ExNexModel_N100_20comp <- sim.phaseI(
      doses=doses, target.tox=0.3, true.tox=dose_tox_scenarios[sim,] %>% dplyr::select(starts_with("ped_tox")) %>% unlist() %>% as.numeric(), 
      estimate_similarity_parameter=FALSE,
      sample.size=12, cohort.size=2, stopping.rule="mean", escalation.rule="mean",
      stanmodel=BLRM_mixture_fixed,  
      stan_data = list("dR"=dR, "K"=length(doses),
                       "mu_1" = informative_mu[[sim]], "mu_2" = informative_mu[[sim]], 
                       "Sigma_1" = informative_Sigma[[sim]], "Sigma_2" = Sigma, 
                       similarity_parameter_alpha=sample(x=results_subset$mean_similarity_parameter_alpha,size=1), 
                       similarity_parameter_beta=sample(x=results_subset$mean_similarity_parameter_beta,size=1)),
      iterations=3000, return_posterior=FALSE)
    
    # inflated BLRM with weights from ExNexModel
    inflated_ExNexModel_N100_20comp <- sim.phaseI(
      doses=doses, target.tox=0.3, true.tox=dose_tox_scenarios[sim,] %>% dplyr::select(starts_with("ped_tox")) %>% unlist() %>% as.numeric(),
      estimate_similarity_parameter=FALSE,
      sample.size=12, cohort.size=2, stopping.rule="mean", escalation.rule="mean",
      stanmodel=BLRM_inflated_fixed,
      stan_data = list("dR"=dR, "K"=length(doses),
                       "mu" = informative_mu[[sim]],
                       "sigma" = c(sqrt(informative_Sigma[[sim]][1,1]), sqrt(informative_Sigma[[sim]][2,2])),
                       "rho" = informative_Sigma[[sim]][2,1] / (sqrt(informative_Sigma[[sim]][1,1]) * sqrt(informative_Sigma[[sim]][2,2])),
                       similarity_parameter_alpha=sample(x=results_subset$mean_similarity_parameter_alpha,size=1), 
                       similarity_parameter_beta=sample(x=results_subset$mean_similarity_parameter_beta,size=1)),
      iterations=3000, return_posterior=FALSE)
    
    ############################################################################################################  
    # sample size 30, 7 compounds
    ############################################################################################################
    
    results_subset <- results %>% 
      filter(grepl(pattern="similarity", x=param) & 
               model=="jags" & similarity_scenario==dose_tox_scenarios$similarity_scenario[sim] & 
               sample_size_adults==30 & nr_compounds==7) %>% 
      pivot_wider(names_from=param, values_from=c(`2.5%`, `50%`, `97.5%`, mean)) %>% unnest() 
    
    # mixture BLRM with weights from MetaModel
    mixture_metaModel_N30_7comp <- sim.phaseI(
      doses=doses, target.tox=0.3, true.tox=dose_tox_scenarios[sim,] %>% dplyr::select(starts_with("ped_tox")) %>% unlist() %>% as.numeric(), 
      estimate_similarity_parameter=FALSE,
      sample.size=12, cohort.size=2, stopping.rule="mean", escalation.rule="mean",
      stanmodel=BLRM_mixture_fixed,  
      stan_data = list("dR"=dR, "K"=length(doses),
                       "mu_1" = informative_mu[[sim]], "mu_2" = informative_mu[[sim]], 
                       "Sigma_1" = informative_Sigma[[sim]], "Sigma_2" = Sigma, 
                       similarity_parameter_alpha=sample(x=results_subset$mean_similarity_parameter_alpha,size=1), 
                       similarity_parameter_beta=sample(x=results_subset$mean_similarity_parameter_beta,size=1)),
      iterations=3000, return_posterior=FALSE)
    
    # inflated BLRM with weights from MetaModel
    inflated_metaModel_N30_7comp <- sim.phaseI(
      doses=doses, target.tox=0.3, true.tox=dose_tox_scenarios[sim,] %>% dplyr::select(starts_with("ped_tox")) %>% unlist() %>% as.numeric(),
      estimate_similarity_parameter=FALSE,
      sample.size=12, cohort.size=2, stopping.rule="mean", escalation.rule="mean",
      stanmodel=BLRM_inflated_fixed,
      stan_data = list("dR"=dR, "K"=length(doses),
                       "mu" = informative_mu[[sim]],
                       "sigma" = c(sqrt(informative_Sigma[[sim]][1,1]), sqrt(informative_Sigma[[sim]][2,2])),
                       "rho" = informative_Sigma[[sim]][2,1] / (sqrt(informative_Sigma[[sim]][1,1]) * sqrt(informative_Sigma[[sim]][2,2])),
                       similarity_parameter_alpha=sample(x=results_subset$mean_similarity_parameter_alpha,size=1), 
                       similarity_parameter_beta=sample(x=results_subset$mean_similarity_parameter_beta,size=1)),
      iterations=3000, return_posterior=FALSE)
    
    results_subset <- results %>% 
      filter(grepl(pattern="similarity", x=param) & 
               model=="stan_ExNexModel" & similarity_scenario==dose_tox_scenarios$similarity_scenario[sim] & 
               sample_size_adults==30 & nr_compounds==7) %>% 
      pivot_wider(names_from=param, values_from=c(`2.5%`, `50%`, `97.5%`, mean)) %>% unnest() 
    
    # mixture BLRM with weights from ExNexModel
    mixture_ExNexModel_N30_7comp <- sim.phaseI(
      doses=doses, target.tox=0.3, true.tox=dose_tox_scenarios[sim,] %>% dplyr::select(starts_with("ped_tox")) %>% unlist() %>% as.numeric(), 
      estimate_similarity_parameter=FALSE,
      sample.size=12, cohort.size=2, stopping.rule="mean", escalation.rule="mean",
      stanmodel=BLRM_mixture_fixed,  
      stan_data = list("dR"=dR, "K"=length(doses),
                       "mu_1" = informative_mu[[sim]], "mu_2" = informative_mu[[sim]], 
                       "Sigma_1" = informative_Sigma[[sim]], "Sigma_2" = Sigma, 
                       similarity_parameter_alpha=sample(x=results_subset$mean_similarity_parameter_alpha,size=1), 
                       similarity_parameter_beta=sample(x=results_subset$mean_similarity_parameter_beta,size=1)),
      iterations=3000, return_posterior=FALSE)
    
    # inflated BLRM with weights from ExNexModel
    inflated_ExNexModel_N30_7comp <- sim.phaseI(
      doses=doses, target.tox=0.3, true.tox=dose_tox_scenarios[sim,] %>% dplyr::select(starts_with("ped_tox")) %>% unlist() %>% as.numeric(),
      estimate_similarity_parameter=FALSE,
      sample.size=12, cohort.size=2, stopping.rule="mean", escalation.rule="mean",
      stanmodel=BLRM_inflated_fixed,
      stan_data = list("dR"=dR, "K"=length(doses),
                       "mu" = informative_mu[[sim]],
                       "sigma" = c(sqrt(informative_Sigma[[sim]][1,1]), sqrt(informative_Sigma[[sim]][2,2])),
                       "rho" = informative_Sigma[[sim]][2,1] / (sqrt(informative_Sigma[[sim]][1,1]) * sqrt(informative_Sigma[[sim]][2,2])),
                       similarity_parameter_alpha=sample(x=results_subset$mean_similarity_parameter_alpha,size=1), 
                       similarity_parameter_beta=sample(x=results_subset$mean_similarity_parameter_beta,size=1)),
      iterations=3000, return_posterior=FALSE)
    
    
    return(bind_rows(fixed_weak %>% add_column("specification"="fixed_weak"), 
                     fixed_strong %>% add_column("specification"="fixed_strong"), 
                     mixture_0.5 %>% add_column("specification"="mixture_0.5"), 
                     inflated_0.5 %>% add_column("specification"="inflated_0.5"), 
                     #mixture_estimated %>% add_column("specification"="mixture_estimated"), 
                     mixture_metaModel_N100_20comp %>% add_column("specification"="mixture_metaModel_N100_20comp"), 
                     mixture_ExNexModel_N100_20comp %>% add_column("specification"="mixture_ExNexModel_N100_20comp"), 
                     inflated_metaModel_N100_20comp %>% add_column("specification"="inflated_metaModel_N100_20comp"), 
                     inflated_ExNexModel_N100_20comp %>% add_column("specification"="inflated_ExNexModel_N100_20comp"),
                     mixture_metaModel_N30_7comp %>% add_column("specification"="mixture_metaModel_N30_7comp"), 
                     mixture_ExNexModel_N30_7comp %>% add_column("specification"="mixture_ExNexModel_N30_7comp"), 
                     inflated_metaModel_N30_7comp %>% add_column("specification"="inflated_metaModel_N30_7comp"), 
                     inflated_ExNexModel_N30_7comp %>% add_column("specification"="inflated_ExNexModel_N30_7comp")) %>% 
             add_column("simID"=sim))
  }
  stopCluster(cl) # end parallel computing
  
  Sys.time() - start.time
  print(i)
  saveRDS(sim_results, file="sim_results.Rds")
}


