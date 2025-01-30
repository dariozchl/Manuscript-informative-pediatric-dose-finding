data{
  int<lower=1> J; // number of compounds
  int<lower=1> K; // number of dose levels - has to be the same in each compound and for both pediatric and adult
  real<lower=0> doses[J,K]; // doses can be different for each compound but have to be the same for pediatric and adults within one compound
  int<lower=0> N_P[J,K]; // number of pediatric patients
  int<lower=0> Y_P[J,K]; // number of pediatric events
  int<lower=0> N_A[J,K]; // number of adult patients
  int<lower=0> Y_A[J,K]; // number of adult events
  int<lower=0> dR[J]; // reference dose
  real<lower=0> tau_priors[2]; // shape and rate parameters of gamma prior for tau
  real mu_means[2]; // means of normal prior for mu
  real<lower=0> mu_sd; // sd of normal prior for mu
  }

parameters{
  matrix[J,2] mu; 
  matrix[J,2] mu_A; 
  matrix[J,2] mu_P; 
  matrix[J,2] logalphabeta_A; // parameters adult
  matrix[J,2] logalphabeta_P; // parameters pediatric
  real<lower=0> tau_alpha; // heterogeneity parameter
  real<lower=0> tau_beta; // heterogeneity parameter
  real<lower=0,upper=1> similarity_parameter_alpha;   // Mixing proportion
  real<lower=0,upper=1> similarity_parameter_beta;   // Mixing proportion
}

transformed parameters{
  real<lower=0,upper=1> p_A[J,K];
  real<lower=0,upper=1> p_P[J,K];

  for(j in 1:J){
    for(k in 1:K){
      p_A[j,k] = inv_logit(logalphabeta_A[j,1] + exp(logalphabeta_A[j,2]) * log(doses[j,k]/dR[j]));
    }
  }
  for(j in 1:J){
    for(k in 1:K){
      p_P[j,k] = inv_logit(logalphabeta_P[j,1] + exp(logalphabeta_P[j,2]) * log(doses[j,k]/dR[j]));
    }
  }
}

model{
  for(j in 1:J){
    target += log_mix(similarity_parameter_alpha, normal_lpdf(logalphabeta_A[j,1] | mu[j,1], tau_alpha), normal_lpdf(logalphabeta_A[j,1] | mu_A[j,1], tau_alpha)); 
    target += log_mix(similarity_parameter_alpha, normal_lpdf(logalphabeta_P[j,1] | mu[j,1], tau_alpha), normal_lpdf(logalphabeta_P[j,1] | mu_P[j,1], tau_alpha)); 
    target += log_mix(similarity_parameter_beta, normal_lpdf(logalphabeta_A[j,2] | mu[j,2], tau_beta), normal_lpdf(logalphabeta_A[j,2] | mu_A[j,2], tau_beta)); 
    target += log_mix(similarity_parameter_beta, normal_lpdf(logalphabeta_P[j,2] | mu[j,2], tau_beta), normal_lpdf(logalphabeta_P[j,2] | mu_P[j,2], tau_beta)); 
    for (k in 1:K) {
      target += binomial_logit_lpmf(Y_A[j,k] | N_A[j,k], logalphabeta_A[j,1] + exp(logalphabeta_A[j,2])*log(doses[j,k]/dR[j]));
      target += binomial_logit_lpmf(Y_P[j,k] | N_P[j,k], logalphabeta_P[j,1] + exp(logalphabeta_P[j,2])*log(doses[j,k]/dR[j]));
    }
    target += normal_lpdf(mu[j,1] | mu_means[1], mu_sd); // same priors for all mu
    target += normal_lpdf(mu[j,2] | mu_means[2], mu_sd);
    target += normal_lpdf(mu_A[j,1] | mu_means[1], mu_sd);
    target += normal_lpdf(mu_A[j,2] | mu_means[2], mu_sd);
    target += normal_lpdf(mu_P[j,1] | mu_means[1], mu_sd);
    target += normal_lpdf(mu_P[j,2] | mu_means[2], mu_sd);
  }
  target += beta_lpdf(similarity_parameter_alpha | 0.5, 0.5);
  target += beta_lpdf(similarity_parameter_beta | 0.5, 0.5);
  target += gamma_lpdf(tau_alpha | tau_priors[1], tau_priors[2]);
  target += gamma_lpdf(tau_beta | tau_priors[1], tau_priors[2]);
}
