data {
  int<lower=1> K; // total number of dose levels 
  vector<lower=0>[K] doses; // predictor variable 
  int<lower=0> tox[K]; // number of tox per dose
  int<lower=0> N[K]; // total number of patients per dose level
  int<lower=0> dR; // reference dose
  vector[2] mu_1;
  vector[2] mu_2;
  cov_matrix[2] Sigma_1;
  cov_matrix[2] Sigma_2;
  real<lower=0,upper=1> similarity_parameter_alpha;
  real<lower=0,upper=1> similarity_parameter_beta;
}

parameters {
  vector[2] logalphabeta;
}

transformed parameters {
  real<lower=0,upper=1> p[K];
  real logalpha;
  real logbeta;
  
  logalpha = logalphabeta[1];
  logbeta = logalphabeta[2];
  
  for(k in 1:K){
    p[k] = inv_logit(logalpha + exp(logbeta) * log(doses[k]/dR));
  }
}

model {
  target += log_mix(similarity_parameter_alpha, normal_lpdf(logalpha | mu_1[1], Sigma_1[1,1]), normal_lpdf(logalpha | mu_2[1], Sigma_2[1,1])); 
  target += log_mix(similarity_parameter_beta, normal_lpdf(logbeta | mu_1[2], Sigma_1[2,2]), normal_lpdf(logbeta | mu_2[2], Sigma_2[2,2])); 
  for (k in 1:K) {
    target += binomial_logit_lpmf(tox[k] | N[k], logalpha + exp(logbeta)*log(doses[k]/dR));
  }
}
