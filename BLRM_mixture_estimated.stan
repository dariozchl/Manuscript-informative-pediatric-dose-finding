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
  real<lower=0> delta_a;
  real<lower=0> delta_b;
}

parameters {
  vector[2] logalphabeta;
  real<lower=0,upper=1> borrowing_parameter;
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
  target += log_sum_exp(log(borrowing_parameter)   + multi_normal_lpdf(logalphabeta | mu_1, Sigma_1),
                        log(1-borrowing_parameter) + multi_normal_lpdf(logalphabeta | mu_2, Sigma_2)); 
  target += beta_lpdf(borrowing_parameter | delta_a, delta_b);
  for (k in 1:K) {
    target += binomial_logit_lpmf(tox[k] | N[k], logalpha + exp(logbeta)*log(doses[k]/dR));
  }
}
