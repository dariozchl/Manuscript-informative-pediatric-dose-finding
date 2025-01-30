data {
  int<lower=1> K; // total number of dose levels 
  vector<lower=0>[K] doses; // predictor variable 
  int<lower=0> tox[K]; // number of tox per dose
  int<lower=0> N[K]; // total number of patients per dose level
  int<lower=0> dR; // reference dose
  vector[2] mu;
  vector<lower=0>[2] sigma; 
  real rho;
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
  cov_matrix[2] Sigma;
  
  Sigma[1,1] = square(sigma[1])/borrowing_parameter;
  Sigma[2,2] = square(sigma[2])/borrowing_parameter;
  Sigma[1,2] = sqrt(Sigma[1,1])*sqrt(Sigma[2,2])*rho;
  Sigma[2,1] = sqrt(Sigma[1,1])*sqrt(Sigma[2,2])*rho;
  
  logalpha = logalphabeta[1];
  logbeta = logalphabeta[2];  
  
  for(k in 1:K){
    p[k] = inv_logit(logalpha + exp(logbeta) * log(doses[k]/dR));
  }
}

model {
  logalphabeta ~ multi_normal(mu, Sigma);
  tox ~ binomial_logit(N, logalpha + exp(logbeta)*log(doses/dR));
  borrowing_parameter ~ beta(delta_a,delta_b);
}
