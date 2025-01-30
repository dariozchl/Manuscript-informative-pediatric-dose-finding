data {
  int<lower=1> K; // total number of dose levels 
  vector<lower=0>[K] doses; // predictor variable 
  int<lower=0> tox[K]; // number of tox per dose
  int<lower=0> N[K]; // total number of patients per dose level
  int<lower=0> dR; // reference dose
  vector[2] mu;
  vector<lower=0>[2] sigma; 
  real rho;
  real<lower=0,upper=1> similarity_parameter_alpha;
  real<lower=0,upper=1> similarity_parameter_beta;
}

transformed data{
  cov_matrix[2] Sigma;
  Sigma[1,1] = square(sigma[1])*100*0.01^similarity_parameter_alpha;
  Sigma[2,2] = square(sigma[2])*100*0.01^similarity_parameter_beta;
  Sigma[1,2] = sqrt(Sigma[1,1])*sqrt(Sigma[2,2])*rho;
  Sigma[2,1] = sqrt(Sigma[1,1])*sqrt(Sigma[2,2])*rho;
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
  logalphabeta ~ multi_normal(mu, Sigma);
  tox ~ binomial_logit(N, logalpha + exp(logbeta)*log(doses/dR));
}
