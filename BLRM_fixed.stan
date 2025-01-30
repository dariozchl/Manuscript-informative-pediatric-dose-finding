data {
        int<lower=1> K; // total number of dose levels 
        vector<lower=0>[K] doses;
        int<lower=0> tox[K]; // number of tox per dose
        int<lower=0> N[K]; // total number of patients per dose level
        int<lower=0> dR; // reference dose
        vector[2] mu;
        cov_matrix[2] Sigma;
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
