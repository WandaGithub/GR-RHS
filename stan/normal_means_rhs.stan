data {
  int<lower=1> p;
  vector[p] y;
  real<lower=0> sigma;
  int<lower=0, upper=1> learn_tau;
  real<lower=0> tau_reference;
  real<lower=0> slab_variance_scale;
  real<lower=0> alpha_b;
  real<lower=0> beta_b;
}
parameters {
  vector[p] log_lambda;
  vector[1] log_c2;
  vector[learn_tau] log_tau_free;
}
transformed parameters {
  real log_tau = log(tau_reference);
  vector[p] log_v;
  if (learn_tau == 1) log_tau = log_tau_free[1];
  for (j in 1:p) {
    real log_q = 2 * (log_tau + log_lambda[j]);
    log_v[j] = -log_sum_exp(-log_c2[1], -log_q);
  }
}
model {
  target += sum(log_lambda - log1p_exp(2 * log_lambda));
  if (learn_tau == 1) {
    real z = log_tau_free[1] - log(tau_reference);
    target += z - log1p_exp(2 * z);
  }
  {
    real z = log_c2[1] - log(slab_variance_scale);
    target += alpha_b * z - (alpha_b + beta_b) * log1p_exp(z)
              - lbeta(alpha_b, beta_b);
  }
  for (j in 1:p)
    y[j] ~ normal(0, exp(0.5 * log_sum_exp(2 * log(sigma), log_v[j])));
}
generated quantities {
  real tau = exp(log_tau);
  vector[1] slab_floor;
  vector[p] kappa;
  vector[p] attenuation_ratio;
  vector[p] beta_conditional_mean;
  slab_floor[1] = inv_logit(2 * log(sigma) - log_c2[1]);
  for (j in 1:p) {
    kappa[j] = inv_logit(2 * log(sigma) - log_v[j]);
    attenuation_ratio[j] = inv_logit(log_c2[1]
                                     - 2 * (log_tau + log_lambda[j]));
    beta_conditional_mean[j] = (1 - kappa[j]) * y[j];
  }
}
