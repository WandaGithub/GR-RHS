

data {
  int<lower=2> N;
  vector[N] y;
  int<lower=2> p;
  int<lower=1> G;
  matrix[N, p] X;
  array[p - 1] int<lower=1, upper=G> group_id;

  int<lower=1> Ntest;
  vector[Ntest] ytest;
  matrix[Ntest, p] Xtest;

  real intercept_mean;
  real<lower=0> intercept_sd;
  real<lower=0> sigma_df;
  real sigma_location;
  real<lower=0> sigma_scale;
  real z_mean;
  real<lower=0> z_sd;
  real<lower=0, upper=p - 1> p0;
  int<lower=0, upper=1> use_tau_calibration;
}

transformed data {
  int pc = p - 1;
  matrix[N, pc] Xc;
  vector[pc] means_X;
  vector[pc] sds_X;
  vector[pc] inv_sds_X;
  vector[N] yc;
  real ymean = mean(y);
  real<lower=0> calibration_multiplier;

  for (j in 1 : pc) {
    means_X[j] = mean(X[ : , j + 1]);
    Xc[ : , j] = X[ : , j + 1] - means_X[j];
    sds_X[j] = sd(X[ : , j + 1]);
    if (sds_X[j] <= 0)
      reject("Every GHS predictor must have positive training SD.");
    inv_sds_X[j] = 1.0 / sds_X[j];
  }

  for (i in 1 : N)
    yc[i] = y[i] - ymean;

  if (use_tau_calibration == 1 && (p0 <= 0 || p0 >= pc))
    reject("p0 must satisfy 0 < p0 < p - 1.");
  if (use_tau_calibration == 1)
    calibration_multiplier = p0 / (pc - p0) / sqrt(N);
  else
    calibration_multiplier = 1.0;
}

parameters {
  real intercept;
  real<lower=0> sigma;
  vector[p - 1] z_beta;

  real<lower=0> tau;

  array[2] vector<lower=0>[p - 1] lambda_aux;
  array[2] vector<lower=0>[G] delta_aux;
}

transformed parameters {
  vector[pc] log_lambda =
    log(lambda_aux[1]) + 0.5 * log(lambda_aux[2]);
  vector[pc] lambda = exp(log_lambda);
  vector[G] log_delta =
    log(delta_aux[1]) + 0.5 * log(delta_aux[2]);
  vector[G] delta = exp(log_delta);
  vector[pc] beta_scaled;
  vector[pc] beta;

  for (j in 1 : pc) {
    beta_scaled[j] =
      z_beta[j]
      * tau
      * exp(log_delta[group_id[j]] + log_lambda[j]);
  }

  beta = beta_scaled .* inv_sds_X;
}

model {
  z_beta ~ normal(z_mean, z_sd);

  tau ~ cauchy(0.0, sigma * calibration_multiplier);

  lambda_aux[1] ~ std_normal();
  lambda_aux[2] ~ inv_gamma(0.5, 0.5);
  delta_aux[1] ~ std_normal();
  delta_aux[2] ~ inv_gamma(0.5, 0.5);

  intercept ~ normal(intercept_mean, intercept_sd);
  sigma ~ student_t(sigma_df, sigma_location, sigma_scale);

  target += normal_id_glm_lpdf(yc | Xc, intercept, beta, sigma);
}

generated quantities {
  real b_intercept =
    ymean + intercept - dot_product(means_X, beta);

  vector[N] log_lik;
  array[N] real y_tilde;
  vector[N] mu_tilde =
    rep_vector(0.0, N) + b_intercept + X[ : , 2 : p] * beta;

  vector[Ntest] log_lik_test;
  array[Ntest] real y_tilde_test;
  vector[Ntest] mu_tilde_test =
    rep_vector(0.0, Ntest)
    + b_intercept
    + Xtest[ : , 2 : p] * beta;

  real<lower=0, upper=1> pred_R2 =
    variance(mu_tilde) / (variance(mu_tilde) + square(sigma));

  real<lower=0, upper=1> bayes_R2_test_design =
    variance(mu_tilde_test)
    / (variance(mu_tilde_test) + square(sigma));

  for (n in 1 : N) {
    log_lik[n] = normal_lpdf(y[n] | mu_tilde[n], sigma);
    y_tilde[n] = normal_rng(mu_tilde[n], sigma);
  }

  for (n in 1 : Ntest) {
    log_lik_test[n] = normal_lpdf(ytest[n] | mu_tilde_test[n], sigma);
    y_tilde_test[n] = normal_rng(mu_tilde_test[n], sigma);
  }
}
