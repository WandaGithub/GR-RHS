
data {
  int<lower=1> N;
  int<lower=1> p;
  matrix[N, p] X;
  vector[N] y;

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

  real tau_location;
  real lambda_location;
  real<lower=0> lambda_scale;
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
  real ymean;
  real<lower=0> calibration_multiplier;

  for (i in 2 : p) {
    means_X[i - 1] = mean(X[ : , i]);
    sds_X[i - 1] = sd(X[ : , i]);
    inv_sds_X[i - 1] = 1.0 / sds_X[i - 1];
    Xc[ : , i - 1] = X[ : , i] - means_X[i - 1];
  }

  ymean = mean(y);
  if (use_tau_calibration == 1 && (p0 <= 0 || p0 >= pc))
    reject("p0 must satisfy 0 < p0 < p - 1.");
  if (use_tau_calibration == 1)
    calibration_multiplier = p0 / (pc - p0) / sqrt(N);
  else
    calibration_multiplier = 1.0;
  for (i in 1 : N) {
    yc[i] = y[i] - ymean;
  }
}

parameters {
  real intercept;
  vector[pc] z_beta;

  vector<lower=0>[pc] lambda;
  real<lower=0> tau;

  real<lower=0> sigma;
}

transformed parameters {
  vector[pc] beta;

  beta = tau * lambda .* z_beta .* inv_sds_X;
}

model {
  target += normal_id_glm_lpdf(yc | Xc, intercept, beta, sigma);

  target += normal_lpdf(z_beta | z_mean, z_sd);

  target += cauchy_lpdf(lambda | lambda_location, lambda_scale);

  target += cauchy_lpdf(
    tau | tau_location,
    sigma * calibration_multiplier
  );

  target += student_t_lpdf(sigma | sigma_df, sigma_location, sigma_scale);

  target += normal_lpdf(intercept | intercept_mean, intercept_sd);
}

generated quantities {
  real b_intercept = ymean + intercept - dot_product(means_X, beta);

  vector[N] log_lik;
  array[N] real y_tilde;
  vector[N] mu_tilde =
    rep_vector(0.0, N) + b_intercept + X[ : , 2 : p] * beta;

  vector[Ntest] log_lik_test;
  array[Ntest] real y_tilde_test;
  vector[Ntest] mu_tilde_test =
    rep_vector(0.0, Ntest) + b_intercept + Xtest[ : , 2 : p] * beta;

  real<lower=0, upper=1> pred_R2 =
    variance(mu_tilde) / (variance(mu_tilde) + sigma ^ 2);

  real<lower=0, upper=1> bayes_R2_test_design =
    variance(mu_tilde_test) / (variance(mu_tilde_test) + sigma ^ 2);

  for (n in 1 : N) {
    log_lik[n] = normal_lpdf(y[n] | mu_tilde[n], sigma);
    y_tilde[n] = normal_rng(mu_tilde[n], sigma);
  }

  for (n in 1 : Ntest) {
    log_lik_test[n] = normal_lpdf(ytest[n] | mu_tilde_test[n], sigma);
    y_tilde_test[n] = normal_rng(mu_tilde_test[n], sigma);
  }
}
