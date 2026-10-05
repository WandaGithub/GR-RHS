
data {
  int<lower=1> N;
  vector[N] y;
  int<lower=1> p;
  int<lower=0> G;
  array[p - 1] int group_id;
  matrix[N, p] X;

  int<lower=1> Ntest;
  vector[Ntest] ytest;
  matrix[Ntest, p] Xtest;

  vector<lower=0>[G] gamma2_shape;
  real<lower=0> gamma2_rate;
  vector<lower=0>[p - 1] lambda2_shape;
  real<lower=0> lambda2_scale;

  real tau_location;
  real<lower=0, upper=p - 1> p0;
  int<lower=0, upper=1> use_tau_calibration;

  real intercept_mean;
  real<lower=0> intercept_sd;

  real<lower=0> sigma_df;
  real sigma_location;
  real<lower=0> sigma_scale;

  real z_mean;
  real<lower=0> z_sd;
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
  for (i in 1 : N) {
    yc[i] = y[i] - ymean;
  }

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

  real<lower=0> tau;
  vector<lower=0>[G] gamma2;
  vector<lower=0>[pc] lambda2;

  vector[pc] z_beta;
}

transformed parameters {
  vector<lower=0>[pc] Sigma;
  vector[pc] beta_scaled;
  vector[pc] beta;

  for (j in 1 : pc) {
    Sigma[j] = square(tau) * gamma2[group_id[j]] * lambda2[j];
  }

  beta_scaled = z_beta .* sqrt(Sigma);

  beta = beta_scaled .* inv_sds_X;
}

model {
  intercept ~ normal(intercept_mean, intercept_sd);
  z_beta ~ normal(z_mean, z_sd);

  tau ~ cauchy(
    tau_location,
    sigma * calibration_multiplier
  );
  sigma ~ student_t(sigma_df, sigma_location, sigma_scale);

  gamma2 ~ gamma(gamma2_shape, gamma2_rate);
  lambda2 ~ inv_gamma(lambda2_shape, lambda2_scale);

  yc ~ normal_id_glm(Xc, intercept, beta, sigma);
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
