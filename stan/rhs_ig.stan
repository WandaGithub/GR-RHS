

functions {
  vector horseshoe(
    vector z,
    vector lambda,
    real tau,
    real c2,
    vector inv_sds_X
  ) {
    int K = rows(z);
    vector[K] lambda2 = square(lambda);

    vector[K] lambda_tilde =
      sqrt(c2 * lambda2 ./ (c2 + tau ^ 2 * lambda2));

    return z .* lambda_tilde * tau .* inv_sds_X;
  }
}

data {
  int<lower=1> N;
  vector[N] y;
  int<lower=0> p;
  matrix[N, p] X;

  real<lower=0> lambda_df;
  real<lower=0> slab_df;
  real<lower=0> slab_scale;

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

  real lambda_location;
  real<lower=0> lambda_scale;
  real tau_location;

  real<lower=0, upper=p - 1> p0;
  int<lower=0, upper=1> use_tau_calibration;
  real<lower=0> tau_scale_multiplier;
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

  if (use_tau_calibration == 1 && (p0 <= 0 || p0 >= pc))
    reject("p0 must satisfy 0 < p0 < p - 1.");

  if (use_tau_calibration == 1) {
    calibration_multiplier = p0 / (pc - p0) / sqrt(N);
  } else {
    calibration_multiplier = 1.0;
  }

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
}

parameters {
  real<lower=0> sigma;
  real intercept;

  vector[pc] z_beta;

  vector<lower=0>[pc] lambda;
  real<lower=0> tau;
  real<lower=0> slab_aux;
}

transformed parameters {
  real<lower=0> c2 = square(slab_scale) * slab_aux;

  vector[pc] beta =
    horseshoe(z_beta, lambda, tau, c2, inv_sds_X);
}

model {
  target += normal_id_glm_lpdf(yc | Xc, intercept, beta, sigma);

  target += normal_lpdf(z_beta | z_mean, z_sd);
  target += normal_lpdf(intercept | intercept_mean, intercept_sd);

  tau ~ cauchy(
    tau_location,
    sigma * calibration_multiplier * tau_scale_multiplier
  );

  slab_aux ~ inv_gamma(0.5 * slab_df, 0.5 * slab_df);

  lambda ~ student_t(lambda_df, lambda_location, lambda_scale);

  sigma ~ student_t(sigma_df, sigma_location, sigma_scale);
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
    variance(mu_tilde_test)
    / (variance(mu_tilde_test) + sigma ^ 2);

  vector[pc] lambda_tilde =
    c2 * square(lambda)
    ./ (c2 + tau ^ 2 * square(lambda));

  for (n in 1 : N) {
    log_lik[n] = normal_lpdf(y[n] | mu_tilde[n], sigma);
    y_tilde[n] = normal_rng(mu_tilde[n], sigma);
  }

  for (n in 1 : Ntest) {
    log_lik_test[n] =
      normal_lpdf(ytest[n] | mu_tilde_test[n], sigma);

    y_tilde_test[n] =
      normal_rng(mu_tilde_test[n], sigma);
  }
}
