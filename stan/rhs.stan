

data {
  int<lower=2> N;
  vector[N] y;
  int<lower=2> p;
  int<lower=1> G;
  matrix[N, p] X;
  array[p - 1] int<lower=1, upper=G> group_id;

  int<lower=2> Ntest;
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
  real<lower=0> tau_scale_multiplier;
  int<lower=0, upper=1> use_tau_fixed;
  real<lower=0> tau_fixed;
  real<lower=0> alpha_b;
  real<lower=0> beta_b;
  real<lower=0> slab_variance_scale;

  real lambda_normal_mean;
  real<lower=0> lambda_normal_sd;
  real<lower=0> lambda_ig_shape;
  real<lower=0> lambda_ig_scale;
}

transformed data {
  int pc = p - 1;
  real<lower=0> calibration_multiplier;

  matrix[N, pc] Xc;
  vector[pc] means_X;
  vector[pc] sds_X;
  vector[pc] inv_sds_X;
  vector[N] yc;
  real ymean;

  if (use_tau_calibration == 1 && (p0 <= 0 || p0 >= pc))
    reject("p0 must satisfy 0 < p0 < p - 1.");

  if (use_tau_calibration == 1) {
    calibration_multiplier =
      p0 / (pc - p0) / sqrt(N);
  } else {
    calibration_multiplier = 1.0;
  }

  for (i in 2 : p) {
    means_X[i - 1] = mean(X[ : , i]);
    sds_X[i - 1] = sd(X[ : , i]);
    if (sds_X[i - 1] <= 0)
      reject("Predictors must have positive standard deviation.");
    inv_sds_X[i - 1] = 1.0 / sds_X[i - 1];
    Xc[ : , i - 1] = X[ : , i] - means_X[i - 1];
  }

  ymean = mean(y);
  for (i in 1 : N)
    yc[i] = y[i] - ymean;
}

parameters {
  real intercept;
  real<lower=0> sigma;

  vector[pc] z_beta;

  real<lower=0> tau_parameter;
  array[2] vector<lower=0>[pc] lambda_aux;

  vector[G] log_c2_g;
}

transformed parameters {
  real<lower=0> tau;
  vector[pc] lambda;
  vector[pc] beta_scaled;
  vector[pc] beta;

  tau = use_tau_fixed == 1 ? tau_fixed : tau_parameter;

  lambda =
    lambda_aux[1] .* sqrt(lambda_aux[2]);

  for (j in 1 : pc) {
    real log_q = 2.0 * (log(tau) + log(lambda[j]));

    real log_w =
      log_c2_g[group_id[j]]
      + log_q
      - log_sum_exp(log_c2_g[group_id[j]], log_q);

    beta_scaled[j] =
      z_beta[j]
      * exp(0.5 * log_w);
  }

  beta = beta_scaled .* inv_sds_X;
}

model {
  z_beta ~ normal(z_mean, z_sd);
  intercept ~ normal(intercept_mean, intercept_sd);
  sigma ~ student_t(sigma_df, sigma_location, sigma_scale);

  if (use_tau_fixed == 0)
    tau_parameter ~ cauchy(0.0, sigma * calibration_multiplier * tau_scale_multiplier);

  lambda_aux[1] ~ normal(lambda_normal_mean, lambda_normal_sd);
  lambda_aux[2] ~ inv_gamma(lambda_ig_shape, lambda_ig_scale);

  for (g in 1 : G) {
    target += alpha_b * (log_c2_g[g] - log(slab_variance_scale))
              - (alpha_b + beta_b)
                * log1p_exp(log_c2_g[g] - log(slab_variance_scale))
              - lbeta(alpha_b, beta_b);
  }

  target += normal_id_glm_lpdf(
    yc | Xc, intercept, beta, sigma
  );
}

generated quantities {
  real b_intercept =
    ymean
    + intercept
    - dot_product(means_X, beta);

  vector[G] c2_g_out;

  vector[N] log_lik;
  array[N] real y_tilde;
  vector[N] mu_tilde =
    rep_vector(0.0, N)
    + b_intercept
    + X[ : , 2 : p] * beta;

  vector[Ntest] log_lik_test;
  array[Ntest] real y_tilde_test;
  vector[Ntest] mu_tilde_test =
    rep_vector(0.0, Ntest)
    + b_intercept
    + Xtest[ : , 2 : p] * beta;

  real<lower=0, upper=1> pred_R2 =
    variance(mu_tilde)
    / (variance(mu_tilde) + square(sigma));

  real<lower=0, upper=1> bayes_R2_test_design =
    variance(mu_tilde_test)
    / (variance(mu_tilde_test) + square(sigma));

  for (g in 1 : G)
    c2_g_out[g] = exp(log_c2_g[g]);

  for (n in 1 : N) {
    log_lik[n] =
      normal_lpdf(y[n] | mu_tilde[n], sigma);

    y_tilde[n] =
      normal_rng(mu_tilde[n], sigma);
  }

  for (n in 1 : Ntest) {
    log_lik_test[n] =
      normal_lpdf(ytest[n] | mu_tilde_test[n], sigma);

    y_tilde_test[n] =
      normal_rng(mu_tilde_test[n], sigma);
  }
}
