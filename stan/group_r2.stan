
functions {
  vector R2D(vector z, vector inv_sds_X, vector phi, real tau2) {
    return z .* sqrt(phi * tau2) .* inv_sds_X;
  }
}

data {
  int<lower=1> N;
  vector[N] y;
  int<lower=1> p;
  matrix[N, p] X;

  int<lower=2> G;
  array[G] int pg;

  real<lower=0> phi_group_concentration;
  real<lower=0> phi_within_concentration;
  real<lower=0> phi_within_rate;

  int<lower=1> Ntest;
  vector[Ntest] ytest;
  matrix[Ntest, p] Xtest;

  real<lower=0> R2_a;
  real<lower=0> R2_b;

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

  for (i in 2 : p) {
    means_X[i - 1] = mean(X[ : , i]);
    sds_X[i - 1] = sd(X[ : , i]);
    Xc[ : , i - 1] = X[ : , i] - means_X[i - 1];
  }

  inv_sds_X = 1.0 ./ sds_X;

  ymean = mean(y);
  for (i in 1 : N) {
    yc[i] = y[i] - ymean;
  }
}

parameters {
  vector[pc] z_beta;

  simplex[G] phi_group;

  real<lower=0, upper=1> R2;

  vector<lower=0>[pc] phi_within_aux;

  real intercept;
  real<lower=0> sigma;
}

transformed parameters {
  vector[pc] beta;
  vector[pc] lambdas;
  real<lower=0> tau2;

  tau2 = sigma ^ 2 * R2 / (1 - R2);

  {
    int pos = 1;

    for (i in 1 : G) {
      beta[pos : pos + pg[i] - 1] =
        R2D(
          segment(z_beta, pos, pg[i]),
          segment(inv_sds_X, pos, pg[i]),
          segment(phi_within_aux, pos, pg[i])
            / sum(segment(phi_within_aux, pos, pg[i])),
          phi_group[i] * tau2
        );

      lambdas[pos : pos + pg[i] - 1] =
        sqrt(
          segment(phi_within_aux, pos, pg[i])
            / sum(segment(phi_within_aux, pos, pg[i]))
            * phi_group[i]
            * tau2
        );

      pos = pos + pg[i];
    }
  }
}

model {
  target += normal_id_glm_lpdf(yc | Xc, intercept, beta, sigma);

  target += beta_lpdf(
    R2 | R2_a * R2_b, (1 - R2_a) * R2_b
  );

  target += dirichlet_lpdf(
    phi_group | rep_vector(phi_group_concentration, G)
  );

  target += gamma_lpdf(
    phi_within_aux | phi_within_concentration, phi_within_rate
  );

  target += normal_lpdf(z_beta | z_mean, z_sd);

  target += student_t_lpdf(
    sigma | sigma_df, sigma_location, sigma_scale
  );
  target += normal_lpdf(
    intercept | intercept_mean, intercept_sd
  );
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

  vector[G] lambda2_groups = phi_group * tau2;

  real<lower=0, upper=1> pred_R2;
  real<lower=0, upper=1> bayes_R2_test_design;

  pred_R2 =
    variance(mu_tilde) / (variance(mu_tilde) + sigma ^ 2);

  bayes_R2_test_design =
    variance(mu_tilde_test)
    / (variance(mu_tilde_test) + sigma ^ 2);

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
