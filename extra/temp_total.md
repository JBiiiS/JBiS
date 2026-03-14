	1	2	3	4
L2	4.629236e-02	4.223528e-02	8.277825e-02	5.438185e-02
MSE	1.404118e-09	5.898676e-10	4.722520e-07	4.722222e-09
MAE	2.477028e-05	1.949222e-05	2.765447e-04	4.894649e-05
MAPE	6.825350e-01	1.273190e+00	6.684201e-01	1.035480e+00
QLIKE	2.427966e-01	3.443787e-01	1.560321e+00	3.197661e-01



	1	2	3	4
L2	3.534094e-02	3.095123e-02	6.612925e-02	4.652887e-02
MSE	1.447883e-09	6.038007e-10	4.185790e-07	4.461364e-09
MAE	2.153048e-05	1.361304e-05	2.613417e-04	3.911875e-05
MAPE	4.410221e-01	4.966606e-01	4.772056e-01	4.449438e-01
QLIKE	3.526520e-01	3.310809e-01	1.813399e+00	6.361198e-01






WO GAN ODE MODEL

cfg = config_ode
M = cfg.total_quantile
alpha = torch.linspace(1 / M, 1, M).to(device)
cfg.learning_rate = 0.01
cfg.seed = 42
device = cfg.device
cfg.num_layers_lstm = 2
cfg.hidden_dim = 5
cfg.lstm_hidden_dim = cfg.hidden_dim
cfg.ode_hidden_dim = 18
cfg.use_learnable_exp = False
cfg.use_non_learnable_exp = True
cfg.exp_deno_init = 3
cfg.lambda_gan = 1.0
cfg.lambda_l2 = 0.1
cfg.only_d_epoch = 20

	1	2	3	4
L2	3.528867e-02	3.121377e-02	6.682929e-02	4.739688e-02
MSE	1.509882e-09	6.361671e-10	4.381908e-07	4.674272e-09
MAE	2.159675e-05	1.354713e-05	2.645820e-04	3.979503e-05
MAPE	4.259684e-01	4.556609e-01	4.541856e-01	4.308276e-01
QLIKE	3.890442e-01	3.660746e-01	1.944446e+00	7.253867e-01


SAME CONFIG WITH THE UPPER ONE / ODE DIS

	1	2	3	4
L2	3.534094e-02	3.095123e-02	6.612925e-02	4.652887e-02
MSE	1.447883e-09	6.038007e-10	4.185790e-07	4.461364e-09
MAE	2.153048e-05	1.361304e-05	2.613417e-04	3.911875e-05
MAPE	4.410221e-01	4.966606e-01	4.772056e-01	4.449438e-01
QLIKE	3.526520e-01	3.310809e-01	1.813399e+00	6.361198e-01



just ode: no revision in   other configs. only qlike route is opened with coef = 0.1 JUST ODE

1	2	3	4
L2	3.541727e-02	3.236426e-02	6.786656e-02	4.452164e-02
MSE	1.512621e-09	5.341784e-10	4.237258e-07	4.072358e-09
MAE	2.219879e-05	1.369775e-05	2.605177e-04	3.729043e-05
MAPE	4.402269e-01	5.401494e-01	4.649183e-01	4.587716e-01
QLIKE	3.459181e-01	3.105827e-01	1.924560e+00	4.800514e-01


0.05
1	2	3	4
L2	3.562202e-02	3.377542e-02	6.678484e-02	4.466359e-02
MSE	1.514927e-09	5.235590e-10	4.135274e-07	4.110912e-09
MAE	2.218979e-05	1.398889e-05	2.577445e-04	3.708275e-05
MAPE	4.623283e-01	5.776272e-01	4.654163e-01	4.516149e-01
QLIKE	2.993730e-01	3.006655e-01	1.811464e+00	4.798349e-01
















cfg = config_ode

cfg.learning_rate = 0.01
cfg.seed = 42
device = cfg.device
cfg.num_layers_lstm = 2
cfg.hidden_dim = 16
cfg.lstm_hidden_dim = cfg.hidden_dim
cfg.ode_hidden_dim = 20
cfg.use_learnable_exp = False
cfg.use_non_learnable_exp = True
cfg.exp_deno_init = 12
cfg.lambda_gan = 1.0
cfg.lambda_l2 = 0.1
cfg.only_d_epoch = 20
lambda_qlike = 0.05

	1	2	3	4
L2	3.466115e-02	3.270749e-02	7.679272e-02	4.499338e-02
MSE	1.325958e-09	5.068722e-10	5.189577e-07	3.892701e-09
MAE	2.124573e-05	1.457679e-05	2.876996e-04	3.801145e-05
MAPE	4.991511e-01	6.675195e-01	5.235336e-01	5.823862e-01
QLIKE	2.525482e-01	2.609367e-01	1.959860e+00	3.979515e-01

Epoch [ 56/1000] | Early Stop Std: 0.063970 | Best Std (Post-ep 30): 0.06397033381191167
   ↳ Val L2: 0.0396 | Val W-Dist: 0.0002 | Val QLIKE: 0.2812

cfg.learning_rate = 0.01
cfg.seed = 42
device = cfg.device
cfg.num_layers_lstm = 2
cfg.hidden_dim = 16
cfg.lstm_hidden_dim = cfg.hidden_dim
cfg.ode_hidden_dim = 20
cfg.use_learnable_exp = False
cfg.use_non_learnable_exp = True
cfg.exp_deno_init = 12
cfg.lambda_gan = 1.0
cfg.lambda_l2 = 0.1
cfg.only_d_epoch = 20
lambda_qlike = 0.05


ODE DIS
L2	3.512990e-02	3.409578e-02	7.599452e-02	4.518152e-02
MSE	1.313908e-09	4.893356e-10	5.100029e-07	3.766769e-09
MAE	2.225603e-05	1.533882e-05	2.846994e-04	3.875630e-05
MAPE	5.767127e-01	7.790751e-01	5.629051e-01	6.588684e-01
QLIKE	2.312081e-01	2.570156e-01	1.762340e+00	3.559254e-01











cfg.learning_rate = 0.01
cfg.seed = 42
device = cfg.device
cfg.num_layers_lstm = 2
cfg.hidden_dim = 8
cfg.lstm_hidden_dim = cfg.hidden_dim
cfg.ode_hidden_dim = 9
cfg.use_learnable_exp = False
cfg.use_non_learnable_exp = True
cfg.exp_deno_init = 17
cfg.only_d_epoch = 20
lambda_qlike = 0.1

1	2	3	4
L2	3.578625e-02	3.673267e-02	7.773114e-02	4.521824e-02
MSE	1.436614e-09	4.373448e-10	4.050372e-07	3.871258e-09
MAE	2.312488e-05	1.470337e-05	2.884840e-04	3.803876e-05
MAPE	5.188560e-01	7.645762e-01	7.226534e-01	5.472745e-01
QLIKE	2.466284e-01	2.553950e-01	1.665073e+00	3.603876e-01


==================================================
ode dis tuning
=================================================

(only ode: val loss 0.65826, 8/9/17)
# Training Objective Weights (Gradient Calculation)
cfg.lambda_gan = 1.0
cfg.lambda_l2 = 0.1
cfg.lambda_qlike = 0.0

# Validation Early Stopping Coefficients (Metric Calculation)
cfg.l2_coef = 1.0
cfg.qlike_coef = 0.1  


Epoch [ 64/1000] | Early Stop Std: 0.063751 | Best Std (Post-ep 30): 0.06375119564208118
   ↳ Val L2: 0.0366 | Val QLIKE: 0.2712
   ↳ Train L2: 0.0385 | Train QLIKE: 0.2286 | Fake: -0.0197 | Real: -0.0305       

==================================================

(위에거에서 lambda l2 0.1 -> 0.3)

Epoch [ 68/500] | Early Stop Std: 0.063570 | Best Std (Post-ep 30): 0.06356966461647641
   ↳ Val L2: 0.0369 | Val QLIKE: 0.2670
   ↳ Train L2: 0.0389 | Train Q LIKE: 0.2186 | Fake: -0.0166 | Real: -0.0184


   1	2	3	4
L2	3.697564e-02	3.623650e-02	7.895191e-02	4.510097e-02
MSE	1.649253e-09	4.358689e-10	3.911508e-07	4.026129e-09
MAE	2.498544e-05	1.573264e-05	3.067708e-04	3.979634e-05
MAPE	6.137331e-01	9.196838e-01	8.811908e-01	6.507763e-01
QLIKE	2.251987e-01	2.663197e-01	1.485669e+00	3.224025e-01



(org)
1	2	3	4
L2	4.629236e-02	4.223528e-02	8.277825e-02	5.438185e-02
MSE	1.404118e-09	5.898676e-10	4.722520e-07	4.722222e-09
MAE	2.477028e-05	1.949222e-05	2.765447e-04	4.894649e-05
MAPE	6.825350e-01	1.273190e+00	6.684201e-01	1.035480e+00
QLIKE	2.427966e-01	3.443787e-01	1.560321e+00	3.197661e-01
































































































































(org)
1	2	3	4
L2	4.629236e-02	4.223528e-02	8.277825e-02	5.438185e-02
MSE	1.404118e-09	5.898676e-10	4.722520e-07	4.722222e-09
MAE	2.477028e-05	1.949222e-05	2.765447e-04	4.894649e-05
MAPE	6.825350e-01	1.273190e+00	6.684201e-01	1.035480e+00
QLIKE	2.427966e-01	3.443787e-01	1.560321e+00	3.197661e-01
