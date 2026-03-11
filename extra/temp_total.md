L2	4.629236e-02	4.223528e-02	8.277825e-02	5.438185e-02
MSE	1.404118e-09	5.898676e-10	4.722520e-07	4.722222e-09
MAE	2.477028e-05	1.949222e-05	2.765447e-04	4.894649e-05
MAPE	6.825350e-01	1.273190e+00	6.684201e-01	1.035480e+00




WO GAN ODE MODEL

DETAILS:
cfg.learning_rate = 0.01
cfg.seed = 42
device = cfg.device
cfg.num_layers_lstm = 2
cfg.hidden_dim = 7
cfg.lstm_hidden_dim = cfg.hidden_dim
cfg.ode_hidden_dim = 15
cfg.use_learnable_exp = False
cfg.use_non_learnable_exp = True
cfg.exp_deno_init = 6
cfg.lambda_gan = 1.0
cfg.lambda_l2 = 0.1
cfg.only_d_epoch = 10

(I can't figure out the exact reason why the val loss is a little bit different from tuning result, but anyway this is so powerful that outweighs other models.
  
val loss = 0.37157)

	1	2	3	4
L2	3.575448e-02	3.158705e-02	7.314786e-02	4.624171e-02
MSE	1.517354e-09	5.748679e-10	4.406833e-07	3.572645e-09
MAE	2.238067e-05	1.337207e-05	2.810595e-04	3.741809e-05
MAPE	4.294788e-01	4.819492e-01	6.000935e-01	5.898243e-01


ODE 8 16 2 15

	3.831919e-02	3.257131e-02	6.996041e-02	4.769887e-02
MSE	1.727720e-09	5.954210e-10	4.287275e-07	4.377197e-09
MAE	2.399498e-05	1.350593e-05	2.673840e-04	3.876100e-05
MAPE	4.286046e-01	4.608989e-01	4.966611e-01	4.708793e-01


ODE 5 18 2 3

1	2	3	4
L2	3.528867e-02	3.121377e-02	6.682929e-02	4.739688e-02
MSE	1.509882e-09	6.361671e-10	4.381908e-07	4.674272e-09
MAE	2.159675e-05	1.354713e-05	2.645820e-04	3.979503e-05
MAPE	4.259684e-01	4.556609e-01	4.541856e-01	4.308276e-01
