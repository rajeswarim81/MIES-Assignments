1. The script imports only numpy to read and write the input dataset and the output results into numpy arrays.

2. While training:
     Input dataset is stored in parameter: X
     Output target is stored in parameter: y

     The test data is stored in: X_test

     The computed probabilities at the output layer are stored in: output (while training)
     The computed probabilities at the output layer are stored in: output_test (while testing)

     The predicted results at the output layer are stored in: output_pred (while training)
     The predicted results at the output layer are stored in: output_pred_test (while testing)


 3. Functions (user defined) were used to compute sigmoid, relu, and their derivatives.

