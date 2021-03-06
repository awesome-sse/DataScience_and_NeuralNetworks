function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
sum1 = 0;
y1 = zeros(m,num_labels);

for i = 1:m,
  y1(i,y(i)) = 1;
endfor

for i = 1:m,
  for k = 1:num_labels,
    sum1 += -1*y1(i,k)*log(h2(i,k)) - (1-y1(i,k))*log(1-h2(i,k));
  endfor
endfor

J = sum1/m;

theta1 = Theta1.^2;
theta2 = Theta2.^2;
J += ((sum(theta1(:)) + sum(theta2(:)) - sum(theta1(:,1)) - sum(theta2(:,1))) * lambda)/(2 * m);

% -------------------------------------------------------------

% 
#=========================================================================

% Unroll gradients
lit_delta_3 = h2.-y1;


lit_delta_2 = ( lit_delta_3 * Theta2) .* ([ones(m, 1) h1].*(1-[ones(m, 1) h1]));
lit_delta_2(:,1) = [];


Theta2_grad = ((lit_delta_3' * [ones(m, 1) h1])).* (1/m);
Theta1_grad = (lit_delta_2' * [ones(m, 1) X]) .* (1/m);
for j=2:3,
  Theta1_grad(:,j) += Theta1(:,j) .* (lambda / m );
endfor
for j=2:5,
  Theta2_grad(:,j) += Theta2(:,j) .* (lambda / m); 
endfor
grad = [Theta1_grad(:) ; Theta2_grad(:)]
