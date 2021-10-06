function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
sum1 = 0;
sum2 = 0;
sum3 = 0;
for i=1:m,
  sum1 += (-y(i) * log(sigmoid(X(i,:) * theta)) - (1 - y(i)) * log(1 - sigmoid(X(i,:) * theta)));
endfor
for j=2:length(theta),
  sum2 += (theta(j))^2;
endfor
J = sum1 / m + lambda/(2*m)*sum2;


for j=1:size(theta),
  sum3 = 0;
  for i=1:m,
    sum3 += (sigmoid(X(i,:) * theta) - y(i))*X(i,j) ;
  endfor
  grad(j,:) = sum3/m
  if j!=1 
    grad(j,:) += lambda/m * theta(j); 
  endif
endfor








% =============================================================

grad = grad(:);

end
