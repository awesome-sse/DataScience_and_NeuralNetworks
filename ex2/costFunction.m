function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
sum1 = 0;
sum2 = 0;
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
for i=1:m, 
  sum1 = sum1 + ((-1*y(i) * log(sigmoid(X(i,:) * theta))) - (1 - y(i))* log(1 - sigmoid(X(i,:) * theta)));
J = 1/m * sum1;
endfor
for j=1:length(theta), 
  for i=1:m,
    sum2 = sum2+(sigmoid(X(i,:) * theta)- y(i))*X(i,j);
  endfor
  grad(j) = sum2/m;
  sum2 = 0;
endfor




% =============================================================

end