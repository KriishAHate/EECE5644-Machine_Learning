%% Q1 PART A SUBQUESTION 1

% Expected risk minimization with 2 classes
clear all, close all,

n = 3; % number of feature dimensions**to 3? or 2?**
N = 10000; % number of iid samples **replaced with the given sample size**
mu(:,1) = [-1/2;-1/2;-1/2]; mu(:,2) = [1;1;1];% **kept it as a 3d column vector**
Sigma(:,:,1) = [1,-0.5,0.3;-0.5,1,-0.5;0.3,-0.5,1]; Sigma(:,:,2) = [1,0.3,-0.2;0.3,1,0.3;-0.2,0.3,1];% added the sigma 3x3 matrix**
p = [0.65,0.35]; % **updated class priors for labels 0 and 1 respectively**
label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % save up space
% Draw samples from each class pdf
for l = 0:1
    %x(:,label==l) = randGaussian(Nc(l+1),mu(:,l+1),Sigma(:,:,l+1));
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
figure(2), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
disp(gamma) %display value of gamma calculated emperically
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma)); % LDA threshold not optimized to minimize its own E[Risk]!

ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
if norm(lambda-[0,1;1,0])<eps % Using 0-1 loss indicates intent to minimize P(error)
    Perror_MAP = [p10,p01]*Nc'/N, % probability of error, empirically estimated
end
