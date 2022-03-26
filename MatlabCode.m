% clean workspace and command window
clear;
clc;

%% 1. Test-statistics simulation
beta2=[0 0.5]; % beta2 value for each hypothesis (1) true hypothesis H0: beta2=0 (2)false hypothesis H0: beta2=0.5
R=1000; % number of simulation datasets for each hypothesis
alpha = 0.05; % significance level
dof=1; % number of restrictions
tol=10^(-6); % convergence tolerance
stat=zeros(R,3); % matrix used for storing test statistics values for each hypothesis
total_stat=zeros(R,1); % matrix initialed for storing total (two hypothesis) test statistics values

for i=1:length(beta2) 
    beta2_H0=beta2(i);% specify different hypothesis value
    for r=1:R
        % generate data
        N=1000;
        X = randn(N,2); % 1000 by 2 matrix
        beta = [1 0]'; % the true value
        lambda = zeros(N,1); 
        lambda=exp(X*beta);
        Y=poissrnd(lambda); 
        X1=X(:,1);
        X2=X(:,2);
        % Netwon-Raphson algorithm
        start=[2;1]; %initial guess
        Xb_0=zeros(N,1); [betahat_u,logL_u]=maxloglik(Y,X,start,Xb_0,tol); % unrestricted model
        Xb_0=beta2(i)*X2; [betahat_r1_1,logL_r1]=maxloglik(Y,X1,start(1),Xb_0,tol); % restricted model
        betahat_r1=[betahat_r1_1;beta2(i)]; % betahat of the unrestricted model
        % results of three tests
        [stat_wald,pValue_wald,h_wald,cValue_wald]=wald_test_p(betahat_u,N,X,beta2_H0,dof);
        [stat_lr,pValue_lr,h_lr,cValue_lr]=lr_test_p(logL_r1,logL_u,dof);
        [stat_score,pValue_score,h_score,cValue_score]=score_test_p(betahat_r1,N,X,Y,dof);
        % store test statistic value
        stat(r,1) = stat_wald;
        stat(r,2) = stat_lr;
        stat(r,3) = stat_score;
    end
    total_stat=[total_stat,stat];
end

total_stat=total_stat(:,2:7); % test statistics values for two hypytheses

% test statistic distribution (true hypothesis H0: beta2=0)
set(0,'defaultfigurecolor','w');
x_lr=0:0.01:8;
set(figure, 'Position', [10 200 1000 220]);
subplot(1,4,1), plot(x_lr,chi2pdf(x_lr,1)),   xlim([0 8]), title('Chi2(1) density')
subplot(1,4,2), histogram(total_stat(:,1),200), xlim([0 8]), title('histogram of Wald statistics')
subplot(1,4,3), histogram(total_stat(:,2),200), xlim([0 8]), title('histogram of LR statistics')
subplot(1,4,4), histogram(total_stat(:,3),200), xlim([0 8]), title('histogram of Score statistics')

% test statistic distribution (false hypothesis H0: beta2=0.5)
x_lr=200:10:600;
set(figure, 'Position', [10 200 700 220]);
subplot(1,3,1), histogram(total_stat(:,4),200), xlim([200 600]), title('histogram of Wald statistics')
subplot(1,3,2), histogram(total_stat(:,5),200), xlim([200 600]), title('histogram of LR statistics')
subplot(1,3,3), histogram(total_stat(:,6),200), xlim([200 600]), title('histogram of Score statistics')

%% 2. Three tests for a single dataset
% generate data
rng(123); % set seed
N=1000;
X = randn(N,2); % 1000 by 2 matrix
lambda = zeros(N,1); 
lambda=exp(X*beta);
Y=poissrnd(lambda); 
X1=X(:,1);
X2=X(:,2);

% H0: beta2=0 (true hypothesis) - Netwon-Raphson algorithm
Xb_0=zeros(N,1); [betahat_u,logL_u]=maxloglik(Y,X,start,Xb_0,tol); % unrestricted (alternative) model
Xb_0=0*X2; [betahat_r1_1,logL_r1]=maxloglik(Y,X1,start(1),Xb_0,tol);  % restricted model H0: beta2=0 (true hypothesis)
betahat_r1=[betahat_r1_1;0]; % batahat for the restricted model

% results of three tests
[stat_wald_t,pValue_wald_t,h_wald_t,cValue_wald_t]=wald_test_p(betahat_u,N,X,0,dof);
[stat_lr_t,pValue_lr_t,h_lr_t,cValue_lr_t]=lr_test_p(logL_r1,logL_u,dof);
[stat_score_t,pValue_score_t,h_score_t,cValue_score_t]=score_test_p(betahat_r1,N,X,Y,dof);
array2table([stat_wald_t,stat_lr_t,stat_score_t;pValue_wald_t,pValue_lr_t,pValue_score_t],'VariableNames', {'Wald', 'LR', 'Score'},'RowNames',{'test statistic', 'p-value'})

% H0: beta2=0.5 (false hypothesis) - Netwon-Raphson algorithm
Xb_0=zeros(N,1); [betahat_u,logL_u]=maxloglik(Y,X,start,Xb_0,tol); % unrestricted (alternative) model
Xb_0=0.5*X2; [betahat_r1_1,logL_r1]=maxloglik(Y,X1,start(1),Xb_0,tol);  % restricted model H0: beta2=0.5 (false hypothesis)
betahat_r1=[betahat_r1_1;0.5]; % batahat for the restricted model

% results of three tests
[stat_wald_f,pValue_wald_f,h_wald_f,cValue_wald_f]=wald_test_p(betahat_u,N,X,0.5,dof);
[stat_lr_f,pValue_lr_f,h_lr_f,cValue_lr_f]=lr_test_p(logL_r1,logL_u,dof);
[stat_score_f,pValue_score_f,h_score_f,cValue_score_f]=score_test_p(betahat_r1,N,X,Y,dof);
array2table([stat_wald_f,stat_lr_f,stat_score_f;pValue_wald_f,pValue_lr_f,pValue_score_f],'VariableNames', {'Wald', 'LR', 'Score'},'RowNames',{'test statistic', 'p-value'})

%% 3. Simulation with different sample sizes
stat_values=zeros(1,3); %matrix used for storing  statistics values for sample
p_values=zeros(1,3);%matrix used for storing p values for sample
for ss=5:5:200 
    % generate data
    rng(123);
    beta = [1 0]'; % the true value
    X = randn(ss,2); % 1000 by 2 matrix
    lambda = zeros(ss,1); 
    lambda=exp(X*beta);
    Y=poissrnd(lambda); % y: poisson distributed
    X1=X(:,1);
    X2=X(:,2);
    % H0: beta2=0 (true hypothesis) - Netwon-Raphson algorithm
    start=[2;1]; %initial guess
    Xb_0=zeros(ss,1); [betahat_u,logL_u]=maxloglik(Y,X ,start,Xb_0,tol); % unrestricted (alternative) model
    Xb_0=0*X2; [betahat_r1_1,logL_r1]=maxloglik(Y,X1,start(1),Xb_0,tol); % restricted model H0: beta2=0 (true hypothesis)
    betahat_r1=[betahat_r1_1;0]; % batahat for the restricted model
    % results of three tests
    [stat_lr,pValue_lr,h_lr,cValue_lr]=lr_test_p(logL_r1,logL_u,dof);
    [stat_wald,pValue_wald,h_wald,cValue_wald]=wald_test_p(betahat_u,ss,X,0,dof);
    [stat_score,pValue_score,h_score,cValue_score]=score_test_p(betahat_r1,ss,X,Y,dof);
    stat_values=[stat_values;stat_wald,stat_lr,stat_score];
    p_values=[p_values;pValue_wald,pValue_lr,pValue_score];
end

stat_values=stat_values(2:end,:);
p_values=p_values(2:end,:);
x_range=5:5:200;
figure();
plot(x_range,stat_values(:,1),'b',x_range,stat_values(:,2),'r',x_range,stat_values(:,3),'g');
title('Test statistics of three tests');
legend('Wald','LR','Score','Location','northwest');
figure();
plot(x_range,p_values(:,1),'b',x_range,p_values(:,2),'r',x_range,p_values(:,3),'g');
legend('Wald','LR','Score','Location','northwest');
title('Pvalues of three tests');

%% local functions
function [betahat,logL]=maxloglik(y,X,start,Xb_0,tol) % maximizes log-likelihood by Newton-Raphson
    betahat_old=start; % initial guess
    incr=Inf;
    while (incr'*incr)>tol
        [logL,s,H]=loglik(y,X,betahat_old,Xb_0);
        betahat=betahat_old-H\s;
        incr=betahat-betahat_old; % =-H\s
        betahat_old=betahat;
    end
end

function [logL,s,H]=loglik(y,X,beta,Xb_0)
    Xb=X*beta+Xb_0; lambda=exp(Xb);
    logL=sum(y.*Xb-lambda-log(factorial(y)));
    s=sum(X.*(y-lambda))'; % Score
    H=-X'*(lambda.*X); 
end

% Wald test
function [stat_wald,pValue_wald,h_wald,cValue_wald]=wald_test_p(betahat_u,N,x,rw,dof)
    alpha = 0.05; % significance level
    TMP = 0;
    for ii=1:N
        TMP = TMP + exp(x(ii,:)*betahat_u)*x(ii,:)'*x(ii,:);
    end
    Rw = [0 1]; 
    EstCovw = TMP\eye(2);
    stat_wald = (Rw*betahat_u-rw)' * inv(Rw*EstCovw*Rw') *(Rw*betahat_u-rw);
    pValue_wald = 1-chi2cdf(stat_wald,dof);
    h_wald= (pValue_wald<= alpha);
    cValue_wald = chi2inv(1-alpha,dof);
end

% LR test
function [stat_lr,pValue_lr,h_lr,cValue_lr]=lr_test_p(logL_r1,logL_u,dof)
    alpha = 0.05; % significance level
    stat_lr = 2*(logL_u-logL_r1);
    pValue_lr = 1-chi2cdf(stat_lr,dof);
    h_lr = (pValue_lr <= alpha);
    cValue_lr = chi2inv(1-alpha,dof);
end

% Score test
function [stat_score,pValue_score,h_score,cValue_score]=score_test_p(betahat_r1,N,x,y,dof)
    alpha = 0.05; % significance level
    uhat = y - exp(sum(x.*repmat(betahat_r1',N,1),2)) ;
    scorei = (x.*repmat(uhat,1,2))';
    score = sum(scorei,2);
    EstCovscore = 0;
    for ii=1:N
    EstCovscore = EstCovscore + exp(x(ii,:)*betahat_r1)*x(ii,:)'*x(ii,:);
    end
    EstCovscore = EstCovscore\eye(2); % covariance matrix of parameters
    stat_score= score'*EstCovscore*score;
    pValue_score = 1-chi2cdf(stat_score,dof);
    h_score = (pValue_score <= alpha);
    cValue_score= chi2inv(1-alpha,dof);
end