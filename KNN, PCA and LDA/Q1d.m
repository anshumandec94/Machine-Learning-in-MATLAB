load('SPECT_valid.txt');
load('SPECT_train.txt');
load('SPECT_test.txt');
sprintf('No. Sigma \t value')
[p1,p2,pc1,pc2]=Bayesian_learning(SPECT_train,SPECT_valid);

%[p1,p2,pc1,pc2]=Bayes_learning(SPECT_train,SPECT_valid);
Bayesian_testing(SPECT_test,p1,p2,pc1,pc2);
%Bayes_testing(SPECT_test,p1,p2,pc1,pc2);