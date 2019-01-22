clc
clear 
%load the dataset
U = load('MNISTnumImages5000.txt');
V=load('MNISTnumLabels5000.txt');
M=cat(2,U,V);
tr=M(1:4000,:);
ts=M(4001:5000,:);
eta0=0.3;
etah=0.3;
mom=0.1;
h=100;
a=0.1;
c=0.1;
b=0;
lambda=0.0001;

%Initialize the weight matrices to random numbers
w1=a.*randn(100,784)+b;
% w11=zeros(100,784);
w2=c.*randn(784,100)+b;
% w22=zeros(784,100);
rh=0.1;
beta = 3;
penal=0;
%normalize the data
for i =1:1:4000
    mean=0;
    sigma=0;
    for j=1:1:784
        mean=mean+tr(i,j);
        sigma=sigma+tr(i,j).^2;
    end
    tr(i,1:784)=(tr(i,1:784)-(mean/784))/(sigma/784);
end
costfunctionEpoch=[];
%Preprocessing is done and here starts the main code.
for epoch=1:1:10
    cost_Function_Digit=zeros(1,10);
    disp(epoch)
%shuffle the rows of input dataset.
tr = tr(randperm(size(tr,1)),:);
costfunctionBatch=[];
% for each of the input below, one batch of 20 datapoints run.
for input =1:1:200
    disp(input)
    from=((input-1)*20)+1;
    to=(input*20);
trbatch=tr(from:to,:);
x = zeros(1,100);
%Here starts the batch
for batch=1:1:20  
sig3=[];
%for each batch we take average activation.
for k=1:1:h
    pr1=dot(w1(k,:),trbatch(batch,1:784));
    sig33=1/(1+exp(-pr1));
    sig3(end+1)=sig33;
end
x=x+sig3;
end
x=x/20;
%we change the x values which are '0' or '1' as they might cause NaN in below formulaes
for i=1:1:100
    if x(i)==1
        x(i)=0.99;
    elseif x(i)==0
        x(i)=0.01;
    end
end
% batch runs from here
w11=zeros(100,784);
w22=zeros(784,100);
count=0;
Costfunction=[];
%each datapoint of a batch has forward propogation and back propogation as follows. 
for batch=1:1:20  
%forward prop from input to hidden
sig1=[];
for k=1:1:h
    pr1=dot(w1(k,:),trbatch(batch,1:784));
    sig11=1/(1+exp(-pr1));
    sig1(end+1)=sig11;
end
%forward prop from hidden to output.
sig2=[];
for k=1:1:784
    pr2=dot(sig1(1,:),w2(k,:));
    sig22=1/(1+exp(-pr2));
    sig2(end+1)=sig22;
end
% calculate error on each point
err1=trbatch(batch,1:784)-sig2(1,:);
%Below is the total error for a datapoint.
CostfunctionPoint=0;
for k=1:1:784
    CostfunctionPoint=CostfunctionPoint+err1(k).^2;
end

Costfunction(end+1)=(1/784)*CostfunctionPoint*(1/2);
%
cost_Function_Digit(trbatch(batch,785)+1)=cost_Function_Digit(trbatch(batch,785)+1)+(1/784)*CostfunctionPoint*(1/2);
%accumulate the changes for weight matrix w2
for i=1:1:784
    for j=1:h
        w22(i,j)=w22(i,j)+(eta0*err1(i)*sig2(i)*(1-sig2(i))*sig1(j));
    end
end
%Calculate the del values for hidden layer.
del=[];
for j=1:1:h
    de=0;
    penal = beta*( ((1-rh) / (1-x(j))) - (rh/x(j)));
    for i=1:784
        de=de+(w2(i,j)*err1(i) *sig2(i)*(1-sig2(i))) ;
    end
    del(end+1)=de-penal;
end
%accumulate the changes for weight matrix w1
for i=1:1:h
    for j=1:784
        w11(i,j)=w11(i,j)+(etah*sig1(i)*(1-sig1(i))*del(i)*trbatch(batch,j));
    end
end
end
%after the batch , make the changes to original matrices.
w2=0.05*w22+w2-(lambda*w2);
w1=0.05*w11+w1-(lambda*w1);
%calculate the error for entire batch and store it in array
costfunctionBatch(end+1)=sum(Costfunction)/20;
end
%%calculate the error for entire 200 groups of batch size 20 and store it in array
costfunctionEpoch(end+1)=sum(costfunctionBatch)/200;
end
cost_Function_Digit=cost_Function_Digit/400;% as 400 examples are there for each digit
%test the code for test data(cost function)
cost_Function_Digit_test=zeros(1,10);
for i =1:1:1000
    mean=0;
    sigma=0;
    for j=1:1:784
        mean=mean+ts(i,j);
        sigma=sigma+ts(i,j).^2;
    end
    ts(i,1:784)=(ts(i,1:784)-(mean/784))/(sigma/784);
end
for i=1:1:1000
sig1=[];
for k=1:1:h
    pr1=dot(w1(k,:),ts(i,1:784));
    sig11=1/(1+exp(-pr1));
    sig1(end+1)=sig11;
end
sig2=[];
for k=1:1:784
    pr2=dot(sig1(1,:),w2(k,:));
    sig22=1/(1+exp(-pr2));
    sig2(end+1)=sig22;
end
err1=ts(i,1:784)-sig2(1,:);
CostfunctionPoint=0;
for k=1:1:784
    CostfunctionPoint=CostfunctionPoint+err1(k).^2;
end
cost_Function_Digit_test(ts(i,785)+1)=cost_Function_Digit_test(ts(i,785)+1)+CostfunctionPoint;

end 
cost_Function_Digit_test=cost_Function_Digit_test/100 ;  %as 100 points of each digit.


%end of the code and here starts visualization
%code to display timeseries
figure(1)
x = [1,2,3,4,5,6,7,8,9,10];
y=costfunctionEpoch;
plot(x,y)
xlabel('Epochs') 
ylabel('Cost Function')

%code to display the features.
figure(2)
for i=1:10
    for j = 1:10
        v = reshape(w1((i-1)*10+j,:),28,28);
        subplot(10,10,(i-1)*10+j)
        image(64*v)
        colormap(gray(64));
        set(gca,'xtick',[])
        set(gca,'xticklabel',[])
        set(gca,'ytick',[])
        set(gca,'yticklabel',[])
        set(gca,'dataaspectratio',[1 1 1]);
    end
end
%plot bargraphs digitwise
val = [cost_Function_Digit_test(:) cost_Function_Digit(:)];
xval = [0 1 2 3 4 5 6 7 8 9];
figure(3)
bar_dig = bar(xval, val, 'grouped');
set(bar_dig(1), 'FaceColor', 'm')
set(bar_dig(2), 'FaceColor', 'g')
xlabel('Digits from "0" to "9"')
ylabel('Cost value')
legend('Train Cost','Test Cost')

m1=mean(cost_Function_Digit);
m2=mean(cost_Function_Digit_test);
val = [m1 m2];
xval = [0 1];
figure(4)
bar_dig = bar(xval, val);
xlabel('Training         Testing')
ylabel('Cost value')


