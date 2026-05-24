clc
close all
clear 
load('matlab.mat')
b1=size(Housing,1);
Housing2=zeros(size(Housing));
Housing2(:,1:5)= table2array(Housing(:,1:5));
Housing2(:,11)= table2array(Housing(:,11));

for i=1:b1
    if isequal(Housing{i,6},"yes")
        Housing2(i,6)=1;
    else
        Housing2(i,6)=0;
    end
    if isequal(Housing{i,7} ,'yes')
        Housing2(i,7)=1;
    else
        Housing2(i,7)=0;
    end
    if isequal(Housing{i,8} ,'yes')
        Housing2(i,8)=1;
    else
        Housing2(i,8)=0;
    end
    if isequal(Housing{i,9} ,'yes')
        Housing2(i,9)=1;
    else
        Housing2(i,9)=0;
    end
    if isequal(Housing{i,10} ,'yes')
        Housing2(i,10)=1;
    else
        Housing2(i,10)=0;
    end
    if isequal(Housing{i,12} ,'yes')
        Housing2(i,12)=1;
    else
        Housing2(i,12)=0;
    end
    if isequal(Housing{i,13} ,'radiator')
        Housing2(i,13)=2;
    elseif isequal(Housing{i,13} ,'package')
        Housing2(i,13)=1;
    else
        Housing2(i,13)=0;
    end
end
%% train and test
datatab2=Housing2(fix((2*b1)/3)+1:end,2:13);
datatab=Housing2(1:fix((2*b1)/3),2:13);
tag2=Housing2(fix((2*b1)/3)+1:end,1)';
tag1=Housing2(1:fix((2*b1)/3),1)';
tt=size(tag1,2);
%%
w = rand(1, 12) ;
yy5 = conv2(datatab, w, 'same');
%Mdl2 = fitcnb(yy5,tag1);
[net44,y]=NN(yy5',tag1);
y=y+mean(tag1);
%%
%DT
tree = fitctree(datatab,tag1);
view(tree);
view(tree,'mode','graph');
%KNN
Mdl = fitcknn(datatab,tag1);
%RF
RegTreeTemp = templateTree('Surrogate','On');
Mdl2 = fitensemble(datatab,tag1,'LSBoost',100,RegTreeTemp);
%%
cluster_n=4;
[marakez,u]=fcm(datatab,cluster_n); 
[value,index]=sort(u,1,'descend');   % moratabsazi sotoni nozooli
out2=index(1,:);
ttact=unique(tag1);
tact=size(unique(tag1),2);
mi=zeros(tact,cluster_n);
tedad=zeros(1,tact);
for cc=1:tact
    [zz]=find(tag1(1,:)==ttact(cc));
    sz=size(zz,2);
    tedad(1,cc)=sz;
end
for hh=1:cluster_n
    [vv]=find(out2(1,:)==hh);
    for cc=1:tact
        [zz]=find(tag1(1,vv)==ttact(cc));
        sz=size(zz,2);
        mi(cc,hh)=sz/tedad(1,cc);
    end
end
%%
tkh=5;
precision=zeros(2,tkh);
mse=zeros(2,tkh);
rmse=zeros(2,tkh);
r2=zeros(2,tkh);
mape=zeros(2,tkh);
mae=zeros(2,tkh);
iter=zeros(1,tkh);
ttkh=[30,60,90,120,150];
%%
for bb=1:tkh
    %for test data
    ttd=ttkh(1,bb);
    initres=tag2(1:ttd);
    res2=zeros(1,ttd);
    for th=1:2
        tic
        for hz=1:ttd
            ri=datatab2(hz,:);
            if th==1
                yyy = conv2(ri, w, 'same');
                res2(1,hz)=abs(net44(yyy'));
            else
                yyy = conv2(ri, w, 'same');
                res21=abs(net44(yyy'));
                %
                ai=zeros(1,cluster_n);
                pi2=zeros(tact,1);
                pkj=zeros(1,tact);
                for hh=1:cluster_n
                    ai(1,hh)=1+norm(ri-marakez(hh,:));
                end
                for gg2=1:tact
                    ppp=0;
                    for gg3=1:cluster_n
                        ppp=ppp+(mi(gg2,gg3)/ai(1,gg3));
                    end
                    pkj(1,gg2)=ppp/cluster_n;
                end
                [val,idx]=sort(pkj,2,'descend');
                res22=ttact(idx(1));
                %
                y2 = predict(tree,ri);
                y3 = predict(Mdl,ri);
                y4 = predict(Mdl2,ri);
                res23=(y2+y3+y4)/3;
                %
                res2(1,hz)=(res21+res22+res23)/3;
            end
        end
        [mape,mae,precision,mse,rmse,r2]=result1(mape,mae,bb,ttd,res2,initres,precision,mse,rmse,r2,th);
        delay(th,bb)=toc;
    end
    iter(1,bb)=ttd;
end
%iter=1:5;
%% result
figure
bar(iter,mse')
xlabel('number of test sample')
ylabel('MSE')
title('Mean Squre Error in House prices prediction')
leg1=legend('Base work','Proposed Scheme');
set(leg1,'Location','NorthWest');
% for i=1:tkh
% text(iter(i),mse(1,i),num2str(mse(1,i),'%05.2f'),'vert','bottom','horiz','center');
% text(iter(i),mse(2,i),num2str(mse(2,i),'%05.2f'),'vert','bottom','horiz','center');   
% end
 figure
bar(iter,rmse')
xlabel('number of test sample')
ylabel('RMSE')
title('Root Mean Squre Error in House prices prediction')
leg1=legend('Base work','Proposed Scheme');
set(leg1,'Location','NorthWest');

figure
bar(iter,r2')
xlabel('number of test sample')
ylabel('R2')
title('R2 index in House prices prediction')
leg1=legend('Base work','Proposed Scheme');
set(leg1,'Location','NorthWest');

figure
bar(iter,mae')
xlabel('number of test sample')
ylabel('MAE')
title('Mean absolute error in House prices prediction')
leg1=legend('Base work','Proposed Scheme');
set(leg1,'Location','NorthWest');

figure
bar(iter,mape')
xlabel('number of test sample')
ylabel('MAPE')
title('Mean Absolute Percentage Error in House prices prediction')
leg1=legend('Base work','Proposed Scheme');
set(leg1,'Location','NorthWest');
