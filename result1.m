function [mape,mae,precision,mse,rmse,r2]=result1(mape,mae,bb,ttd,res2,initres,precision,mse,rmse,r2,th)
mse1=0;
for i=1:ttd
    mse1=mse1+((res2(1,i)-initres(1,i))^2);
end
mse1=mse1/ttd;

mae1=0;
for i=1:ttd
    mae1=mae1+abs(res2(1,i)-initres(1,i));
end
mae1=mae1/ttd;

mape1=0;
for i=1:ttd
    mape1=mape1+abs((res2(1,i)-initres(1,i))/res2(1,i));
end
mape1=mape1/ttd;

yk2=0;
for i=1:ttd
    yk2=yk2+(initres(1,i)^2);
end
fkyk=0;
for i=1:ttd
    fkyk=fkyk+(res2(1,i)*initres(1,i));
end
fk2yk2=0;
for i=1:ttd
    fk2yk2=fk2yk2+((res2(1,i)^2)*yk2);
end
r0=fkyk/fk2yk2;
mse(th,bb)=mse1/ttd;
rmse(th,bb)=sqrt(mse1)/ttd;
mae(th,bb)=mae1/ttd;
mape(th,bb)=mape1/ttd;
r2(th,bb)=r0/ttd;
end