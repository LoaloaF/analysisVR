function [X, Y, StartY] = GetSync(Sync_in)
%Input array 13xN
%Earlier values are in raws that are lower
%Thus, flip array upside down
Sync_t = flipud(Sync_in);
%reshape Sync_t
Sync_t = reshape(Sync_t,1,[]);

L = false(8,length(Sync_t));
%vector for remapping
V = [0x80 0x40 0x20 0x10 0x08 0x04 0x02 0x01];
%remapping into bit array 
for I_t = 1:8  
   L(I_t,:) = bitand(V(I_t), Sync_t)>0; 
 end
L = reshape(L,104,[]);
L(101:104,:) = []; %delete unused bits
L = reshape(L,1,[]);
X = 1+find((L(2:end) & ~L(1:(end-1))) | (~L(2:end) & L(1:(end-1)))); %rising and falling, indexes
Y = L(X); %mark rizing by true;
StartY = L(1);
end