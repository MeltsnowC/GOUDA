function b = Min_Max_Normalization(x)
b = zeros(size(x));
for i = 1: size(b,1)
    for j = 1:size(b,2)
        b(i,j) = (x(i,j)-min(x,[],'all'))/(max(x,[],'all')-min(x,[],'all'));
    end
end
