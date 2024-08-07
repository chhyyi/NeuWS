function z = gzn2(tpixel, NApixel,m)
% Zernike modes are generated using reference code:
% https://kr.mathworks.com/matlabcentral/fileexchange/7687-zernike-polynomials

%tpixel is the total num of the image; NApixel is diameter of the NA
x=linspace(-tpixel/NApixel,tpixel/NApixel,tpixel);
[X,Y] = meshgrid(x,x);
[theta,r] = cart2pol(X,Y);
idx = r<=1;
z = zeros(size(X));
z(idx) = zernfun2(m,r(idx),theta(idx));

%figure
%pcolor(x,x,z), shading interp
%axis square, colorbar
%title(['Zernike function Z_' num2str(n) '^' num2str(m) '(r,\theta)'])
end