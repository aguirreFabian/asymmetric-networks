H[x_] := 1/2 * Erfc[x/Sqrt[2]];
NG[x_, mu_, sigma2_] := 1/Sqrt[2*Pi*sigma2]*Exp[-1/2 * ((x-mu)^2)/sigma2];
N0[x_] := 1/Sqrt[2*Pi]*Exp[-1/2*x^2];
f1[A_, B_] := NIntegrate[x*N0[x]*H[(A-x)/B],{x,-Infinity,Infinity}]/NIntegrate[N0[x]*H[(A-x)/B],{x,-Infinity,Infinity}];
f2[A_, B_] := NIntegrate[x^2*N0[x]*H[(A-x)/B],{x,-Infinity,Infinity}]/NIntegrate[N0[x]*H[(A-x)/B],{x,-Infinity,Infinity}];
g[{x_, y_, q_, eta_, kappa_, t_, T_}] := {x*(1-t) + t*eta*f1[kappa - q*x, Sqrt[q*(y-x^2)]], y*(1-t) + t*(1 - eta^2 + eta^2*f2[kappa - q*x,Sqrt[q*(y-x^2)]]), q, eta, kappa, t, T};
mueq[q_, eta_, kappa_, T_]:= Nest[g, {0, 1, q, eta, kappa, 0.5, T}, T];
vals = Table[Table[Table[Table[mueq[q, eta, kappa*N[Sqrt[q+1]], T],{eta, -0.9, 0.9, 0.1}],{T, 100, 300, 100}],{kappa, 0.5, 2, 0.5}],{q,2,21,3}];
Print[vals];
