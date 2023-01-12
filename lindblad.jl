using LinearAlgebra
using Printf

if (length(ARGS)<5)
	println("Usage: julia lindblad.jl J gamma p tf dt")
	exit()
end

J = parse(Float32,ARGS[1])
gamma = parse(Float32,ARGS[2])
p=parse(Float32,ARGS[3])
tf = parse(Float32,ARGS[4]) 
dt = parse(Float32,ARGS[5])
omega = 1

sx = [0 1; 1 0]
sz = [1 0; 0 -1]
sm = [0 0; 1 0]
id2 = [1 0; 0 1]
id4 = kron(id2,id2)

function Hf(omega)
	hf = omega*(kron(sz,id2)+kron(id2,sz))/2.0
	return -im*(kron(hf,id4)-kron(id4,hf))
end

function Hint(J)
	hint = J*kron(sx,sx)
	return -im*(kron(hint,id4)-kron(id4,hint))
end

function Decoherence(gamma)
	L0 = kron(sm,id2)
	L1 = kron(id2,sm)

	decoherence0 = kron(L0,L0)-0.5*(kron(transpose(L0)*L0,id4)+kron(id4,transpose(L0)*L0))
	decoherence1 = kron(L1,L1)-0.5*(kron(transpose(L1)*L1,id4)+kron(id4,transpose(L1)*L1))
	return gamma*(decoherence0+decoherence1)
end

function EntanglementEntropy(r)
	r = reshape(r,(4,4))
	c = real((r[1,1]+r[2,2])*(r[3,3]+r[4,4])-(r[3,1]+r[4,2])*(r[1,3]+r[2,4]))
	l1 = 0.5*(1.0+sqrt(1.0-4.0*c))
	l2 = 0.5*(1.0-sqrt(1.0-4.0*c))
	return -l1*log2(l1)-l2*log2(l2)
end


M = Hf(omega)+Hint(J)+Decoherence(gamma)
dmatrix = [(1-p)/4; 0; (1-p)/4; 0; 0; (1+p)/4; p/2; (1-p)/4; (1-p)/4; p/2; (1+p)/4; 0; 0; (1-p)/4; 0; (1-p)/4]

t=0.0
while(t<tf)
	global dmatrix,t,dt
	k1 = dt*M*dmatrix
	k2 = dt*M*(dmatrix+k1/2.0)
	k3 = dt*M*(dmatrix+k2/2.0)
	k4 = dt*M*(dmatrix+k3)
	dmatrix = dmatrix+(k1+2.0*k2+2.0*k3+k4)/6.0
	@printf "%.4f %.4f\n" t EntanglementEntropy(dmatrix)	
	t+=dt
end
