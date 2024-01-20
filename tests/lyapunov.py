import numpy as np
import cvxpy as cvx


A = np.array([
	[1, 0],
	[0, 2]
])

P = cvx.Variable((2,2), symmetric=True)

LMI = A.T@P + P@A

constraints = [P >> 1e-7*np.eye(2), LMI << 0]

prob = cvx.Problem(cvx.Minimize(0), constraints)

prob.solve(solver=cvx.CVXOPT, verbose=True)

print(prob.status)
print(' ')

print(P.value)