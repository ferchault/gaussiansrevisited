#!/usr/bin/env python
# usage: python general.py [Number s primitives] [Number p primitives] ...
import pyscf.gto
import pyscf.scf
import scipy.optimize as sco
import numpy as np
import sys
from typing import Iterable
import json

#JSON output, fix warning, make system independent per atom basis set^

def even_tempered(l: int, alpha: float, beta: float, N: int) -> np.ndarray[float]:
    return alpha * beta ** np.arange(1, N + 1)


def to_basis(exponents, Ns) -> Iterable:
    return [[N, [exponent, 1]] for exponent, N in zip(exponents, Ns)]


def do_mol(basis):
    try:
        mol = pyscf.gto.M(
            atom=f"Be 0 0 -1.226791616343017; Be 0 0 1.226791616343017",
            basis=basis,
            verbose=0,
        )
    except:
        return 0
    calc = pyscf.scf.RHF(mol)
    try:
        e = calc.kernel()
        if not calc.converged:
            return 0
        return e
    except:
        return 0


def tempered_to_bas(x0, Ns):
    x0 = np.abs(x0)
    nl = int(len(x0) / 2)
    bas = []
    for l, N in zip(range(nl), Ns):
        alpha, beta = x0[l * 2 : l * 2 + 2]
        bas += to_basis(even_tempered(l, alpha, beta, N), [l] * N)
    return bas

def relaxed_to_bas(exponents, angular):
    return [[ang, [exponent, 1]] for exponent, ang in zip(exponents, angular)]

def first_stage(x0, Ns):
    return do_mol({"Be": tempered_to_bas(x0, Ns)})

def second_stage(x0, angular):
    return do_mol({"Be": relaxed_to_bas(x0, angular)})


if __name__ == "__main__":
    CBS = -29.1341759449
    try:
        # init
        args = [int(_) for _ in sys.argv[1:]]
        x0 = [400, 0.2] * len(args)

        # first stage
        res = sco.minimize(first_stage, x0, args=(args,))
        print("Basis set")
        basis = tempered_to_bas(res.x, args)
        print(basis)
        print("Error to CBS [Ha]")
        print(res.fun - CBS)

        # second stage
        angular = [_[0] for _ in basis]
        exponents = [_[1][0] for _ in basis]
        print (relaxed_to_bas(exponents, angular))
        res = sco.minimize(second_stage, exponents, args=(angular,))
        print ("Basis set")
        print (relaxed_to_bas(res.x, angular))
        print ("Error to CBS [Ha]")
        print (res.fun - CBS)
    except:
        basis = sys.argv[1]
        print("Contracted", do_mol(basis) - CBS)
        print(
            "Uncontracted",
            do_mol({"Be": pyscf.gto.uncontract(pyscf.gto.basis.load(basis, "Be"))})
            - CBS,
        )
