#!/usr/bin/env python
# usage: python general.py [Number s primitives] [Number p primitives] ...
from multiprocessing.sharedctypes import Value
import site
import pyscf.gto
import pyscf.scf
import scipy.optimize as sco
import numpy as np
import sys
import json
import click

# JSON output, fix warning, make system independent per atom basis set^


class Molecule:
    atomspec: str
    elements: list
    basisspec: dict

    def __init__(self, atomspec: str, basisspec: str) -> dict:
        """Transform inputs once."""
        self.elements = [_.strip().split()[0] for _ in atomspec.split(";")]

        # give atoms unique names
        self._atomtypes = []
        seenelements = {_: 0 for _ in set(self.elements)}
        for element in self.elements:
            self._atomtypes.append(f"{element}{seenelements[element]}")
            seenelements[element] += 1

        # update atomspec with unique names
        self.atomspec = []
        for atomtype, spec in zip(self._atomtypes, pyscf.gto.format_atom(atomspec)):
            self.atomspec.append((atomtype, spec[1]))

        # parse basis
        self.basisspec = dict()
        for elementspec in basisspec.split(";"):
            element, spec = elementspec.split(":")
            self.basisspec[element] = [int(_) for _ in spec.split(".")]

    def initial_guess(self) -> list:
        """Guess for the first stage. Informed by empirical values."""
        guess = []
        for element in self.elements:
            guess += [400, 0.2] * sum(self.basisspec[element])

        self._is_tempered = True
        return guess

    def first_to_second(self, x0):
        # todo
        self._is_tempered = False

    def _get_basis(self, basis):
        if type(basis) == str:
            return basis
        if self._is_tempered:
            return self._tempered_to_basis(basis)
        raise ValueError("dang")

    def evaluate(self, basis) -> float:
        basis = self._get_basis(basis)
        try:
            mol = pyscf.gto.M(
                atom=self.atomspec,
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

    def _tempered_to_basis(self, tempered: list) -> list:
        """Converts a vector of tempered parameters into the corresponding PySCF basis set data structure."""
        tempered = np.abs(tempered)
        basis = dict()
        offset = 0
        for atomtype, element in zip(self._atomtypes, self.elements):
            site_basis = []
            angular = self.basisspec[element]
            nl = len(angular)

            subset = tempered[offset : offset + nl * 2]
            offset += len(angular)

            for l, N in zip(range(nl), angular):
                alpha, beta = subset[l * 2 : l * 2 + 2]
                site_basis += to_basis(even_tempered(l, alpha, beta, N), [l] * N)

            basis[atomtype] = site_basis
        return basis


def even_tempered(l: int, alpha: float, beta: float, N: int):
    return alpha * beta ** np.arange(1, N + 1)


def to_basis(exponents, Ns) -> list:
    return [[N, [exponent, 1]] for exponent, N in zip(exponents, Ns)]


# def relaxed_to_bas(exponents, angular):
#     return [[ang, [exponent, 1]] for exponent, ang in zip(exponents, angular)]


# def first_stage(tempered: list, system: Molecule) -> float:
#     """Simple optimization target."""
#     return system.evaluate(tempered_to_basis(tempered))


# def second_stage(x0, angular, atomspec):
#     """Simple optimization target."""
#     return do_mol(atomspec, {"Be": relaxed_to_bas(x0, angular)})


@click.command()
@click.option(
    "--atomspec",
    default="Be 0 0 -1.226791616343017; Be 0 0 1.226791616343017",
    help="System to optimize",
)
@click.option(
    "--basisspec",
    default="Be:3.1;Li:2.1",
    help="element:# basis primitives in increasing angular momenta.",
)
def twostage(atomspec, basisspec):
    system = Molecule(atomspec, basisspec)

    # reference calculations
    for basis in "D":
        print(f"{basis} contracted:", system.evaluate(f"cc-pV{basis}Z"))
        print(f"{basis} uncontracted:", system.evaluate(f"unc-cc-pV{basis}Z"))

    # first stage: evenly tempered basis set guess
    guess = system.initial_guess()
    res = sco.minimize(lambda _: system.evaluate(_), guess)
    print(f"First stage: {res.fun}")
    print(system._tempered_to_basis(res.x))

    # second stage: relax all coefficients from there
    # guess = system.first_to_second(res.fun)
    # res = sco.minimize(second_stage, guess, args=(system,))
    # print(f"Second stage: {res.fun}")
    # print(free_to_basis(res.x, system))

    # try:
    #     # init
    #     args = [int(_) for _ in sys.argv[1:]]
    #     x0 = [400, 0.2] * len(args)

    #     # first stage
    #     res = sco.minimize(first_stage, x0, args=(args,atomspec))
    #     print("Basis set")
    #     basis = tempered_to_bas(res.x, args)
    #     print(basis)
    #     print("Error to CBS [Ha]")
    #     print(res.fun - CBS)

    #     # second stage
    #     angular = [_[0] for _ in basis]
    #     exponents = [_[1][0] for _ in basis]
    #     print (relaxed_to_bas(exponents, angular))
    #     res = sco.minimize(second_stage, exponents, args=(angular,atomspec))
    #     print ("Basis set")
    #     print (relaxed_to_bas(res.x, angular))
    #     print ("Error to CBS [Ha]")
    #     print (res.fun - CBS)
    # except:
    #     basis = sys.argv[1]
    #     print("Contracted", do_mol(basis) - CBS)
    #     print(
    #         "Uncontracted",
    #         do_mol({"Be": pyscf.gto.uncontract(pyscf.gto.basis.load(basis, "Be"))})
    #         - CBS,
    #     )


if __name__ == "__main__":
    twostage()
