#!/usr/bin/env python
import pyscf.gto
import pyscf.scf
import scipy.optimize as sco
import numpy as np
import click
import json
from concurrent import futures
import os
import nevergrad as ng
import warnings

warnings.filterwarnings("ignore")


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
            guess += [400, 0.3] * sum(self.basisspec[element])

        return guess

    def first_to_second(self, x0):
        self._skeleton_basis = self._tempered_to_basis(x0)
        guess = []
        for atomtype in sorted(self._skeleton_basis.keys()):
            guess += [_[1][0] for _ in self._skeleton_basis[atomtype]]
        return guess

    def _relaxed_to_basis(self, basis):
        basisspec = self._skeleton_basis.copy()
        basis = list(basis)
        for atomtype in sorted(basisspec.keys()):
            for nprimitive in range(len(basisspec[atomtype])):
                basisspec[atomtype][nprimitive][1][0] = basis.pop()
        return basisspec

    def evaluate_reference(self, basis) -> float:
        return self._evaluate(basis)

    def evaluate_tempered(self, basis) -> float:
        return self._evaluate(self._tempered_to_basis(basis))

    def evaluate_relaxed(self, basis) -> float:
        return self._evaluate(self._relaxed_to_basis(basis))

    def _evaluate(self, basis) -> float:
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
                even_tempered = alpha * beta ** np.arange(1, N + 1)
                site_basis += [
                    [N, [exponent, 1]] for exponent, N in zip(even_tempered, [l] * N)
                ]

            basis[atomtype] = site_basis
        return basis


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
@click.option(
    "--budget", default=100, help="Number of trial runs for the initial guess."
)
@click.option(
    "--reference", default=False, help="Whether to do the reference calculations."
)
def twostage(atomspec, basisspec, budget, reference):
    system = Molecule(atomspec, basisspec)
    resultpack = {"atomspec": system.atomspec, "basisspec": basisspec}

    # reference calculations
    if reference:
        for nzeta in "DTQ5":
            for contraction in ["", "unc"]:
                basis = f"{contraction}cc-pV{nzeta}Z"
                resultpack[basis] = system.evaluate_reference(basis)

    # first stage: evenly tempered basis set guess
    guess = system.initial_guess()
    parametrization = len(guess)
    optimizer = ng.optimizers.NGOpt(
        parametrization=parametrization, budget=budget, num_workers=os.cpu_count()
    )
    with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
        recommendation = optimizer.minimize(system.evaluate_tempered, executor=executor)
    guess = recommendation.value
    res = sco.minimize(system.evaluate_tempered, guess)
    resultpack["tempered"] = res.fun
    resultpack["tempered_basis"] = system._tempered_to_basis(res.x)

    # second stage: relax all coefficients from there
    guess = system.first_to_second(res.x)
    res = sco.minimize(system.evaluate_relaxed, guess)

    resultpack["relaxed"] = res.fun
    resultpack["relaxed_basis"] = system._relaxed_to_basis(res.x)

    print(json.dumps(resultpack))


if __name__ == "__main__":
    twostage()
