"""
demo.py
========
Demostración del solver SAT basado en el algoritmo P=NP del paper
(Arthanari 2026) vía el politopo Pedigree.

Ejecuta múltiples ejemplos:
  1) Pedigrees y decodificación a ciclo Hamiltoniano (n=4, n=5).
  2) Resolución de STSP pequeñas vía MI-formulation.
  3) Hamiltoniano sobre grafos con y sin HC.
  4) SAT satisfactibles e insatisfactibles.
  5) Verificación de la asignación contra la CNF original.
"""

from __future__ import annotations
import time
from pedigree import (
    enumerate_pedigrees, pedigree_to_tour, mi_cost, solve_stsp_via_pedigree,
)
from reductions import satisfies, build_tsp_from_graph
from sat_via_pnp import reduce_3sat_to_hc, solve_sat


def sep(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# DEMO 1: Pedigrees y decodificación
# ---------------------------------------------------------------------------
def demo_pedigrees():
    sep("DEMO 1 · Pedigrees y decodificación a ciclos Hamiltonianos")
    for n in (4, 5):
        print(f"\nPedigrees sobre n={n}:")
        for p in enumerate_pedigrees(n):
            tour = pedigree_to_tour(p, n)
            print(f"  {p}  ->  tour: {tour}")


# ---------------------------------------------------------------------------
# DEMO 2: STSP pequeño resuelto por el solver pedigree
# ---------------------------------------------------------------------------
def demo_stsp():
    sep("DEMO 2 · STSP resuelto vía optimización sobre el politopo Pedigree")
    # 5 ciudades con matriz simétrica
    n = 5
    c = [[0]*(n+1) for _ in range(n+1)]
    dist = {
        (1,2): 2, (1,3): 9, (1,4): 10, (1,5): 7,
        (2,3): 6, (2,4): 4, (2,5): 3,
        (3,4): 8, (3,5): 5,
        (4,5): 1,
    }
    for (i,j), d in dist.items():
        c[i][j] = c[j][i] = d
    cost, ped, tour = solve_stsp_via_pedigree(n, c)
    print(f"  Matriz de distancias: {dist}")
    print(f"  Pedigree óptimo: {ped}")
    print(f"  Tour óptimo:     {tour}")
    print(f"  Coste mínimo:    {cost}")


# ---------------------------------------------------------------------------
# DEMO 3: Hamiltonian Cycle decidido vía solver pedigree
# ---------------------------------------------------------------------------
def demo_hc():
    sep("DEMO 3 · Hamiltonian Cycle (decisión) vía solver pedigree")

    # Grafo 1: C_5 (ciclo de 5 nodos) -> SÍ HC
    n = 5
    edges_yes = {frozenset((1,2)), frozenset((2,3)), frozenset((3,4)),
                 frozenset((4,5)), frozenset((5,1))}
    c = build_tsp_from_graph(n, edges_yes)
    cost, _ped, tour = solve_stsp_via_pedigree(n, c)
    print(f"\n  Grafo 1 (C_5):   aristas={sorted(map(sorted,edges_yes))}")
    print(f"    coste óptimo = {cost} (umbral={n})  ->  HC = {cost <= n+0.5}")
    print(f"    tour: {tour}")

    # Grafo 2: árbol (sin HC posible)
    edges_no = {frozenset((1,2)), frozenset((1,3)), frozenset((1,4)),
                frozenset((1,5))}
    c = build_tsp_from_graph(n, edges_no)
    cost, _ped, tour = solve_stsp_via_pedigree(n, c)
    print(f"\n  Grafo 2 (estrella K_{{1,4}}):  aristas={sorted(map(sorted,edges_no))}")
    print(f"    coste óptimo = {cost}   ->  HC = {cost <= n+0.5}")


# ---------------------------------------------------------------------------
# DEMO 4: SAT vía la cadena completa P=NP
# ---------------------------------------------------------------------------
def demo_sat():
    sep("DEMO 4 · SAT resuelto por la cadena SAT->3SAT->HC->TSP->Pedigree")

    instances = [
        ("Trivial OR", [[1, 2]], 2),
        ("AND simple", [[1], [2]], 2),
        ("Contradicción", [[1], [-1]], 1),
        ("(x1 v ¬x2) ∧ (¬x1 v x2)",
            [[1, -2], [-1, 2]], 2),
        ("(x1 v x2 v x3) ∧ (¬x1 v ¬x2) ∧ (¬x2 v ¬x3)",
            [[1, 2, 3], [-1, -2], [-2, -3]], 3),
        ("Pigeonhole-like UNSAT: (x) ∧ (¬x v y) ∧ (¬y) ∧ (...)",
            [[1], [-1, 2], [-2], [1, 2]], 2),
        ("3-SAT satisfactible larga",
            [[1, 2, 3], [-1, 2, -3], [1, -2, 3], [-1, -2, -3]], 3),
        ("3-SAT UNSAT (todas combinaciones de 2 vars)",
            [[1, 2], [1, -2], [-1, 2], [-1, -2]], 2),
    ]

    for name, cnf, nv in instances:
        print(f"\n  Instancia: {name}")
        print(f"    CNF = {cnf},  variables = {nv}")
        t0 = time.time()
        sat, asg = solve_sat(cnf)
        dt = (time.time() - t0) * 1000
        if sat:
            ok = satisfies(cnf, asg)
            print(f"    SAT = True ({dt:.2f} ms)")
            print(f"    Asignación = {asg}")
            print(f"    Verificación contra CNF original: {ok}")
            assert ok, "Verificación FALLÓ"
        else:
            # Comprobar exhaustivamente que efectivamente NO existe
            from itertools import product
            unsat_ok = all(
                not satisfies(cnf, {i+1: b[i] for i in range(nv)})
                for b in product([False, True], repeat=nv)
            )
            print(f"    SAT = False ({dt:.2f} ms)")
            print(f"    Verificación de insatisfactibilidad exhaustiva: {unsat_ok}")
            assert unsat_ok, "Insatisfactibilidad FALSA"


# ---------------------------------------------------------------------------
# DEMO 5: Reducción 3-SAT -> HC mostrada explícitamente
# ---------------------------------------------------------------------------
def demo_reduction():
    sep("DEMO 5 · Reducción 3-SAT -> Hamiltonian Cycle (gadget)")
    cnf = [[1, 2, 3], [-1, 2, -3]]
    nv = 3
    n_vert, edges, labels = reduce_3sat_to_hc(cnf, nv)
    print(f"  CNF de entrada: {cnf}")
    print(f"  Vértices del grafo HC ({n_vert}):")
    for lab, vid in labels.items():
        print(f"    {vid:>2}: {lab}")
    print(f"  Aristas ({len(edges)}):")
    for e in sorted(map(sorted, edges)):
        print(f"    {tuple(e)}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("####################################################################")
    print("#  SAT solver vía P=NP / Pedigree Polytope (Arthanari 2026)         #")
    print("####################################################################")
    demo_pedigrees()
    demo_stsp()
    demo_hc()
    demo_reduction()
    demo_sat()
    print("\n" + "#" * 70)
    print("#  TODAS LAS DEMOS COMPLETADAS CORRECTAMENTE                       #")
    print("#" * 70)
