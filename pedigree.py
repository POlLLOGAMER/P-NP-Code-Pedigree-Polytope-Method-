"""
pedigree.py
============
Implementación del marco de Pedigree Polytope siguiendo el paper
"Lean 4 Machine-Verified Proof of P = NP via the Pedigree Polytope
Membership Problem" (Arthanari, 2026, arXiv:2606.03194).

Conceptos implementados:
  - Triángulos Δ_k = {i,j,k} con i<j<k
  - Pedigree: secuencia ({1,2,3},{i4,j4,4},...,{in,jn,n})
  - Decodificación pedigree -> ciclo Hamiltoniano (operador frontera ∂2)
  - MI-formulation del STSP: coste C_{ijk} = c_ik + c_jk - c_ij
  - Enumeración estructurada de pedigrees válidos
  - Resolución del STSP vía optimización sobre el politopo pedigree

NOTA: Este módulo realiza la búsqueda enumerativa sobre el conjunto de
pedigrees (el resultado teórico de Arthanari es que la membresía/optimización
es polinomial vía multicommodity flow + Tardos; aquí materializamos el
algoritmo a nivel combinatorio para que las demos sean ejecutables y
verificables).
"""

from __future__ import annotations
from typing import List, Tuple, Iterable, Optional
from itertools import product

Triangle = Tuple[int, int, int]   # (i,j,k) con i<j<k
Pedigree = Tuple[Triangle, ...]   # secuencia de n-2 triángulos


# ---------------------------------------------------------------------------
# Generación de pedigrees válidos
# ---------------------------------------------------------------------------
def edges_of_3tour() -> List[Tuple[int, int]]:
    """Aristas del 3-tour inicial sobre {1,2,3}: (1,2),(2,3),(1,3)."""
    return [(1, 2), (2, 3), (1, 3)]


def insert_city(edges: List[Tuple[int, int]],
                used_edge: Tuple[int, int],
                k: int) -> List[Tuple[int, int]]:
    """Inserta la ciudad k en used_edge=(i,j): retira (i,j), añade (i,k),(j,k)."""
    i, j = used_edge
    new = [e for e in edges if e != used_edge]
    new.append(tuple(sorted((i, k))))
    new.append(tuple(sorted((j, k))))
    return new


def enumerate_pedigrees(n: int) -> Iterable[Pedigree]:
    """Genera todos los pedigrees sobre [n].

    Un pedigree es una secuencia de n-2 triángulos, comenzando por {1,2,3},
    y para cada k=4..n se elige una arista disponible (i,j) (i<j<k) en
    el tour actual, generando el triángulo {i,j,k}.
    """
    base: Triangle = (1, 2, 3)
    if n == 3:
        yield (base,)
        return

    def rec(k: int, edges: List[Tuple[int, int]], seq: List[Triangle]):
        if k > n:
            yield tuple(seq)
            return
        # Para cada arista disponible (i,j), insertar k.
        for (i, j) in list(edges):
            new_edges = insert_city(edges, (i, j), k)
            seq.append((i, j, k))
            yield from rec(k + 1, new_edges, seq)
            seq.pop()

    yield from rec(4, edges_of_3tour(), [base])


# ---------------------------------------------------------------------------
# Decodificar pedigree -> ciclo Hamiltoniano (operador frontera)
# ---------------------------------------------------------------------------
def pedigree_to_tour(p: Pedigree, n: int) -> List[int]:
    """Devuelve el ciclo Hamiltoniano como lista [v1,v2,...,vn,v1]."""
    edges = edges_of_3tour()
    for tri in p[1:]:
        i, j, k = tri
        edges = insert_city(edges, (i, j), k)
    # Reconstrucción del ciclo a partir del conjunto de aristas
    adj = {v: [] for v in range(1, n + 1)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    tour = [1]
    prev = None
    cur = 1
    while len(tour) <= n:
        nxt = [v for v in adj[cur] if v != prev]
        if not nxt:
            break
        nxt = nxt[0]
        tour.append(nxt)
        prev, cur = cur, nxt
    return tour


# ---------------------------------------------------------------------------
# MI-formulation: coste de un pedigree dada matriz de costes c[i][j]
# ---------------------------------------------------------------------------
def mi_cost(p: Pedigree, c: List[List[float]]) -> float:
    """Coste total del tour codificado por el pedigree usando la MI-formulation.

    Coste base = c[1][2] + c[2][3] + c[1][3]
    Coste de insertar k en (i,j) = c[i][k] + c[j][k] - c[i][j]
    """
    total = c[1][2] + c[2][3] + c[1][3]
    for tri in p[1:]:
        i, j, k = tri
        total += c[i][k] + c[j][k] - c[i][j]
    return total


# ---------------------------------------------------------------------------
# Optimización STSP sobre el politopo Pedigree (M3P + MI)
# ---------------------------------------------------------------------------
def solve_stsp_via_pedigree(n: int,
                            c: List[List[float]],
                            forbidden_edges: Optional[set] = None
                            ) -> Tuple[float, Pedigree, List[int]]:
    """Resuelve el STSP minimizando coste sobre todos los pedigrees.

    forbidden_edges: conjunto de aristas frozenset({u,v}) que NO pueden
    aparecer en el tour. Se utiliza para reducciones (p.ej. Hamiltonian
    Cycle -> TSP).
    Devuelve (coste_min, pedigree_óptimo, tour_óptimo).
    """
    best_cost = float("inf")
    best_p: Optional[Pedigree] = None
    best_tour: Optional[List[int]] = None
    for p in enumerate_pedigrees(n):
        cost = mi_cost(p, c)
        if cost >= best_cost:
            continue
        tour = pedigree_to_tour(p, n)
        if forbidden_edges:
            ok = True
            for a, b in zip(tour, tour[1:]):
                if frozenset((a, b)) in forbidden_edges:
                    ok = False
                    break
            if not ok:
                continue
        best_cost = cost
        best_p = p
        best_tour = tour
    assert best_p is not None and best_tour is not None
    return best_cost, best_p, best_tour
