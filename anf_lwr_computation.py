import matplotlib.pyplot as plt

from anf_lwr_utils import *
from collections import defaultdict, Counter
import itertools as it
from tqdm import tqdm
from sympy.utilities.iterables import multiset_permutations  # Used to enumerate the orbit of a vector
from sage.crypto.boolean_function import BooleanFunction
from hashlib import sha256

TQDM_DISABLE = False


def cardinality_stabilizer_monomial(v):
    s = set(v)
    card_stabilizer = 1
    for value in s:
        if value:  # The 0 values is not taken into account (i.e. unused variables are not taken into account)
            card_stabilizer *= factorial(v.count(value))
    return card_stabilizer


def cardinality_orbit_permuted_vector(v):
    """ Return the cardinality of the orbit of a vector v for the natural action of the
    symmetric group: Orb(v) = {s.v, s in Sym(d)} """

    assert factorial(len(v)) % cardinality_stabilizer_monomial(v) == 0
    return factorial(len(v)) // cardinality_stabilizer_monomial(v)


def precomputations_of_ax_power_u(m):
    # m exponent DOTn(a, x)^m
    word_size = m.bit_length()

    # The ring GF(2)[b_i, y_i, 0 <= i < w] used for the generic ANF of (a.x)^u where u \in [0, m]
    generic_ring = BooleanPolynomialRing(2*word_size, ['b%d'%i for i in range(word_size)] + ['y%d'%i for i in range(word_size)])
    vars_b, vars_y = generic_ring.gens()[:word_size], generic_ring.gens()[word_size:]

    # The coordinates of a.x
    ax = binary_multiplication(vars_b, vars_y)
    # The products (a.x)^u for all u
    ax_power_u = []
    for u in range(m + 1):
        bin_u = [int(c) for c in bin(u)[2:][::-1]]  # u written in binary u = (u_0, u_1, ...)
        ax_power_u.append(product([ax[ind_c] for ind_c, c in enumerate(bin_u) if c]))  # prod_{i} (a.x)^u_i

    return generic_ring, ax_power_u


def canonical_monomials_ax_of_fixed_support_length(d, m, generic_ring, anf_ax_power_u):
    word_size = m.bit_length()
    # The ring GF(2)[a_i_j, x_i_j]
    ring, vars_a, vars_x = lwr_ring(word_size, d)

    # The d dictionaries associating for each i the variables b_j's to a_i_j's. and y_j's to x_i_j's.
    substitution_dicts = []
    for i in range(d):
        l = [(generic_ring('b%d'%j), ring('a%d_%d'%(i, j))) for j in range(word_size)] + \
            [(generic_ring('y%d' % j), ring('x%d_%d' % (i, j))) for j in range(word_size)]
        substitution_dicts.append({d1: d2 for d1, d2 in l})

    canonical_monomials_dict = defaultdict(int)

    # For all partitions of m of length d
    for partition in tqdm(Partitions(m, length=d), desc="Partitions of length %d"%d, disable=TQDM_DISABLE):
        # 1) Compute the size of Orb(partition)
        orbit_partition_cardinality = cardinality_orbit_permuted_vector(list(partition))
        # 2) Compute the terms of one of the product and count the corresponding number of canonical monomials with multiplicity
        initial_product = product([anf_ax_power_u[p].subs(substitution_dicts[ind_p]) for ind_p, p in enumerate(partition)])
        canonical_terms_in_initial_product = Counter([monomial_to_canonical_mask(t) for t in initial_product])

        # 3) For each encountered canonical monomial a^ux^v
        for canon_monom, mult_in_initial_prod in canonical_terms_in_initial_product.items():
            # 3.1) Compute the size of Orb(a^ux^v)
            orbit_cardinality = cardinality_orbit_permuted_vector(canon_monom)
            # 3.2) Compute the overall number of appearance of a^ux^v (see paper)
            final_mult = (mult_in_initial_prod * orbit_partition_cardinality) / orbit_cardinality
            # 3.3) Keep a^ux^v iff it appears an odd number of times
            if final_mult % 2 == 1:
                canonical_monomials_dict[canon_monom] += 1
            assert mult_in_initial_prod * orbit_partition_cardinality % orbit_cardinality == 0

    # In the obtained dictionary, only keep the canonical monomial with an odd number of appearances.
    # For each a^ux^v, we store a^u in the bucket corresponding to x^v
    canonical_monomial_sorted_by_v = defaultdict(list)
    for monom, mult in canonical_monomials_dict.items():
        if mult % 2 == 1:
            mask_a, mask_x = [], []
            for ai, xi in monom:
                mask_a.append(ai)
                mask_x.append(xi)
            canonical_monomial_sorted_by_v[tuple(mask_x)].append(tuple(mask_a))
    return canonical_monomial_sorted_by_v


def orbit_of_au_in_stab_xv_iterator(mask_au, dict_xv):
    """ Return the set {\sigma . u, \sigma \in \stab(v)}.
    """

    orbits_of_subvectors = []
    for v, bucket in dict_xv.items():
        # Consider a maximal subvector of u,  w = {u_i}_{i \in I},  s.t. #{v_i, i \in I} = 1.
        sub_vector_u = [mask_au[i] for i in bucket]
        # Compute and store the orbit of w
        orbits_of_subvectors.append(multiset_permutations(sub_vector_u))

    # Enumerate the cartesian product of the sets Orb(sub_vector_u) for all subvectors
    # No need for reordering since (u, v) is s.t. v is already sorted.
    for p in it.product(*orbits_of_subvectors):
        p = tuple(it.chain(*p))
        yield p


def card_orbit_of_au_in_stab_xv(mask_au, dict_xv):
    """ Return the set {\sigma . u, \sigma \in \stab(v)}.
    """

    card = 1
    for v, bucket in dict_xv.items():
        # Consider a maximal subvector of u,  w = {u_i}_{i \in I},  s.t. #{v_i, i \in I} = 1.
        sub_vector_u = [mask_au[i] for i in bucket]
        # Compute and store the orbit of w
        card *= cardinality_orbit_permuted_vector(sub_vector_u)
    return card


def permutation_u_to_v(mask_u, mask_v):
    p = [None for _ in range(len(mask_u))]
    last_encountered = dict()
    for i, val in enumerate(mask_v):
        if val not in last_encountered:
            last_encountered[val] = -1
        for j in range(last_encountered[val] + 1, len(mask_u)):
            if mask_u[j] == val:
                last_encountered[val] = j
                p[i] = j
                break
    return p


def sn_modulo_stab_u_iterator(mask_u):
    for x in multiset_permutations(mask_u):  # For x in Orb(u)
        yield permutation_u_to_v(mask_u, x)


def action_on_polynomial(poly_masks, perm):
    return set([tuple([m[perm[i]] for i in range(len(m))]) for m in poly_masks])


def coefficients_of_canonical_xv_iterator(canonical_monomials_ax_sorted_by_x):

    for mask_xv, list_of_mask_au in canonical_monomials_ax_sorted_by_x.items():
        coeff_of_xv = set()
        # The dictionary dict_xv is built such that for any b \in \NN: dict_v[b] = {i s.t. v_i = b}.
        # Thus, dict_v.keys = {v_i, 0 \leq i \leq d-1}
        dict_xv = defaultdict(set)
        for i_m, m in enumerate(mask_xv):
            dict_xv[m].add(i_m)

        for mask_au in list_of_mask_au:
            for au_permuted in orbit_of_au_in_stab_xv_iterator(mask_au, dict_xv):
                if au_permuted in coeff_of_xv:
                    print('Problem')
                coeff_of_xv.add(au_permuted)
        # The coefficient of x^v is fully computed
        yield mask_xv, frozenset(coeff_of_xv)


def orbit_and_stabilizer_poly_mod_stabilizer_monom(monom_mask, coeff_masks):
    orbit_of_poly = set()
    orbit_of_poly.add(coeff_masks)

    stab_poly_mod_stab_monom = []

    for perm in sn_modulo_stab_u_iterator(monom_mask):
        permuted_coeff = action_on_polynomial(coeff_masks, perm)
        if permuted_coeff == coeff_masks:
            stab_poly_mod_stab_monom.append(perm)
        else:
            orbit_of_poly.add(frozenset(permuted_coeff))

    return stab_poly_mod_stab_monom, orbit_of_poly


def digest_coeff(coeff):
    return sha256(str(sorted(coeff)).encode('utf-8')).hexdigest()


def count_canonical_coefficients(canonical_monomials_ax_sorted_by_x, n_min):
    """
    From the list of coefficients of canonical monomials, compute the list of canonical coefficients alpha \in GF(2)[a]
    """

    dict_orbit_size_by_length = dict()
    ring, vars_a, vars_x = lwr_ring(m.bit_length(), n_min)

    for mask_xv, coeff_of_xv in tqdm(coefficients_of_canonical_xv_iterator(canonical_monomials_ax_sorted_by_x), desc="Coefficients of each xv", disable=TQDM_DISABLE, total=len(canonical_monomials_ax_sorted_by_x)):
        l = len(coeff_of_xv)
        if l not in dict_orbit_size_by_length:
            dict_orbit_size_by_length[l] = defaultdict(tuple)
        already_seen = False
        orbit_of_coeff = set()
        for perm in sn_modulo_stab_u_iterator(mask_xv):
            digest = digest_coeff(action_on_polynomial(coeff_of_xv, perm))
            if digest in dict_orbit_size_by_length[l]:
                already_seen = True
                break
            orbit_of_coeff.add(digest)
        if not already_seen:  # Otherwise, compute the orbit and store it
            if n_min < 16:
                coeff = sum([ring(mask_to_monomial_str(t)) for t in coeff_of_xv])
                r = BooleanPolynomialRing(len(coeff.variables()), [str(v) for v in coeff.variables()])
                coeff = sum([r(t) for t in coeff.terms()])
                fun = BooleanFunction(coeff)
                lut = fun.truth_table(format='int')
                dict_orbit_size_by_length[l][digest_coeff(coeff_of_xv)] = (len(orbit_of_coeff), lut.count(1)/len(lut))
            else:
                dict_orbit_size_by_length[l][digest_coeff(coeff_of_xv)] = (len(orbit_of_coeff), -1)

    return dict_orbit_size_by_length


def bias_coefficients_of_canonical_monomials(canonical_monomials_ax_sorted_by_x, m, n_min, n_max):
    size_orbit_size_supp_bias = []
    ring, vars_a, vars_x = lwr_ring(m.bit_length(), n_min)
    for mask_xv, coeff_of_xv in tqdm(coefficients_of_canonical_xv_iterator(canonical_monomials_ax_sorted_by_x), desc="Coefficients of each xv", disable=TQDM_DISABLE, total=len(canonical_monomials_ax_sorted_by_x)):
        coeff = sum([ring(mask_to_monomial_str(t)) for t in coeff_of_xv])
        r = BooleanPolynomialRing(len(coeff.variables()), [str(v) for v in coeff.variables()])
        coeff = sum([r(t) for t in coeff.terms()])
        fun = BooleanFunction(coeff)
        lut = fun.truth_table(format='int')
        proba_of_1 = lut.count(1)/len(lut)
        size_orbit_size_supp_bias.append((cardinality_orbit_permuted_vector(mask_xv), len(mask_xv), proba_of_1))
    avg_list = []
    for n in range(n_min, n_max + 1):
        avg_list.append(0)
        for size_orb, size_supp, bias in size_orbit_size_supp_bias:
            avg_list[-1] += size_orb * bias * binomial(n, size_supp)
        avg_list[-1] /= (binomial(n + m, n) -1)

    return avg_list


def plot_some_curves(n_max, m_values, plots, filename, categories=None):
    plt.figure(figsize=(11, 3))
    plots_obj = []
    linestyles = ['-', "-", "-", "--", '--', '--']
    if categories is not None:
        enum_m_vals = m_values + m_values
    else:
        enum_m_vals = m_values
    for i_m, m in enumerate(enum_m_vals):
        p, = plt.plot(range(m, n_max + 1), plots[i_m], label="$m=%d$"%m, linestyle=linestyles[i_m])
        plots_obj.append(p)
    xticks = range(1, int(log(n_max, 2)) + 1)
    for i in xticks:
        plt.axvline(x=2**i, color='black', alpha=0.15, linestyle='--')

    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.yscale('log', base=2)
    plt.xscale('log', base=2)
    plt.xlim(xmin=2, xmax=n_max)
    plt.yticks(fontsize=16)

    dummy_tophead, = plt.plot([0], marker='None',
               linestyle='None', label='dummy-tophead')
    plt.xticks([2**i for i in xticks], labels=["$2^{%d}$"%i for i in xticks], fontsize=16)
    if categories is not None:
        leg = plt.legend([dummy_tophead] + plots_obj[:len(m_values)] + [dummy_tophead] + plots_obj[len(m_values):],
                [categories[0]] + ["$m=%d$"%m for m in m_values] + [categories[1]] + ["$m=%d$"%m for m in m_values], ncol=2, fontsize=12)  # Two columns, vertical group labels
        plt.gca().add_artist(leg)

    plt.savefig(filename)


if __name__ == "__main__":
    n_max = 4096
    m_values = [2, 4, 8]
    sparsity_xv = []
    sparsity_alpha = []

    precomputations_of_ax_power_u(max(m_values))

    average_sparsity_equations = []
    average_sparsity_equations_Qnm = []
    for m in m_values:
        print('current m', m)
        generic_ring, anf_ax_power_u = precomputations_of_ax_power_u(m)

        mult_xv = []
        mult_alpha = []
        mult_sparsity_Qnm = []

        all_canon_monom = dict()
        for d in range(1, m+1):

            canonical_monomials_ax_sorted_by_x = canonical_monomials_ax_of_fixed_support_length(d, m, generic_ring, anf_ax_power_u)
            for k, v in canonical_monomials_ax_sorted_by_x.items():
                all_canon_monom[k] = v
            multiplicity_monomials_xv = sum([cardinality_orbit_permuted_vector(t) for t in canonical_monomials_ax_sorted_by_x.keys()])
            mult_xv.append(multiplicity_monomials_xv)

            # canonical coeff of support of size d
            canonical_coefficients_orbit_size = count_canonical_coefficients(canonical_monomials_ax_sorted_by_x, m)
            # Nb of canonical coeff of support of size d
            multiplicity_coeff_alpha = sum(sum([len_orb for len_orb, bias in l.values()]) for l in canonical_coefficients_orbit_size.values())
            mult_alpha.append(multiplicity_coeff_alpha)
            if m < 16:
                # Avg sparsity (in the basis Qn, m) for coeff of |supp| = d
                sparsity_multiplicity_supp_d = sum(sum([len_orb * bias for len_orb, bias in l.values()]) for l in canonical_coefficients_orbit_size.values())
                mult_sparsity_Qnm.append(sparsity_multiplicity_supp_d)

        if m < 16:
            avg_list = bias_coefficients_of_canonical_monomials(all_canon_monom, m, m, n_max)
            average_sparsity_equations.append(avg_list)


        sparsity_xv.append([])
        sparsity_alpha.append([])
        average_sparsity_equations_Qnm.append([])
        for n in range(m, n_max + 1):
            sparsity_xv[-1].append(0)
            sparsity_alpha[-1].append(0)
            average_sparsity_equations_Qnm[-1].append(0)
            for d in range(1, m + 1):
                sparsity_xv[-1][-1] += mult_xv[d-1] * binomial(n, d)
                sparsity_alpha[-1][-1] += mult_alpha[d-1] * binomial(n, d)
                if m < 16:
                    average_sparsity_equations_Qnm[-1][-1] += mult_sparsity_Qnm[d-1] * binomial(n, d)
            average_sparsity_equations_Qnm[-1][-1] /= (binomial(n + m, n) - 1)
            if m in [2, 4, 8, 16]:
                assert sparsity_xv[-1][-1] == binomial(n + m, m) - 1
            if n in [64, 128, 256]:
                print("m=%d"%m, "n=%d"%n, "rank=%d"%sparsity_alpha[-1][-1], "sparsity=%.5f"%average_sparsity_equations_Qnm[-1][-1])


    plots = [[s / t for s, t in zip(sparsity_alpha[i_m], sparsity_xv[i_m])] for i_m, m in enumerate(m_values)]
    plot_some_curves(n_max, m_values, plots, "02_10_25_ratio_nb_terms.pdf")
    plot_some_curves(n_max, m_values, average_sparsity_equations, "02_10_25_avg_sparsity.pdf")
    plot_some_curves(n_max, m_values, average_sparsity_equations + average_sparsity_equations_Qnm, "02_10_25_avg_sparsity_Qnm.pdf", ["$\mathrm{Exp}_{x}(F^{m,n})$", "$\mathcal{Q}^{m,n}$"])
