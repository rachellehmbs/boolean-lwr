from sage.all import *

##############
# ANF of LWR #
##############
def binary_sum(x, y):
    """ Compute the binary representation of the integer sum of two integers x and y of the same length q.
    Also works if x and y are replaced by vectors of Boolean polynomials.
    :param x: The first vector.
    :param y: The second vector.
    :return: The binary sum of x and y.
    """
    assert len(x) == len(y)
    q = len(x)
    c = [0 for _ in range(q)]  # c stands for carry
    bin_sum = [0 for _ in range(q)]

    for i in range(q):
        if i > 0:
            c[i] = x[i-1]*y[i-1] + x[i-1]*c[i-1] + y[i-1]*c[i-1]
        bin_sum[i] = x[i] + y[i] + c[i]
    return bin_sum


def multiple_binary_sum(list_of_words):
    """ Compute the binary representation of the integer sum of a list of integers
     of the same length q.
     Also works if each element is replaced by a vector of Boolean polynomials."""
    length = len(list_of_words)
    assert all([len(x) == len(list_of_words[0]) for x in list_of_words])
    if length == 1:
        return list_of_words[0]
    elif length == 2:
        return binary_sum(list_of_words[0], list_of_words[1])
    else:
        first_half = multiple_binary_sum(list_of_words[:length//2])
        second_half = multiple_binary_sum(list_of_words[length // 2:])
        return binary_sum(first_half, second_half)


def binary_multiplication(a, x):
    """ Compute the binary representation of the multiplication of two integers a and x
     of the same length q.
    Also works if a and x are replaced by vectors of Boolean polynomials.
    Uses the middle-school multiplication algorithm :
        1 - for each i, x is multiplied by a_i and shifted by i positions.
        2- Finally each (a_i * x) << i are summed.
    """
    assert len(x) == len(a)
    q = len(x)
    list_of_words = []
    for i in range(q):
        x_shifted_by_i = [0 for _ in range(i)] + [x[j] for j in range(q-i)]
        list_of_words.append([x * a[i] for x in x_shifted_by_i])
    return multiple_binary_sum(list_of_words)


def lwr_ring(word_size, n):
    """ Build a Boolean polynomial ring for LWR study with :
        - word_size*n binary variables named a
        - word_size*n binary variables named x.
    """
    names = [var_name + '%d_%d'%(i, j) for var_name in ['a', 'x'] for i in range(n) for j in range(word_size)]
    ring = BooleanPolynomialRing(2 * word_size * n, names=names)

    # The a-variables are the n*q first variables, the others are x-variables.
    vars_a = ring.gens()[:n * word_size]
    vars_x = ring.gens()[n * word_size:]

    return ring, vars_a, vars_x


def dot_product_anf(word_size, n, vars_a, vars_x):
    """ Given sets of variables (output by lwr_ring),
    builds the ANF of sum_{i = 1}^n a_ix_i where a_i, x_i are integers of size q. """
    # Matrices of variables: rows = round index, columns = var index
    round_by_round_a = [list(vars_a[i*word_size:(i+1)*word_size]) for i in range(n)]
    round_by_round_x = [list(vars_x[i*word_size:(i+1)*word_size]) for i in range(n)]

    # Recursive construction of the ANF of LWR
    ax = [binary_multiplication(round_by_round_a[i], round_by_round_x[i]) for i in range(n)]
    return multiple_binary_sum(ax)


#################################
# Monomial <-> Masks conversion #
#################################
def monomial_to_mask_aux(monom, n):
    """ Returns the mask corresponding to the Boolean monomial monom of at most n*q variables.
    In other words, if the monomial is monom = prod_{i} x_i^{u_i} or monom = prod_{i} a_i^{u_i}, returns the vector (u_1, ..., u_n).
    """
    # Works with monomial or string representation of a monomial
    if type(monom) != str:
        monom = str(monom)
    # Split the variables xi_j
    s = monom.split('*')

    masks = [0 for _ in range(n)]
    for var in s:
        # Compute (i, j) from xi_j
        var_index, var_bit = var[1:].split('_')
        var_index, var_bit = int(var_index), int(var_bit)
        # Put the j-th bit of the i-th integer to 1
        masks[var_index] |= (1 << var_bit)
    return masks


def monomial_to_mask(monom):
    """ Returns the masks corresponding to the Boolean monomial monom.
    In other words, if the monomial is monom = prod_{i =1}^{n} a_i^{u_i}x_j^{v_j}, returns the vector ((u_1, v_i), ..., (u_n, v_n)).
    If the monomial is monom = prod_{i=1}^n x_i^{u_i} or monom = prod_{i} a_i^{u_i}, returns the vector (u_1, ..., u_n).
    """
    # Works with monomial or string representation of a monomial
    if type(monom) != str:
        monom = str(monom)

    nb_of_vars = index_of_max_var(monom) + 1

    if monom.count('x') and monom.count('a'):
        # Find the position of the first x to call twice the function to_mask
        t = monom.find('x')
        return [(a, b) for a, b in zip(monomial_to_mask_aux(monom[:t-1], nb_of_vars), monomial_to_mask_aux(monom[t:], nb_of_vars))]
    else:
        return monomial_to_mask_aux(monom, nb_of_vars)


def index_of_max_var(monom):
    if type(monom) != str:
        monom = str(monom)
    max_index = monom.split('*')[-1].split('_')[0]  # Take the last term, xi_j or ai_j and recover i
    max_index = int(max_index[1:])
    return max_index


def monomial_to_canonical_mask(monom):
    """ Return the canonical masks corresponding to a monomial of at most n variables.
    In other words, return a sorted list of masks (or pair of masks) without 0 (or (0, 0))."""

    mask = monomial_to_mask(monom)
    if type(mask[0]) is tuple:
        return tuple(sorted([m for m in mask if m != (0, 0)], key=lambda tup: (tup[1],tup[0])))
    else:
        return tuple(sorted([m for m in mask if m != 0]))


def mask_to_monomial_str(mask, var='x'):
    """ Convert a mask corresponding to a Boolean monomial to a string describing the monomial
    or two strings if the monomial depends on a and x."""
    if type(mask[0]) == tuple:
        mask_a = [a for (a, x) in mask]
        mask_x = [x for (a, x) in mask]
        monom_a = mask_to_monomial_str(mask_a, var='a')
        monom_x = mask_to_monomial_str(mask_x, var='x')
        return monom_a, monom_x
    else:
        s = ''
        for ind_m, m in enumerate(mask):
            for ind_i, i in enumerate([int(c) for c in bin(m)[2:][::-1]]):
                if i:
                    if s:
                        s += '*'
                    s += var + str(ind_m) + '_' + str(ind_i)
        return s


def monom_to_latex(mask, var='x'):
    m = ""
    for i, v in enumerate(mask):
        # if m:
        #     m += "*"
        m += var+"^{%d}_{%d}"%(v, i)
    return m


def poly_to_latex(poly_masks, var='x'):
    p = ""
    for mask in poly_masks:
        if p:
            p += " + "
        p += monom_to_latex(mask, var)
    return p
