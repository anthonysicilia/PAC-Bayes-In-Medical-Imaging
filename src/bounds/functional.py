from math import log, sqrt, log2, ceil
import torch

def hoeffding_bound(mc_est, m, delta):
    """
    Hoeffding's Bound
    """
    return mc_est + sqrt(1 / 2 * log(2 / delta) / m)

def mauer_term(kl_div, m, delta):
    """
    A Note on the PAC-Bayesian Theorem
    The term in the upperbound of Theorem 5
    """
    return (kl_div + log((2 * sqrt(m)) / delta)) / m

def langford_caruana_term(m, delta):
    """
    (Not) Bounding the True Error
    The term in the upperbound of Theorem 2.5
    """
    return log(2 / delta) / m

def dziugaite_pinsker_bound(sample_loss, _mauer_term):
    """
    On the Role of Data in PAC Bayes Bounds
    Appendix Theorem D.2

    notes: use torch.sqrt to preserve kl gradients
    """
    return sample_loss + torch.sqrt(_mauer_term / 2)

def dziugaite_moment_bound(sample_loss, _mauer_term):
    """
    On the Role of Data in PAC Bayes Bounds
    Appendix Theorem D.2

    notes: use torch.sqrt to preserve gradients
    """
    third_term = torch.sqrt(_mauer_term * (_mauer_term + 2 * sample_loss))
    return sample_loss + _mauer_term + third_term

def dziugaite_variational_bound(sample_loss, kl_div, m, delta):
    """
    On the Role of Data in PAC Bayes Bounds
    Appendix Theorem D.2
    """
    _mauer_term = mauer_term(kl_div, m, delta)
    a = dziugaite_pinsker_bound(sample_loss, _mauer_term)
    b = dziugaite_moment_bound(sample_loss, _mauer_term)
    # min preserves gradients
    return min(a, b)

# Removing... It should never be tighter than the mauer bound
# def pac_bayes_dziugaite_variational_bound(sample_loss, m, n, delta, kl_div):
#     """
#     On the Role of Data in PAC Bayes Bounds
#     Appendix Theorem D.2
#     + application of Langford and Caruana's Result
#     Uses inverted KL diverence
    
#     param notes:
#     m is the number data samples
#     n is the number of hypothesis samples
#     """
#     delta = delta / 2
#     mc_sample_loss = inv_kl(sample_loss, langford_caruana_term(n, delta))
#     return dziugaite_variational_bound(mc_sample_loss, kl_div, m, delta)

def pac_bayes_hoeffding_bound(sample_loss, m, n, delta, *args):
    """
    Hoeffding bound
    + application of Langford and Caruana's Result
    + Pinsker's Inequality

    Technically just two applications of Hoeffding's bound
    param notes:
    m is the number data samples
    n is the number of hypothesis samples
    """
    delta = delta / 2
    mc_sample_loss = hoeffding_bound(sample_loss, n, delta)
    return hoeffding_bound(mc_sample_loss, m, delta)

# Is this valid? The LC bound may only be applicable for hypotheses
# def iterated_langford_and_caruana_bound(sample_loss, m, n, delta, *args):
#     """
#     (Not) Bounding the True Error
#     Theorem 2.5 

#     Uses inverted KL divergence
#     Iterated applications (both hypotheses and data samples)

#     Technically just two applications of Hoeffding's bound
#     param notes:
#     m is the number data samples
#     n is the number of hypothesis samples
#     """
#     delta = delta / 2
#     mc_sample_loss = inv_kl(sample_loss, langford_caruana_term(n, delta))
#     return inv_kl(mc_sample_loss, langford_caruana_term(m, delta))

def mauer_bound(sample_loss, m, n, delta, kl_div):
    """
    A Note on the PAC-Bayesian Theorem
    Theorem 5
    +
    (Not) Bounding the True Error
    Theorem 2.5

    Uses inverted KL divergence

    param notes:
    m is the number data samples
    n is the number of hypothesis samples
    """
    delta = delta / 2
    mc_sample_loss = inv_kl(sample_loss, langford_caruana_term(n, delta))
    return inv_kl(mc_sample_loss, mauer_term(kl_div, m, delta))

def rivasplata_fquad_bound(sample_loss, kl_div, m, delta):
    """
    PAC-Bayes with Backprop
    Theorem 1
    and
    Tighter risk certificates for neural networks
    Theorem 1

    Adapted from the github repo:
    https://github.com/mperezortiz/PBB

    param notes:
    m is the number data samples

    other notes:
    use torch.sqrt to preserve gradients
    """
    a = kl_div + log((2 * sqrt(m)) / delta)
    b = 2 * m
    repeated_kl_ratio = a / b
    first_term = torch.sqrt(sample_loss + repeated_kl_ratio)
    second_term = torch.sqrt(repeated_kl_ratio)
    return (first_term + second_term) ** 2

def mauer_bound_ortiz_impl(sample_loss, m, n, delta, kl_div):
    """
    Adapted from the following github repo:
    https://github.com/mperezortiz/PBB

    Used in this repo for testing purposes.

    param notes:
    m is the number data samples
    n is the number of hypothesis samples
    """
    delta = delta / 2
    mc_sample_loss = inv_kl_ortiz_impl(sample_loss, log(2 / delta) / n)
    return inv_kl_ortiz_impl(mc_sample_loss, (kl_div + log((2 * sqrt(m)) / delta)) / m)

def inv_kl_ortiz_impl(qs, ks):
    """
    A numerical approximation
    of inverse KL divergence by bisection
    method.

    Adapted from the following git repo:
    https://github.com/mperezortiz/PBB

    Used in this repo for testing purposes.

    Inversion of the binary kl
    Parameters
    ----------
    qs : float
        Empirical risk
    ks : float
        second term for the binary kl inversion
    """
    # computation of the inversion of the binary KL
    qd = 0
    ikl = 0
    izq = qs
    dch = 1-1e-10
    while((dch-izq)/dch >= 1e-5):
        p = (izq+dch)*.5
        if qs == 0:
            ikl = ks-(0+(1-qs)*log((1-qs)/(1-p)))
        elif qs == 1:
            ikl = ks-(qs*log(qs/p)+0)
        else:
            ikl = ks-(qs*log(qs/p)+(1-qs) * log((1-qs)/(1-p)))
        if ikl < 0:
            dch = p
        else:
            izq = p
        qd = p
    return qd

def binary_kl(q, p):
    """
    Binary KL divergence KL(q || p)
    """
    assert p != 1 and p != 0
    return q * log(q / p) + (1 - q) * log((1 - q) / (1 - p))

def inv_kl(q, epsilon, tolerance=1e-7):
    """
    Compute the inverse KL divergence 
    by way of the bisection method:
    https://en.wikipedia.org/wiki/Bisection_method

    Specifcally, we find the root of the below
    f(p) = epsilon - KL(q || p)
    f(q) = epsilon
    f(1) = -infinity; i.e., should have KL(q || 1 - 1e-8) > epsilon
    """
    a = q
    b = 1 - tolerance

    f = lambda x: epsilon - binary_kl(q, x)

    sign = lambda x: x > 0

    if not a < b:
        raise AssertionError('Condition a < b violated for bisection method.')
    if not f(a) > 0:
        raise AssertionError('Condition f(a) > 0 violated for bisection method.')
    if not f(b) < 0:
        print('Condition f(b) < 0 violated for bisection method.')
        print('Implies root 1 >= p > 1 - tolerance. So, returning 1.')
        return 1.0

    max_its = ceil(log2(abs(b - a) / tolerance))
    i = 0

    while i <= max_its:

        p = (a + b) / 2

        fp = f(p)

        if fp == 0 or abs(b - a) / 2 < tolerance:
            return p

        if sign(f(p)) == sign(f(a)):
            a = p
        else:
            b = p

        i += 1

    return p
