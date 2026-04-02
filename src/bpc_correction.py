"""Bayesian Post-Correction (Eq. 1-3).

P(d|x,e) = P(x|d) * P(d|e) / P(x|e)
where P(x|d) ~ softmax, P(d|e) = element prior.
"""
import numpy as np

ELEMENT_TO_CLASSES = {"Deck": [0,1], "Pavement": [2,3], "Wall": [4,5]}

def get_element_prior(element_type, prior_type="element_constrained", n_classes=6, alpha=0.0):
    """Generate element-specific prior P(d|e)."""
    uniform = np.ones(n_classes) / n_classes
    if prior_type == "uniform": return uniform
    prior = np.zeros(n_classes)
    for i in ELEMENT_TO_CLASSES.get(element_type, []): prior[i] = 1.0
    s = prior.sum()
    prior = prior / s if s > 0 else uniform
    return (1.0 - alpha) * prior + alpha * uniform

def bayesian_post_correction(softmax, prior):
    """Apply Bayes rule: posterior = softmax * prior, normalized."""
    posterior = softmax * prior
    total = posterior.sum()
    return posterior / total if total > 1e-12 else softmax.copy()

def correct_predictions(softmax_outputs, element_labels, prior_type="element_constrained", alpha=0.0):
    """Correct all predictions using BIM element context."""
    n = len(softmax_outputs)
    corrected = np.empty(n, dtype=int)
    for i in range(n):
        prior = get_element_prior(element_labels[i], prior_type, softmax_outputs.shape[1], alpha)
        posterior = bayesian_post_correction(softmax_outputs[i], prior)
        corrected[i] = np.argmax(posterior)
    return corrected
