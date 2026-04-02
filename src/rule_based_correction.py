"""Rule-based BIM correction.

If predicted class belongs to wrong element, reassign to
highest-probability class within correct element.
"""
import numpy as np

ELEMENT_CLASSES = {"Deck": [0,1], "Pavement": [2,3], "Wall": [4,5]}

def rule_based_correct(pred_idx, element_type, softmax):
    """Correct single prediction."""
    valid = ELEMENT_CLASSES.get(element_type, list(range(len(softmax))))
    if pred_idx in valid: return pred_idx
    return max(valid, key=lambda i: softmax[i])

def correct_all(predictions, element_labels, softmax_outputs):
    """Correct all predictions."""
    return np.array([rule_based_correct(predictions[i], element_labels[i],
                     softmax_outputs[i]) for i in range(len(predictions))])
