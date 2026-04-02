"""Utility functions for BIM-BPC experiments."""
import numpy as np

CLASS_NAMES = ["Deck_crack", "Deck_no_crack", "Pavement_crack",
               "Pavement_no_crack", "Wall_crack", "Wall_no_crack"]

def get_element(class_name):
    """Return element type from class name."""
    return class_name.split("_")[0]

def count_cross_element_errors(y_true, y_pred, class_names=None):
    """Count predictions where predicted element != true element."""
    if class_names is None: class_names = CLASS_NAMES
    return sum(1 for t, p in zip(y_true, y_pred)
               if get_element(class_names[t]) != get_element(class_names[p]))
