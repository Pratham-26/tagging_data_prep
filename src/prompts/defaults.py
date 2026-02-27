DEFAULT_INSTRUCTION = "Classify the text into one of the following labels."


def get_base_fields() -> list[dict]:
    """Return the base signature fields for DSPy serialization."""
    return [
        {"prefix": "Text:", "description": "${text}"},
        {"prefix": "Candidate Labels:", "description": "${candidate_labels}"},
        {"prefix": "Label Descriptions:", "description": "${label_descriptions}"},
        {"prefix": "Predicted Index:", "description": "${predicted_index}"},
    ]
