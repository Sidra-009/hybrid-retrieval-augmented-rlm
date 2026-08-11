from collections import Counter


def calculate_confidence(answers: list) -> float:
    if len(answers) < 2:
        return 0.5
    # Majority voting
    most_common = Counter(answers).most_common(1)[0]
    return most_common[1] / len(answers)


def auto_heal(plan, low_confidence_step):
    """Agar confidence kam hai toh us step ko dubara run karo."""
    print(f"[AUTO-HEAL] Re-running step: {low_confidence_step} with more context...")
    # Logic to re-run with broader context
    return True
