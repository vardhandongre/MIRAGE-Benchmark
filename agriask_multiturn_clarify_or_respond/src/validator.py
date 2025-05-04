def validate_decision(goal, decision, clarification_question, revealed_fact):
    if decision == "<Clarify>" and clarification_question:
        return "?" in clarification_question and len(clarification_question.split()) >= 5
    if decision == "<Respond>":
        return True
    return False