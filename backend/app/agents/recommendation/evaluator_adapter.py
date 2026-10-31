def evaluation_to_score(summary):
    """
    Convert Evaluation Agent summary into
    a numerical score for Optuna.
    """

    if not isinstance(summary, dict):
        raise ValueError("Invalid evaluation summary")

    classes = summary.get("classes", [])

    if not classes:
        raise ValueError("No class evaluation results found")

    class_scores = []

    for cls in classes:
        pass_rate = cls.get("pass_rate")

        if pass_rate is None:
            raise ValueError(
                f"Missing pass rate for {cls.get('class_id')}"
            )

        if not 0 <= pass_rate <= 1:
            raise ValueError("Invalid pass rate")

        class_scores.append(pass_rate)

    # Equal weight for every defect class
    return sum(class_scores) / len(class_scores)