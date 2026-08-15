import random


def sample_questions(bank, n, seed):
    rng = random.Random(seed)
    return rng.sample(bank, min(n, len(bank)))


def administer(model, questions, context: str = "") -> dict:
    answers, yes_vas = [], []
    for q in questions:
        prompt = f"{context}Question: {q['text']}\nAnswer yes or no.\nAnswer:"
        y, n = model.yes_no_logprobs(prompt)
        is_yes = y > n
        answers.append({"id": q["id"], "yes": bool(is_yes)})
        if is_yes:
            yes_vas.append((q["v"], q["a"]))
    if yes_vas:
        va = [sum(v for v, _ in yes_vas) / len(yes_vas),
              sum(a for _, a in yes_vas) / len(yes_vas)]
    else:
        va = [0.5, 0.5]
    return {"va": va, "n_yes": len(yes_vas), "answers": answers}
