"""Таблицы результатов: Markdown (для отчёта/README) и LaTeX (booktabs)."""


def markdown_table(headers: list[str], rows: list[list]) -> str:
    """Markdown-таблица; числа форматируются с 2 знаками, остальное - как есть."""
    def fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def latex_table(headers: list[str], rows: list[list], caption: str = "") -> str:
    """LaTeX-таблица в стиле booktabs (toprule/midrule/bottomrule)."""
    def fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.2f}"
        text = str(value)
        return text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")

    col_spec = "l" + "c" * (len(headers) - 1) if len(headers) > 1 else "l"
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(fmt(h) for h in headers) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(fmt(v) for v in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)
