"""Eval: paired bootstrap, таблицы, детектор утечки."""
import pytest

from backend.eval.bootstrap import paired_bootstrap
from backend.eval.leak_detector import detect_leaks
from backend.eval.tables import latex_table, markdown_table

# paired bootstrap

def test_bootstrap_identical_lists_not_significant():
    result = paired_bootstrap([7.0] * 50, [7.0] * 50, n_iterations=500)
    assert result["mean_diff"] == 0.0
    assert result["significant"] is False


def test_bootstrap_detects_real_difference():
    a = [8.0, 7.5, 8.5, 9.0, 8.2, 7.8, 8.4, 8.1, 8.6, 7.9] * 5  # режим A лучше
    b = [5.5, 6.0, 5.8, 6.2, 5.4, 5.9, 6.1, 5.6, 6.3, 5.7] * 5
    result = paired_bootstrap(a, b, n_iterations=500, seed=42)
    assert result["mean_diff"] > 2.0
    assert result["ci95"][0] > 0  # весь интервал выше нуля
    assert result["significant"] is True


def test_bootstrap_requires_equal_lengths():
    with pytest.raises(ValueError, match="одинаковое число"):
        paired_bootstrap([1.0, 2.0], [1.0])


def test_bootstrap_requires_nonempty():
    with pytest.raises(ValueError, match="Нет данных"):
        paired_bootstrap([], [])


# таблицы

def test_markdown_table_format():
    table = markdown_table(
        ["Режим", "Score", "Рост"],
        [["baseline", 5.1234, 0.0], ["full", 7.9876, 55.8]],
    )
    lines = table.splitlines()
    assert lines[0] == "| Режим | Score | Рост |"
    assert "| baseline | 5.12 | 0.00 |" in lines
    assert "| full | 7.99 | 55.80 |" in lines


def test_latex_table_escapes_specials():
    table = latex_table(["Режим", "Доля"], [["tm_hybrid", 93.5], ["100% & более", 1.0]])
    assert r"\toprule" in table and r"\bottomrule" in table
    assert r"100\% \& более" in table


# детектор утечки

def test_leak_detector_catches_exact_dup():
    leaks = detect_leaks(["Отток 8%, цель 5%"], ["Отток 8%, цель 5%"])
    assert len(leaks) == 1 and leaks[0]["similarity"] == 1.0


def test_leak_detector_catches_near_dup():
    leaks = detect_leaks(
        ["Отток клиентов вырос до 8 процентов за месяц"],
        ["Отток клиентов вырос до 8 процентов за 2 месяца"],
        threshold=0.85,
    )
    assert len(leaks) == 1 and leaks[0]["similarity"] >= 0.85


def test_leak_detector_ignores_different_situations():
    leaks = detect_leaks(
        ["Отток клиентов вырос до 8 процентов"],
        ["Нужно выбрать две инициативы из бэклога на квартал"],
    )
    assert leaks == []
