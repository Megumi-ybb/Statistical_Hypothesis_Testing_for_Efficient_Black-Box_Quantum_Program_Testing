from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision" / "analysis_out" / "original_tables"

PRINTED = {
    ("table5_marginal_typei_by_d", 2, "auto"): 0.056,
    ("table5_marginal_typei_by_d", 10, "tolerance"): 0.992,
    ("table7_joint_typei_by_d", 10, "chi_square"): 0.042,
    ("table7_joint_typei_by_d", 6, "mmdagg"): 0.027,
    ("table8_joint_power_by_m", 50, ("power_bitflip", "mmdagg")): 0.868,
    ("table8_joint_power_by_m", 100, ("power_depolarize", "chi_square")): 0.819,
    ("table9_joint_bitflip_power_by_d_m50", 10, "chi_square"): 0.016,
    ("table9_joint_bitflip_power_by_d_m200", 10, "chi_square"): 0.574,
}


def med(df: pd.DataFrame, index: str, values) -> pd.DataFrame:
    return df.pivot_table(index=index, columns="test_method", values=values, aggfunc="median").round(3)


def main() -> None:
    m = pd.read_csv(ROOT / "Marginal" / "combined_results.csv")
    j = pd.read_csv(ROOT / "Joint" / "combined_results.csv")
    mb, jb, jf = m[m.case == "baseline"], j[j.case == "baseline"], j[j.case == "falsecase"]
    tables = {
        "table5_marginal_typei_by_d": med(mb, "n_qubits", "type_i_error"),
        "table6_marginal_typei_by_m": med(mb, "m_samples", "type_i_error"),
        "table7_joint_typei_by_d": med(jb, "n_qubits", "type_i_error"),
        "table8_joint_power_by_m": med(jf, "m_samples", ["power_bitflip", "power_asymmetric", "power_depolarize"]),
    }
    for size in (50, 200, 500):
        tables[f"table9_joint_bitflip_power_by_d_m{size}"] = med(jf[jf.m_samples == size], "n_qubits", "power_bitflip")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv")
        print(f"== {name}\n{table.to_string()}\n")
    for (name, row, col), value in PRINTED.items():
        got = tables[name].loc[row, col]
        assert abs(got - value) < 5e-4, f"{name}[{row}, {col}] = {got}, article prints {value}"
    print(f"check: {len(PRINTED)} values printed in the article reproduce")


if __name__ == "__main__":
    main()
