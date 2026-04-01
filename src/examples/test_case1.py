import random


def generate_complex_test_data(filename, total_chars=25000):
    # 模拟包含条件逻辑的文本片段
    logic_blocks = [
        "当指标 {indicator} > {threshold} 时，使用方案 {plan_a}，剂量为 {dose_a}；",
        "当指标 {indicator} 处于 {range} 时，切换至方案 {plan_b}，并增加 {drug_c}；",
        "在 {season} 种植 {variety} 大豆，需注意 {weather} 导致的逻辑分支变化。"
    ]

    content = "知识体系原始测试数据 - 自动化生成\n"
    while len(content) < total_chars:
        block = random.choice(logic_blocks).format(
            indicator=random.choice(["血糖", "血压", "湿度", "土壤温度"]),
            threshold=random.randint(10, 100),
            plan_a="方案-X", dose_a="50mg",
            range="30-50", plan_b="方案-Y", drug_c="增效剂-Z",
            season="春季", variety="黑农48", weather="连续强降雨"
        )
        content += block + " " * random.randint(1, 10)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"已生成包含逻辑分支的超长数据集: {filename}, 总长度: {len(content)}")


generate_complex_test_data("branch_logic_test.txt")