"""Create shared PyCharm launchers using the project's existing SSH interpreter."""
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT.parent / ".run"
    template = ET.parse(destination / "Server_Train_A.run.xml")
    recipes = [
        ("Paper_0_Plan", "Paper 0 - Preview full experiment plan", "run_paper_experiments.py", "--stage all"),
        ("Paper_1_Repeats", "Paper 1 - Repeat A B seeds 1 2", "run_paper_experiments.py", "--stage repeats --seeds 0 1 2 --execute"),
        ("Paper_Recovery_Repeats", "Paper Recovery - Restart incomplete repeats", "run_paper_experiments.py", "--stage repeats --seeds 0 1 2 --restart-incomplete --execute"),
        ("Paper_2_Baselines", "Paper 2 - MCNN CSRNet A B seed0", "run_paper_experiments.py", "--stage baselines --seeds 0 --execute"),
        ("Paper_Recovery_Baselines", "Paper Recovery - Restart incomplete baselines", "run_paper_experiments.py", "--stage baselines --seeds 0 --restart-incomplete --execute"),
        ("Paper_3_Ablations", "Paper 3 - Architecture loss A B seed0", "run_paper_experiments.py", "--stage ablations --seeds 0 --execute"),
        ("Paper_4_Speed", "Paper 4 - Server speed comparison", "run_paper_experiments.py", "--stage speed --execute"),
        ("Paper_5_Qualitative", "Paper 5 - A B qualitative cases", "run_paper_experiments.py", "--stage qualitative --execute"),
        ("Paper_6_Report", "Paper 6 - Audit mean std report", "run_paper_experiments.py", "--stage report --seeds 0 1 2 --execute"),
        ("Paper_7_All", "Paper 7 - Complete all seeds baselines ablations", "run_paper_experiments.py", "--stage all --seeds 0 1 2 --execute"),
        ("Paper_Smoke", "Paper Check - Synthetic pipeline", "check_paper_pipeline.py", "--device 0"),
    ]
    for filename, name, script, parameters in recipes:
        import copy
        tree = copy.deepcopy(template)
        configuration = tree.getroot().find("configuration")
        configuration.set("name", name)
        for option in configuration.findall("option"):
            if option.get("name") == "SCRIPT_NAME":
                option.set("value", f"$PROJECT_DIR$/dc-yolo26-crowd/scripts/{script}")
            if option.get("name") == "PARAMETERS":
                option.set("value", parameters)
        ET.indent(tree, space="  ")
        tree.write(destination / f"{filename}.run.xml", encoding="unicode")
    print(f"Created {len(recipes)} shared launchers: {destination}")


if __name__ == "__main__":
    main()
