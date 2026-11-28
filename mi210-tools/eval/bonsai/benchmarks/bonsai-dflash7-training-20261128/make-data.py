from pathlib import Path
import json, subprocess

base = Path(__file__).resolve().parent


def extract(tag, folder):
    response = json.loads((folder / (tag + ".json")).read_text())
    rows = [response["warmup"], *response["requests"]]
    labels = base / (tag + ".tokens")
    labels.write_text("\n".join(" ".join(map(str, x["tokens"])) for x in rows) + "\n")
    subprocess.run(
        [
            str(base / "extract"),
            str(folder / (tag + ".bin")),
            str(labels),
            str(base / (tag + ".feat")),
        ],
        check=True,
    )


if __name__ == "__main__":
    for i in range(6):
        tag = f"teacher{i}-1"
        z = json.loads((base / (tag + ".json")).read_text())
        if len(z["requests"]) != 16:
            raise ValueError(tag + " incomplete")
        assert all(len(q["tokens"]) == 144 for q in z["requests"])
        extract(tag, base)
        print("extracted", tag, flush=True)
