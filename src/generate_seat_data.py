import json
import os
import re

DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEAT_DATA = os.path.join(DIRECTORY, "data", "weat", "hi", "lang_spec")
SEAT_DATA = os.path.join(DIRECTORY, "data", "seat", "hi", "lang_spec")
SEAT_TEMPLATES = os.path.join(DIRECTORY, "data", "seat", "hi", "templates.jsonl")
TEST_EXT = ".jsonl"


def _test_sort_key(test):
    """Return tuple to be used as a sort key for the specified test name.
    Break test name into pieces consisting of the integers in the name
    and the strings in between them.
    """
    key = ()
    prev_end = 0
    for match in re.finditer(r"\d+", test):
        key = key + (test[prev_end : match.start()], int(match.group(0)))
        prev_end = match.end()
    key = key + (test[prev_end:],)

    return key


def _load_json(sent_file):
    """Load from json. We expect a certain format later, so do some post processing."""
    print(f"Loading {sent_file}...")
    all_data = json.load(open(sent_file, "r"))
    data = {}
    for k, v in all_data.items():
        examples = v["examples"]
        data[k] = examples
        v["examples"] = examples

    return all_data


def generate_seat_data():
    all_tests = sorted(
        [
            entry[: -len(TEST_EXT)]
            for entry in os.listdir(WEAT_DATA)
            if not entry.startswith(".") and entry.endswith(TEST_EXT)
        ],
        key=_test_sort_key,
    )

    SEAT_templates = json.load(open(SEAT_TEMPLATES, "r"))

    for i, test in enumerate(all_tests):
        encs = _load_json(os.path.join(WEAT_DATA, f"{test}{TEST_EXT}"))
        for item in encs.keys():
            if encs[item]["type"] in SEAT_templates.keys():
                encs[item]["templates"] = SEAT_templates[encs[item]["type"]]
                encs[item]["sentences"] = []
                for template in encs[item]["templates"]:
                    for example in encs[item]["examples"]:
                        encs[item]["sentences"].append(template.replace("_", example))

        with open(os.path.join(SEAT_DATA, f"sent-{test}{TEST_EXT}"), "w") as f:
            json.dump(encs, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    generate_seat_data()
