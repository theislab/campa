#!python
from pathlib import Path
from configparser import ConfigParser
import os
import shutil


def prepare_config(args):
    def get_yn_input():
        r = input()
        while r.lower() not in ("y", "n"):
            r = input("Unknown value. Either input y or n:\n")
        if r.lower() == "y":
            return True
        return False

    def get_path_input(name):
        p = input(f"Please enter the full path to {name}:\n")
        p = os.path.abspath(p)
        if not os.path.isdir(p):
            print(f"{p} is not a valid directory")
            return get_path_input()
        return p

    def change_path(config, name, section="DEFAULT"):
        print(f"The current {name} is {config.get(section, name)}. Would you like to change it? (y/n)")
        if not get_yn_input():
            print(f"Leaving {name} unchanged.")
        else:
            new_path = get_path_input(name)
            config.set(section, name, new_path)
            with open(config_path, "w") as configfile:
                config.write(configfile)
            print(f"Successfully changed {name} to {new_path}")

    config_path = Path.home() / ".config" / "campa" / "campa.ini"
    # read config file from scripts_dir (parent dir of dir that this file is in)
    example_config_path = Path(__file__).resolve().parent.parent / "campa.ini.example"
    print(example_config_path)
    # check if custom config exists
    if not config_path.is_file() or args.force:
        print(f"No campa.ini found in {config_path}. Creating default config file.")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(example_config_path, config_path)

    else:
        print(f"Found existing campa.ini in {config_path}")

    # check test data
    config = ConfigParser()
    config.read(config_path)
    cur_test_data = config.get("data", "TestData")
    if cur_test_data == "" or not Path(cur_test_data).is_file():
        print("setting up TestData config")
        # add test data
        config.set(
            "data",
            "TestData",
            str(Path(__file__).resolve().parent.parent / "notebooks" / "params" / "TestData_constants.py"),
        )
        with open(config_path, "w") as configfile:
            config.write(configfile)

    print("Would you like to configure your campa.ini config now? (y/n)")
    if get_yn_input():
        # read config file
        config = ConfigParser()
        config.read(config_path)
        change_path(config, "experiment_dir")
        change_path(config, "data_dir")
    else:
        print("Exiting without setting up campa.ini")

    # print currently registered data config files
    config = ConfigParser()
    config.read(config_path)
    print(f"Currently registered data configs in {config_path}:")
    for key, val in config.items("data"):
        if key not in ["experiment_dir", "data_dir"]:
            print(f"\t{key}: {val}")
    print(f"To change or add data configs, please edit {config_path} directly.")
