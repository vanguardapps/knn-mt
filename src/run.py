import os
import pathlib
import subprocess
import sys
from dotenv import load_dotenv


def main():
    load_dotenv()

    command = sys.argv[1]

    SHELL = os.environ["SHELL"]

    PGHOST = os.environ["PGHOST"]
    PGUSER = os.environ["PGUSER"]
    PGPORT = os.environ["PGPORT"]
    PGDATABASE = os.environ["PGDATABASE"]

    if command == "start-db":
        script_path = pathlib.Path("src", "scripts", "start-db.sh")
        subprocess.run(f"{SHELL} {script_path}", shell=True)
    elif command == "stop-db":
        subprocess.run("docker-compose down --remove-orphans", shell=True)
    elif command == "psql":
        subprocess.run(
            f"psql --host={PGHOST} --port={PGPORT} --username={PGUSER} --dbname={PGDATABASE}",
            shell=True,
        )


if __name__ == "__main__":
    main()