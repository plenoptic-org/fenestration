from pathlib import Path


def delete_directory_files(path: Path):
    for file in path.rglob("*"):
        if file.is_dir():
            delete_directory_files(file)
            file.rmdir()
        else:
            file.unlink()


delete_directory_files(Path("./api/generated"))
delete_directory_files(Path("./_build"))
