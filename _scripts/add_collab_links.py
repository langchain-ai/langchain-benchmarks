import argparse
import json
import os

from git import Git, Repo


def get_repo_base_url(directory: str) -> str:
    """Retrieves the base URL of the repository."""
    try:
        repo = Repo(directory, search_parent_directories=True)
        remote_url = repo.remotes.origin.url
        if remote_url.endswith(".git"):
            remote_url = remote_url[:-4]
        result = (
            remote_url.replace("git@", "https://").replace("https://github.com:", "")
            + "/blob/main/"
        )
        print(result)
        return result
    except Exception as e:
        print("Error retrieving repository URL:", e)
        return ""


def add_collab_link(cell_content: list, filepath: str, repo_base_url: str) -> list:
    """Inserts the 'Open In Collab' link into the cell content if it doesn't exist."""

    if repo_base_url:
        collab_link = f"[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{repo_base_url}{filepath})".replace(
            "/./", "/"
        )
        if collab_link not in "\n".join(cell_content):
            cell_content = cell_content[:1] + [collab_link+"\n"] + cell_content[1:]

    return cell_content


def process_directory(directory: str) -> None:
    """Traverses the directory and updates .ipynb files if necessary."""
    repo_base_url = get_repo_base_url(directory)
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".ipynb"):
                print("Checking", file)
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as ipynb_file:
                    ipynb_data = json.load(ipynb_file)

                try:
                    first_cell_content = ipynb_data["cells"][0]["source"]
                except Exception as e:
                    print("Skipping", filepath, e)
                    continue
                modified_content = add_collab_link(
                    first_cell_content, filepath, repo_base_url
                )
                if modified_content != first_cell_content:
                    print("Inserting link into", filepath)
                    ipynb_data["cells"][0]["source"] = modified_content

                    with open(filepath, "w", encoding="utf-8") as ipynb_file:
                        json.dump(ipynb_data, ipynb_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default=".")
    args = parser.parse_args()
    process_directory(args.directory)
