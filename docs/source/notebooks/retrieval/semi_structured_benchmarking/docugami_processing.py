import os
import tempfile
from pathlib import Path
from time import sleep
from typing import Dict, List

import requests
from dgml_utils.segmentation import get_chunks_str
from docugami import Docugami
from docugami.types import Document as DocugamiDocument

api_key = "cukRRrDv1lwn2BnM1jiiUmjO+ptFxvg8tfmqpFRkrzlXXUtSWvTulr0ZzY5uCsQHp3doM2ZC3gcOUw+/TbzGB3FaqEcDujPRvgxy1trlhi884QAJ9GL8/AROM2KL6qYdLaelqD5IrGTteqFS9kVPhBaOxkHKYFTTkkjEkU2MOBp03JQAcfs1rt+e3t4mjz1Uf6VeBspRruvr85Jv4OMvqdsZO8qfhexyIcNeIr0Hy7jAc9dCxkvX1mKRxYf+4kPjxD6F9Pdkmf/QuQS5gc5lKKNodehADGL0hPYpbJQF+m+wnUTQYPPwb1YOQecxRx2YqwBjxY0LTyMLMWCjQggggA=="
os.environ["DOCUGAMI_API_KEY"] = api_key

client = Docugami()


def upload_files(local_paths: List[str], docset_name: str) -> List[DocugamiDocument]:
    # Docs
    docset_list_response = client.docsets.list(name=docset_name)
    if docset_list_response and docset_list_response.docsets:
        # Docset already exists with this name
        docset_id = docset_list_response.docsets[0]
    else:
        dg_docset = client.docsets.create(name=docset_name)
        docset_id = dg_docset.id

    document_list_response = client.documents.list(limit=int(1e5))
    dg_docs: List[DocugamiDocument] = []
    if document_list_response and document_list_response.documents:
        new_names = [Path(f).name for f in local_paths]

        dg_docs = [
            d
            for d in document_list_response.documents
            if Path(d.name).name in new_names
        ]
        existing_names = [Path(d.name).name for d in dg_docs]

        # Upload any files not previously uploaded
        for f in local_paths:
            if Path(f).name not in existing_names:
                dg_docs.append(
                    client.documents.contents.upload(
                        file=Path(f).absolute(),
                        docset_id=docset_id,
                    )
                )
    return dg_docs


def wait_for_xml(dg_docs: List[DocugamiDocument]) -> dict[str, str]:
    dgml_paths: dict[str, str] = {}
    while len(dgml_paths) < len(dg_docs):
        for doc in dg_docs:
            doc = client.documents.retrieve(doc.id)  # update with latest
            current_status = doc.status
            if current_status == "Error":
                raise Exception(
                    "Document could not be processed, please confirm it is not a zero length, corrupt or password protected file"
                )
            elif current_status == "Ready":
                dgml_url = doc.docset.url + f"/documents/{doc.id}/dgml"
                headers = {"Authorization": f"Bearer {api_key}"}
                dgml_response = requests.get(dgml_url, headers=headers)
                if not dgml_response.ok:
                    raise Exception(
                        f"Could not download DGML artifact {dgml_url}: {dgml_response.status_code}"
                    )
                dgml_contents = dgml_response.text
                with tempfile.NamedTemporaryFile(delete=False, mode="w") as temp_file:
                    temp_file.write(dgml_contents)
                    temp_file_path = temp_file.name
                    dgml_paths[doc.name] = temp_file_path

        print(f"{len(dgml_paths)} docs done processing out of {len(dg_docs)}...")

        if len(dgml_paths) == len(dg_docs):
            # done
            return dgml_paths
        else:
            sleep(30)
