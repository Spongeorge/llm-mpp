import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import Image
import hashlib
from tqdm import tqdm
import os
import json
import random
import time
from pathlib import Path
import pandas as pd

from datasets import load_dataset

import deepsmiles
import selfies
from pubchempy import get_compounds

import httpx
import google.auth
from google.auth.transport.requests import Request
import os

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(__file__).resolve().parent / 'data/'
OUTPUT_ROOT = Path(__file__).resolve().parent / 'output/'

BBBP_PROMPT = "Determine whether the following molecule is likely to penetrate the blood brain barrier. First provide reasoning, and then a yes or no decision in the form \"Decision: Yes/No\". Molecule: {molecule_string}"
BACE_PROMPT = "Determine whether the following molecule is likely to inhibit the Human beta-secretase 1 enzyme. First provide reasoning, and then a yes or no decision in the form \"Decision: Yes/No\". Molecule: {molecule_string}"
ESOL_PROMPT = "Predict the log water solubility in mols per litre. First provide reasoning, and then a numeric value in the form \"Decision: X\". Molecule: {molecule_string}"
TOX_PROMPT = "Determine whether the following molecule is likely to be toxic to humans. First provide reasoning, and then a yes or no decision in the form \"Decision: Yes/No\". Molecule: {molecule_string}"
HFE_PROMPT = "Predict the hydration free energy in kcal/mol of the following molecule. First provide reasoning, and then a numeric value in the form \"Decision: X\". Molecule: {molecule_string}"

TASK_TO_PROMPT = {
    "bbbp": BBBP_PROMPT,
    "bace": BACE_PROMPT,
    "esol": ESOL_PROMPT,
    "freesolv": HFE_PROMPT,
    "clintox": TOX_PROMPT
}

TASK_TO_LABEL_COLNAME = {
    "bbbp": "p_np",
    "bace": "Class",
    "esol": "measured log solubility in mols per litre",
    "freesolv": "expt",
    "clintox": "CT_TOX"
}

TASK_TO_TYPE = {
    "bbbp": "binary classification",
    "bace": "binary classification",
    "esol": "regression",
    "freesolv": "regression",
    "clintox": "binary classification"
}

REP_COLNAMES = ["smiles", "deepsmiles", "selfies", "inchi", "iupac"]


def get_credentials():
    credentials, project_id = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())
    return credentials.token


def build_endpoint_url(
        region: str,
        project_id: str,
        model_name: str,
        model_version: str,
        streaming: bool = False,
):
    base_url = f"https://{region}-aiplatform.googleapis.com/v1/"
    project_fragment = f"projects/{project_id}"
    location_fragment = f"locations/{region}"
    specifier = "streamRawPredict" if streaming else "rawPredict"
    model_fragment = f"publishers/mistralai/models/{model_name}@{model_version}"
    url = f"{base_url}{'/'.join([project_fragment, location_fragment, model_fragment])}:{specifier}"
    return url


def retry_with_exponential_backoff(
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 16,
):
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"ERROR, RETRYING. {e}")
                num_retries += 1

                if num_retries > max_retries:
                    return "MAXIMUM RETRIES ERROR"

                delay *= exponential_base * (1 + jitter * random.random())

                time.sleep(delay)

    return wrapper


print_first_n = 5


@retry_with_exponential_backoff
def completion_with_backoff(model, template: str, molecule: str, fewshot_exemplars=None):
    if fewshot_exemplars is not None:
        prompt = template.split('.')[0] + '.\n' + '\n'.join(
            [f'Molecule: {example[0]}\nDecision: {example[1]}' for example in
             fewshot_exemplars]) + '\n' + template.format(
            molecule_string=molecule)
    else:
        prompt = template.format(molecule_string=molecule)

    global print_first_n
    if print_first_n > 0:
        print(prompt)
        print()
        print_first_n -= 1

    response = model.generate_content(prompt, generation_config={"temperature": 0}).text

    return response


@retry_with_exponential_backoff
def completion_with_backoff_direct(model_name, template, molecule, url, headers, fewshot_exemplars=None):
    if fewshot_exemplars is not None:
        prompt = template.split('.')[0] + '.\n' + '\n'.join(
            [f'Molecule: {example[0]}\nDecision: {example[1]}' for example in
             fewshot_exemplars]) + '\n' + template.format(
            molecule_string=molecule)
    else:
        prompt = template.format(molecule_string=molecule)

    global print_first_n
    if print_first_n > 0:
        print(prompt)
        print()
        print_first_n -= 1

    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "stream": False
    }

    # Make the call
    with httpx.Client() as client:
        resp = client.post(url, json=data, headers=headers, timeout=None)

        try:
            response = resp.json()["choices"][0]["message"]["content"]
        except:
            time.sleep(5)
            headers = {
                "Authorization": f"Bearer {get_credentials()}",
                "Accept": "application/json",
            }
            return completion_with_backoff_direct(model_name, template, molecule, url, headers, fewshot_exemplars)

    return response


def smiles2deepsmiles(smiles):
    converter = deepsmiles.Converter(rings=True, branches=True)
    return converter.encode(smiles)


def smiles2selfies(smiles):
    return selfies.encoder(smiles.strip(), strict=False)


def smiles2inchi(smiles):
    compounds = get_compounds(smiles, namespace='smiles')
    match = compounds[0]
    return match.inchi


def smiles2iupac(smiles):
    compounds = get_compounds(smiles, namespace='smiles')
    match = compounds[0]
    return match.iupac_name


def smiles2formula(smiles):
    compounds = get_compounds(smiles, namespace='smiles')
    match = compounds[0]
    return match.molecular_formula


def prepare_datasets():
    dfs_to_retrieve = []

    if os.path.exists(DATA_ROOT / "bbbp_test_fs_final.pkl"):
        df_bbbp = pd.read_pickle(DATA_ROOT / "bbbp_test_fs_final.pkl")
    else:
        data_bbbp = load_dataset("katielink/moleculenet-benchmark", 'bbbp')['test']
        df_bbbp = data_bbbp.to_pandas()
        dfs_to_retrieve.append(("bbbp", df_bbbp))

    if os.path.exists(DATA_ROOT / "bace_test_fs_final.pkl"):
        df_bace = pd.read_pickle(DATA_ROOT / "bace_test_fs_final.pkl")
    else:
        data_bace = load_dataset("katielink/moleculenet-benchmark", 'bace')['test']
        df_bace = data_bace.to_pandas()
        dfs_to_retrieve.append(("bace", df_bace))

    if os.path.exists(DATA_ROOT / "esol_test_fs_final.pkl"):
        df_esol = pd.read_pickle(DATA_ROOT / "esol_test_fs_final.pkl")
    else:
        data_esol = load_dataset("katielink/moleculenet-benchmark", 'esol')['test']
        df_esol = data_esol.to_pandas()

        dfs_to_retrieve.append(("esol", df_esol))

    if os.path.exists(DATA_ROOT / "freesolv_test_fs_final.pkl"):
        df_freesolv = pd.read_pickle(DATA_ROOT / "freesolv_test_fs_final.pkl")
    else:
        data_freesolv = load_dataset("katielink/moleculenet-benchmark", 'freesolv')['test']
        df_freesolv = data_freesolv.to_pandas()

        dfs_to_retrieve.append(("freesolv", df_freesolv))

    if os.path.exists(DATA_ROOT / "clintox_test_fs_final.pkl"):
        df_clintox = pd.read_pickle(DATA_ROOT / "clintox_test_fs_final.pkl")
    else:
        data_clintox = load_dataset("katielink/moleculenet-benchmark", 'clintox')['test']
        df_clintox = data_clintox.to_pandas()

        dfs_to_retrieve.append(("clintox", df_clintox))

    for name, df in tqdm(dfs_to_retrieve):
        df['deepsmiles'] = df['smiles'].apply(smiles2deepsmiles)
        df['iupac'] = df['smiles'].apply(smiles2iupac)
        df['inchi'] = df['smiles'].apply(smiles2inchi)
        df['selfies'] = df['smiles'].apply(smiles2selfies)

        df.to_pickle(f"C:/Users/Georgio/PycharmProjects/string-mpp/{name}.pkl")

    return df_bbbp, df_bace, df_esol, df_freesolv, df_clintox


def build_fewshot_examples(df, k, task: str, representation: str, multi_rep: bool = False):
    n = len(df['fewshot'][int(k.split("_")[-1])])

    if multi_rep:
        molecule_strings = [" ".join([str(df['fewshot'][int(k.split("_")[-1])][j][1][rep]) for rep in REP_COLNAMES]) for j in range(n)]
    else:
        molecule_strings = [df['fewshot'][int(k.split("_")[-1])][j][1][representation] for j in range(n)]

    if TASK_TO_TYPE[task] == "binary classification":
        labels = ["Yes" if df['fewshot'][int(k.split("_")[-1])][j][1][TASK_TO_LABEL_COLNAME[task]] else "No" for j in
                  range(n)]
    else:
        labels = [df['fewshot'][int(k.split("_")[-1])][j][1][TASK_TO_LABEL_COLNAME[task]] for j in range(n)]

    return zip(molecule_strings, labels)


def main(config_path: str = PROJECT_ROOT / 'config.json'):
    vertexai.init(project="malamute-llama", location="us-central1",
                  api_endpoint="us-central1-aiplatform.googleapis.com", service_account="")

    with open(config_path, 'r') as f:
        config = json.load(f)

    model_name = config['model_name']

    if model_name in ["publishers/mistralai/models/mistral-large"]:
        mode = "api_direct"
        project_id = "llm-mpp"
        region = "us-central1"
        access_token = get_credentials()

        model = "mistral-large"
        model_version = "2407"
        is_streamed = False

        url = build_endpoint_url(
            project_id=project_id,
            region=region,
            model_name=model,
            model_version=model_version,
            streaming=is_streamed
        )

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }
    else:
        mode = "generative_model"
        model = GenerativeModel(model_name)

    output_path = f"C:/Users/Georgio/PycharmProjects/string-mpp/id_to_pred_mistral-large-2407_fewshot_multi.json"

    if os.path.exists(output_path):
        with open(output_path, 'r', encoding="utf-8") as f:
            id_to_pred = json.load(f)
    else:
        id_to_pred = {}

    df_bbbp, df_bace, df_esol, df_freesolv, df_clintox = prepare_datasets()

    if config['multi_rep']:
        representations = ["multi"]
    else:
        representations = ['smiles', 'deepsmiles', 'selfies', 'iupac',
                           'inchi']

    for task, df in zip(["bbbp", "bace", "clintox", "esol", "freesolv"],
                        [df_bbbp, df_bace, df_clintox, df_esol, df_freesolv]):

        for representation in representations:  # 'smiles', 'deepsmiles', 'selfies', 'iupac', 'inchi'
            print(f"Running {task} with {representation}...")

            for k in tqdm(list(set([f"{task}_{representation}_{i}" for i in range(len(df))]) - set(id_to_pred.keys()))):
                if config['multi_rep']:
                    molecule_string = " ".join(
                        df[rep][int(k.split("_")[-1])] for rep in ["smiles", "deepsmiles", "selfies", "inchi", "iupac"])
                else:
                    molecule_string = df[representation][int(k.split("_")[-1])]

                assert mode in ["generative_model", "api_direct"]
                if mode == "generative_model":
                    if config['few_shot']:
                        text = completion_with_backoff(model, TASK_TO_PROMPT[task],
                                                       molecule_string,
                                                       build_fewshot_examples(df, k, task, representation,
                                                                              config['multi_rep']))
                else:  # API direct
                    if config['few_shot']:
                        text = completion_with_backoff_direct(model, TASK_TO_PROMPT[task],
                                                              molecule_string, url, headers,
                                                              build_fewshot_examples(df, k, task, representation,
                                                                                     config['multi_rep']))
                    else:
                        text = completion_with_backoff_direct(model, TASK_TO_PROMPT[task],
                                                              molecule_string, url, headers)

                id_to_pred[f"{k}"] = text

                with open(output_path, 'w', encoding="utf-8") as f:
                    json.dump(id_to_pred, f, indent=4)

                time.sleep(1)


if __name__ == '__main__':
    main()
