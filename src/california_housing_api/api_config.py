import secrets
import hashlib
import os

API_KEYS = {}


def generate_api_key():
    api_key = secrets.token_hex(32)
    hashed = hashlib.sha256(api_key.encode()).hexdigest()
    API_KEYS[hashed] = True
    return api_key


def save_api_keys():
    with open(os.getcwd() + os.sep + "api_keys.txt", "w") as f:
        for hashed in API_KEYS:
            f.write(f"{hashed}\n")


def load_api_keys():
    if os.path.exists(os.getcwd() + os.sep + "api_keys.txt"):
        with open(os.getcwd() + os.sep + "api_keys.txt", "r") as f:
            for line in f:
                API_KEYS[line.strip()] = True


load_api_keys()
if not API_KEYS:
    generate_api_key()
    save_api_keys()
