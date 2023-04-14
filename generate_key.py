import pickle
from pathlib import Path

import streamlit_authenticator as stauth

name = ["admin","user"]
usernames = ["admin1","user1"]
password = ["123456","qwerty"]


hashed_passwords = stauth.Hasher(password).generate()

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("wb") as file:
    pickle.dump(hashed_passwords, file)