import hashlib
import hmac
import re
import secrets

def hash_password(password: str, salt: str = None) -> tuple:

def verify_password(password: str, salt: str, stored_hash: str) -> bool:

def validate_password_strength(pw: str) -> dict:

def validate_username(un: str) -> bool:

def validate_email(email: str) -> bool:

