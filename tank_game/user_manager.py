import json
import os
import hashlib

USERS_FILE = os.path.join(os.path.dirname(__file__), 'users.json')


def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def register_user(username, password):
    """Register a new user"""
    users = load_users()
    if username in users:
        return False, "User already exists"
    
    users[username] = {
        'password': hash_password(password),
        'score': 0,
        'games_played': 0
    }
    save_users(users)
    return True, "User registered successfully"


def login_user(username, password):
    """Verify user login"""
    users = load_users()
    if username not in users:
        return False, "User not found"
    
    if users[username]['password'] != hash_password(password):
        return False, "Incorrect password"
    
    return True, "Login successful"


def get_user_stats(username):
    """Get user statistics"""
    users = load_users()
    if username in users:
        return users[username]
    return None


def update_user_stats(username, score, games_played):
    """Update user stats"""
    users = load_users()
    if username in users:
        users[username]['score'] = score
        users[username]['games_played'] = games_played
        save_users(users)
