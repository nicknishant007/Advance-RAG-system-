import sqlite3

DB_PATH="storage/ingestion.db" 

def get_connection():
    return sqlite3.connect(DB_PATH)