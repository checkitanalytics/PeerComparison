import os
import psycopg2
from psycopg2.extras import RealDictCursor
import streamlit as st

def get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('PGHOST'),
            port=os.getenv('PGPORT'),
            database=os.getenv('PGDATABASE'),
            user=os.getenv('PGUSER'),
            password=os.getenv('PGPASSWORD')
        )
        return conn
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        return None

def init_database():
    """Initialize database tables"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Create comparison_sets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS comparison_sets (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                companies TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error initializing database: {str(e)}")
        if conn:
            conn.close()
        return False

def save_comparison_set(name, description, companies):
    """
    Save a comparison set to the database
    
    Args:
        name (str): Name of the comparison set
        description (str): Description of the comparison set
        companies (list): List of company ticker symbols
    
    Returns:
        bool: True if successful, False otherwise
    """
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Convert companies list to comma-separated string
        companies_str = ','.join(companies)
        
        cursor.execute("""
            INSERT INTO comparison_sets (name, description, companies)
            VALUES (%s, %s, %s)
        """, (name, description, companies_str))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error saving comparison set: {str(e)}")
        if conn:
            conn.close()
        return False

def get_all_comparison_sets():
    """
    Retrieve all saved comparison sets
    
    Returns:
        list: List of comparison sets
    """
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT id, name, description, companies, created_at, updated_at
            FROM comparison_sets
            ORDER BY updated_at DESC
        """)
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Convert companies string back to list
        for result in results:
            result['companies'] = result['companies'].split(',')
        
        return results
    except Exception as e:
        st.error(f"Error retrieving comparison sets: {str(e)}")
        if conn:
            conn.close()
        return []

def delete_comparison_set(set_id):
    """
    Delete a comparison set
    
    Args:
        set_id (int): ID of the comparison set to delete
    
    Returns:
        bool: True if successful, False otherwise
    """
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM comparison_sets WHERE id = %s
        """, (set_id,))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error deleting comparison set: {str(e)}")
        if conn:
            conn.close()
        return False

def update_comparison_set(set_id, name, description, companies):
    """
    Update an existing comparison set
    
    Args:
        set_id (int): ID of the comparison set
        name (str): New name
        description (str): New description
        companies (list): New list of companies
    
    Returns:
        bool: True if successful, False otherwise
    """
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        companies_str = ','.join(companies)
        
        cursor.execute("""
            UPDATE comparison_sets
            SET name = %s, description = %s, companies = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (name, description, companies_str, set_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error updating comparison set: {str(e)}")
        if conn:
            conn.close()
        return False
