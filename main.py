"""
@file main.py
@brief Central CLI entry point for the Codebase Analysis project.
@details Provides a menu-driven interface to generate summaries, manage vector 
databases, and preview indexed data.
"""

import os
import subprocess
import sys
import chromadb
from pathlib import Path

def clear_screen():
    """
    @brief Clears the terminal screen for better readability.
    """
    os.system("cls" if os.name == "nt" else "clear")

def create_summaries():
    """
    @brief Generates summaries using the summary agent
    """
    clear_screen()
    codebase = Path(input("\nEnter the name of the codebase to analyze: ").strip()).resolve()
    codebase_name = codebase.name

    # get relative path to target codebase from targetCodebases directory


    print("Building vector database...")
    subprocess.run([sys.executable, "-m", "src.build_database", str(codebase)], text=True)
    
    print("Generating summaries...")
    subprocess.run([sys.executable, "-m", "agent.file_summary_agent", str(codebase)], text=True)

    print("Generating summary database...")
    subprocess.run([sys.executable, "-m", "src.build_database_JSON", codebase_name], text=True)

    print("Summaries generated successfully!")
    input("Press enter to return to main menu...")

def view_collections(db_type: str):
    """
    @brief Displays existing collections and allows the user to preview their contents.
    """
    clear_screen()
    db_dir = Path("vectorStores").resolve()
    if not db_dir.exists():
        print("No vector stores found. Please generate summaries first.")
        input("Press enter to return to main menu...")
        return
    
    # initialize client
    client = chromadb.PersistentClient(path = str(db_dir))
    collections = client.list_collections()
    
    # get suffix based on db type
    suffix = "_summary_db" if db_type == "summary" else "_code_db"

    # get relevant collections
    relevant = [col for col in collections if col.name.endswith(suffix)]

    if not relevant:
        print(f"No {db_type} collections found. Please generate summaries first.")
        input("Press enter to return to main menu...")
        return
    
    print(f"--- Available {db_type.capitalize()} Collections ---")
    for i, col in enumerate(relevant):
        print(f"{i+1}. {col.name}")
        
    choice = input(f"\nSelect a number to preview (or 'b' to go back): ")
    if choice.isdigit() and int(choice) <= len(relevant):
        target = relevant[int(choice)-1]
        results = target.peek(limit=3) # Grab first 3 entries
            
        print(f"\n--- Previewing: {target.name} ---")
        for idx, doc in enumerate(results['documents']):
            print(f"\n[{idx+1}] {doc[:600]}...") # Show first 600 chars
        
    input("\nPress Enter to return to menu...")

def main_menu():
    while True:
        clear_screen()
        print("========================================")
        print("   CODEBASE ANALYSIS SYSTEM - CLI")
        print("========================================")
        print("1. Create Summaries & Index Codebase")
        print("2. View Summary Collections")
        print("3. View Source Code Collections")
        print("4. Exit")
        print("----------------------------------------")
        
        choice = input("Select an option (1-4): ")

        if choice == '1':
            create_summaries()
        elif choice == '2':
            view_collections(db_type="summary")
        elif choice == '3':
            view_collections(db_type="source")
        elif choice == '4':
            print("Exiting system. Goodbye!")
            sys.exit()
        else:
            print("Invalid selection. Try again.")

if __name__ == "__main__":
    main_menu()