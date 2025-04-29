import os
import sys
import subprocess
import threading
import time
from rich.console import Console

# Initialize console
console = Console()

def run_backend():
    """Run the Flask backend server"""
    try:
        console.print("[bold blue]Starting Phoenix Agent Backend...[/bold blue]")
        # Use the file name of your backend script
        subprocess.run([sys.executable, "phoenix_agent.py"], check=True)
    except KeyboardInterrupt:
        console.print("[yellow]Backend server stopped[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Backend Error: {str(e)}[/bold red]")

def run_frontend():
    """Run the frontend development server"""
    try:
        console.print("[bold green]Starting Phoenix Agent Frontend...[/bold green]")
        # Change directory to the frontend folder and run the appropriate command
        subprocess.run(["npm", "start"], cwd="frontend", check=True)
    except KeyboardInterrupt:
        console.print("[yellow]Frontend server stopped[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Frontend Error: {str(e)}[/bold red]")

def main():
    """Main function to launch both servers"""
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    frontend_thread = threading.Thread(target=run_frontend, daemon=True)
    
    try:
        backend_thread.start()
        console.print("[blue]Backend thread started[/blue]")
        
        # Small delay to ensure backend starts first
        time.sleep(2)
        
        frontend_thread.start()
        console.print("[green]Frontend thread started[/green]")
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("[yellow]Shutting down servers...[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
    finally:
        console.print("[bold]Phoenix Agent stopped[/bold]")

if __name__ == "__main__":
    main()