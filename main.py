#!/usr/bin/env python3
"""AI Assistant powered by AWS Bedrock with SSO authentication."""

import sys
import signal
import readline  # noqa: F401 — enables arrow-key history in input()

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.prompt import Prompt
from rich.text import Text
from botocore.exceptions import ClientError, NoCredentialsError

from config import load_config
from bedrock import BedrockClient

console = Console()

HELP_TEXT = """
[bold cyan]Available commands:[/bold cyan]

  [yellow]/help[/yellow]              Show this message
  [yellow]/reset[/yellow]             Clear conversation history and start fresh
  [yellow]/model [id][/yellow]        Show or change the current model
  [yellow]/profile [name][/yellow]    Show or change the AWS SSO profile
  [yellow]/quit[/yellow]  [yellow]/exit[/yellow]    Exit the assistant

[dim]Tip: Press Ctrl+C to cancel the current input, Ctrl+D to quit.[/dim]
"""

COMMON_MODELS = [
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "amazon.nova-pro-v1:0",
    "amazon.nova-lite-v1:0",
]


def print_header(config: dict) -> None:
    profile = config["aws"]["profile"]
    region = config["aws"]["region"]
    model = config["model"]["id"]
    console.print(
        Panel(
            f"[bold green]AI Assistant[/bold green] — AWS Bedrock\n\n"
            f"  Profile : [cyan]{profile}[/cyan]\n"
            f"  Region  : [cyan]{region}[/cyan]\n"
            f"  Model   : [cyan]{model}[/cyan]\n\n"
            f"Type [yellow]/help[/yellow] for commands  •  [yellow]/quit[/yellow] to exit",
            border_style="green",
            padding=(0, 2),
        )
    )


def handle_sso_error(profile: str, error: Exception) -> None:
    msg = str(error)
    if any(k in msg for k in ("ExpiredToken", "UnauthorizedSSO", "SSOToken", "token")):
        console.print(f"\n[red bold]AWS credentials expired or missing.[/red bold]")
        console.print(f"Run this command to log in:\n")
        console.print(f"  [yellow]aws sso login --profile {profile}[/yellow]\n")
    else:
        console.print(f"\n[red]AWS error:[/red] {error}\n")


def run_chat(client: BedrockClient, config: dict) -> None:
    system_prompt: str = config["system_prompt"]

    # Gracefully handle Ctrl+C (cancel input line, don't exit)
    def _sigint(sig, frame):
        console.print()  # newline after ^C

    signal.signal(signal.SIGINT, _sigint)

    while True:
        try:
            user_input = Prompt.ask(
                f"\n[bold blue]You[/bold blue] [dim](turn {client.turn_count + 1})[/dim]"
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # --- Commands ---
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit"):
                console.print("[dim]Goodbye![/dim]")
                break

            elif cmd == "/help":
                console.print(HELP_TEXT)

            elif cmd == "/reset":
                client.reset()
                console.print("[green]Conversation history cleared.[/green]")

            elif cmd == "/model":
                if arg:
                    client.model_id = arg
                    config["model"]["id"] = arg
                    console.print(f"[green]Model changed to:[/green] [cyan]{arg}[/cyan]")
                else:
                    console.print(f"[yellow]Current model:[/yellow] [cyan]{client.model_id}[/cyan]")
                    console.print("\n[dim]Common models:[/dim]")
                    for m in COMMON_MODELS:
                        console.print(f"  [dim]{m}[/dim]")

            elif cmd == "/profile":
                if arg:
                    config["aws"]["profile"] = arg
                    # Reinitialise client with new profile
                    try:
                        new_client = BedrockClient(config)
                        client.__dict__.update(new_client.__dict__)
                        console.print(f"[green]Profile changed to:[/green] [cyan]{arg}[/cyan]")
                    except Exception as e:
                        console.print(f"[red]Failed to switch profile:[/red] {e}")
                else:
                    console.print(f"[yellow]Current profile:[/yellow] [cyan]{config['aws']['profile']}[/cyan]")

            else:
                console.print(f"[red]Unknown command:[/red] {cmd}  (type /help for a list)")
            continue

        # --- Stream response ---
        console.print(f"\n[bold green]Assistant[/bold green]")
        full_text = ""
        try:
            with Live(console=console, refresh_per_second=20, vertical_overflow="visible") as live:
                for chunk in client.chat(user_input, system_prompt):
                    full_text += chunk
                    # Render as Markdown so code blocks, bold, etc. look nice
                    live.update(Markdown(full_text))

        except (ClientError, NoCredentialsError) as e:
            handle_sso_error(config["aws"]["profile"], e)
            # Roll back the user message that was added to history
            if client.history and client.history[-1]["role"] == "user":
                client.history.pop()
        except Exception as e:
            console.print(f"[red]Unexpected error:[/red] {e}")
            if client.history and client.history[-1]["role"] == "user":
                client.history.pop()


def main() -> None:
    config = load_config()
    print_header(config)

    try:
        client = BedrockClient(config)
    except NoCredentialsError:
        profile = config["aws"]["profile"]
        console.print(f"\n[red bold]No AWS credentials found for profile '{profile}'.[/red bold]")
        console.print(f"Run:\n  [yellow]aws sso login --profile {profile}[/yellow]\n")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Failed to initialise Bedrock client:[/red] {e}\n")
        sys.exit(1)

    run_chat(client, config)


if __name__ == "__main__":
    main()
