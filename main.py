#!/usr/bin/env python3
"""AI Assistant powered by AWS Bedrock with SSO authentication."""

import sys
import os
import signal
import readline  # noqa: F401 — enables arrow-key history in input()

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.prompt import Prompt
from rich.table import Table
from botocore.exceptions import ClientError, NoCredentialsError

from config import load_config
from bedrock import BedrockClient
from context import FileContext

console = Console()

HELP_TEXT = """
[bold cyan]Commands:[/bold cyan]

  [yellow]/load <path>[/yellow]       Load a file or directory into context (sent to Bedrock)
  [yellow]/unload [path][/yellow]     Remove a specific file from context, or clear all
  [yellow]/context[/yellow]           Show currently loaded files
  [yellow]/reset[/yellow]             Clear conversation history (keeps loaded files)
  [yellow]/model [id][/yellow]        Show or change the model
  [yellow]/profile [name][/yellow]    Show or change the AWS SSO profile
  [yellow]/help[/yellow]              Show this message
  [yellow]/quit[/yellow]  [yellow]/exit[/yellow]    Exit

[dim]Tip: Use @filename in your message to load a file inline on the fly.
Example: "explain @src/main.py" — loads the file and sends it with your question.[/dim]
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
    cwd = os.getcwd()
    console.print(
        Panel(
            f"[bold green]AI Assistant[/bold green] — AWS Bedrock\n\n"
            f"  Profile : [cyan]{profile}[/cyan]\n"
            f"  Region  : [cyan]{region}[/cyan]\n"
            f"  Model   : [cyan]{model}[/cyan]\n"
            f"  Workdir : [cyan]{cwd}[/cyan]\n\n"
            f"Type [yellow]/help[/yellow] for commands  •  [yellow]/load <path>[/yellow] to add files  •  [yellow]/quit[/yellow] to exit",
            border_style="green",
            padding=(0, 2),
        )
    )


def show_context(ctx: FileContext) -> None:
    files = ctx.list_loaded()
    if not files:
        console.print("[dim]No files loaded. Use /load <path> to add files.[/dim]")
        return
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("File", style="cyan")
    table.add_column("Size", justify="right")
    for label, chars in files:
        table.add_row(label, f"{chars:,} chars")
    table.caption = f"Total: {ctx.total_chars():,} chars"
    console.print(table)


def handle_sso_error(profile: str, error: Exception) -> None:
    msg = str(error)
    if any(k in msg for k in ("ExpiredToken", "UnauthorizedSSO", "SSOToken", "token")):
        console.print(f"\n[red bold]AWS credentials expired or missing.[/red bold]")
        console.print(f"Run:\n\n  [yellow]aws sso login --profile {profile}[/yellow]\n")
    else:
        console.print(f"\n[red]AWS error:[/red] {error}\n")


def run_chat(client: BedrockClient, config: dict) -> None:
    system_prompt: str = config["system_prompt"]
    ctx = FileContext()
    cwd = os.getcwd()

    def _sigint(sig, frame):
        console.print()

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

        # ── Commands ──────────────────────────────────────────────────
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

            elif cmd == "/load":
                if not arg:
                    console.print("[red]Usage: /load <file or directory>[/red]")
                else:
                    messages = ctx.load(arg, base_dir=cwd)
                    for msg in messages:
                        console.print(msg)

            elif cmd == "/unload":
                if arg:
                    removed = ctx.unload_path(arg)
                    if removed:
                        console.print(f"[green]Removed:[/green] {arg}")
                    else:
                        console.print(f"[red]Not found in context:[/red] {arg}")
                else:
                    n = ctx.unload()
                    console.print(f"[green]Cleared {n} file(s) from context.[/green]")

            elif cmd == "/context":
                show_context(ctx)

            elif cmd == "/model":
                if arg:
                    client.model_id = arg
                    config["model"]["id"] = arg
                    console.print(f"[green]Model →[/green] [cyan]{arg}[/cyan]")
                else:
                    console.print(f"[yellow]Current model:[/yellow] [cyan]{client.model_id}[/cyan]")
                    console.print("\n[dim]Common models:[/dim]")
                    for m in COMMON_MODELS:
                        console.print(f"  [dim]{m}[/dim]")

            elif cmd == "/profile":
                if arg:
                    config["aws"]["profile"] = arg
                    try:
                        new_client = BedrockClient(config)
                        client.__dict__.update(new_client.__dict__)
                        console.print(f"[green]Profile →[/green] [cyan]{arg}[/cyan]")
                    except Exception as e:
                        console.print(f"[red]Failed to switch profile:[/red] {e}")
                else:
                    console.print(f"[yellow]Current profile:[/yellow] [cyan]{config['aws']['profile']}[/cyan]")

            else:
                console.print(f"[red]Unknown command:[/red] {cmd}  (type /help)")
            continue

        # ── Expand @file references ───────────────────────────────────
        if "@" in user_input:
            user_input, at_msgs = ctx.resolve_at_references(user_input, base_dir=cwd)
            for msg in at_msgs:
                console.print(f"[dim]{msg}[/dim]")

        # ── Stream response ───────────────────────────────────────────
        file_context = ctx.get_context_block() or None

        if file_context:
            total = ctx.total_chars()
            console.print(f"[dim]  (sending {total:,} chars of file context)[/dim]")

        console.print(f"\n[bold green]Assistant[/bold green]")
        full_text = ""
        try:
            with Live(console=console, refresh_per_second=20, vertical_overflow="visible") as live:
                for chunk in client.chat(user_input, system_prompt, file_context):
                    full_text += chunk
                    live.update(Markdown(full_text))

        except (ClientError, NoCredentialsError) as e:
            handle_sso_error(config["aws"]["profile"], e)
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
