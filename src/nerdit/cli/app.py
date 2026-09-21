"""Nerdit CLI application."""

import typer

import nerdit
from nerdit.utils.frozen import restore_host_loader_env

app = typer.Typer(
    name="nerdit",
    help=(
        "The local AI PaaS — deploy and operate apps on your own hardware, "
        "with first-class local AI"
    ),
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"nerdit {nerdit.__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    pass


# Import and register commands (deferred to avoid circular imports).
# PLR0915: one deferred import + one registration per verb — the list IS the CLI
# surface, so it grows one statement per shipped verb by construction.
def _register_commands() -> None:  # noqa: PLR0915
    from nerdit.cli.commands.apply import apply  # noqa: F811
    from nerdit.cli.commands.backup import backup, restore  # noqa: F811
    from nerdit.cli.commands.capabilities import capabilities  # noqa: F811
    from nerdit.cli.commands.check_deps import check_deps  # noqa: F811
    from nerdit.cli.commands.config import config_app  # noqa: F811
    from nerdit.cli.commands.connect import connect  # noqa: F811
    from nerdit.cli.commands.daemon import daemon_app  # noqa: F811
    from nerdit.cli.commands.db import db_app  # noqa: F811
    from nerdit.cli.commands.deploy import deploy  # noqa: F811
    from nerdit.cli.commands.dev import dev  # noqa: F811
    from nerdit.cli.commands.diagnose import diagnose  # noqa: F811
    from nerdit.cli.commands.doctor import doctor  # noqa: F811
    from nerdit.cli.commands.domains import domains_app  # noqa: F811
    from nerdit.cli.commands.events import events  # noqa: F811
    from nerdit.cli.commands.exit import exit_daemon  # noqa: F811
    from nerdit.cli.commands.init import init  # noqa: F811
    from nerdit.cli.commands.license import license_app  # noqa: F811
    from nerdit.cli.commands.link import link, unlink  # noqa: F811
    from nerdit.cli.commands.logs import logs  # noqa: F811
    from nerdit.cli.commands.mcp import mcp  # noqa: F811
    from nerdit.cli.commands.models import models_app  # noqa: F811
    from nerdit.cli.commands.projects import projects_app  # noqa: F811
    from nerdit.cli.commands.proxy import proxy_app  # noqa: F811
    from nerdit.cli.commands.routes import routes  # noqa: F811
    from nerdit.cli.commands.secrets import secrets_app  # noqa: F811
    from nerdit.cli.commands.serve import serve  # noqa: F811
    from nerdit.cli.commands.services import services_app  # noqa: F811
    from nerdit.cli.commands.share import share, unshare  # noqa: F811
    from nerdit.cli.commands.store import store_app  # noqa: F811
    from nerdit.cli.commands.system import disk, gc  # noqa: F811
    from nerdit.cli.commands.token import token_app  # noqa: F811
    from nerdit.cli.commands.trust import trust, untrust  # noqa: F811
    from nerdit.cli.commands.uninstall import uninstall  # noqa: F811
    from nerdit.cli.commands.update import update  # noqa: F811
    from nerdit.cli.commands.vars import vars_app  # noqa: F811

    app.command(name="check-deps")(check_deps)
    app.command()(init)
    app.command()(serve)
    app.command()(deploy)
    # (P40d) The declaration verb: every [services.<name>] of a [project].
    app.command()(apply)
    app.command()(dev)
    app.command()(logs)
    app.command()(connect)
    app.command(name="mcp")(mcp)
    app.command(name="exit")(exit_daemon)
    app.command()(trust)
    app.command()(untrust)
    app.command()(link)
    app.command()(unlink)
    app.command()(share)
    app.command()(unshare)
    app.command()(uninstall)
    app.command()(update)
    app.command()(diagnose)
    app.command()(doctor)
    app.command()(capabilities)
    app.command()(routes)
    app.command()(events)
    app.command()(disk)
    app.command()(gc)
    app.command()(backup)
    app.command()(restore)
    app.add_typer(token_app, name="token")
    app.add_typer(config_app, name="config")
    app.add_typer(services_app, name="services")
    app.add_typer(secrets_app, name="secrets")
    # (P40b) The project noun: list/create/show/delete of the grouping every
    # deployed service belongs to; the name is reserved for the creating token.
    app.add_typer(projects_app, name="projects")
    # (P40c) Variables of a project: plain or secret, project or service scope.
    # `nerdit secrets` stays an alias surface forever (D-P40-1).
    app.add_typer(vars_app, name="vars")
    app.add_typer(models_app, name="models")
    app.add_typer(db_app, name="db")
    app.add_typer(store_app, name="store")
    app.add_typer(proxy_app, name="proxy")
    app.add_typer(daemon_app, name="daemon")
    # A group, not three flat verbs: ``list``/``add``/``remove``
    # are meaningless without the noun, and it keeps the exposure surface
    # (``share``/``unshare``/``domains``) legible at the top level.
    app.add_typer(domains_app, name="domains")
    # (P17d D-LIC5) A group, not three flat verbs: ``install``/``status``/
    # ``remove`` are meaningless without the noun, and the group leaves room for
    # a future ``nerdit license show`` without another top-level name.
    app.add_typer(license_app, name="license")


_register_commands()


def main() -> None:
    # Children must see the host's loader path, not the bundle's (frozen only).
    restore_host_loader_env()
    app()
