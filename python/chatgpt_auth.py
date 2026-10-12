#!/usr/bin/env python3
"""ChatGPT plan authorization for this local VisionAssist installation."""
import argparse
import base64
from contextlib import contextmanager
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser

ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = ISSUER + "/api/accounts/authorize"
TOKEN_URL = ISSUER + "/api/accounts/oauth/token"
RESOURCE = "https://api.openai.com/v1"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke " + PLAN_SCOPE
STORAGE_DIR = Path(__file__).resolve().parents[1] / ".local" / "chatgpt"
USAGE_URL = "https://chatgpt.com/#settings/Usage"
LIGHTWEIGHT_MODELS = ("gpt-5.6-luna", "gpt-5.4-nano")


class AuthError(Exception):
    """A user-readable error that never contains credentials."""


def request_json(url, form=None, access_token=None):
    headers = {}
    data = None
    if form is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        data = urllib.parse.urlencode(form).encode()
    if access_token:
        headers["Authorization"] = "Bearer " + access_token
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        status = error.code
        error.close()
        if status == 401:
            raise AuthError("Your ChatGPT connection needs renewing. Press L to sign in again.") from None
        if status == 429:
            raise AuthError("ChatGPT usage limit reached. Review your plan or app limit in ChatGPT Settings, Usage.") from None
        raise AuthError(f"ChatGPT returned HTTP {status}. Please try again or sign in again.") from None
    except OSError:
        raise AuthError("Could not connect to ChatGPT. Check your connection and try again.") from None
    except (ValueError, TypeError):
        raise AuthError("ChatGPT returned an invalid response. Please try again.") from None


def atomic_save(path, data):
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(descriptor, "w") as file:
            json.dump(data, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def locked_store():
    STORAGE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    STORAGE_DIR.chmod(0o700)
    descriptor = os.open(STORAGE_DIR / "session.lock", os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = STORAGE_DIR / "sessions.json"
        try:
            data = json.loads(path.read_text()) if path.exists() else {
                "ext_agent_host_id": "urn:uuid:" + str(uuid.uuid4()),
                "profiles": [], "active_client_id": None
            }
            if not isinstance(data, dict) or not isinstance(data.get("profiles"), list):
                raise ValueError()
            if not isinstance(data.get("ext_agent_host_id"), str) or not data["ext_agent_host_id"]:
                raise ValueError()
            if any(not isinstance(p, dict) or not isinstance(p.get("client_id"), str) for p in data["profiles"]):
                raise ValueError()
        except (OSError, ValueError):
            raise AuthError("The saved ChatGPT connection could not be read. Restore the session file or move it aside and sign in again.") from None
        if not path.exists():
            atomic_save(path, data)
        yield data
        atomic_save(path, data)


def active_profile(data):
    return next((profile for profile in data["profiles"]
                 if profile["client_id"] == data.get("active_client_id")), None)


def validate_callback(query, state, saved_client_id=None):
    received = query.get("state", [""])
    if len(received) != 1 or not secrets.compare_digest(received[0], state):
        raise AuthError("Sign-in callback did not match this attempt. Please try again.")
    if "error" in query:
        raise AuthError("ChatGPT sign-in was declined or cancelled. Your previous connection is unchanged.")
    codes = query.get("code", [])
    clients = query.get("client_id", [])
    if len(codes) != 1 or not codes[0] or len(clients) > 1:
        raise AuthError("ChatGPT sign-in did not return a complete registration.")
    client_id = clients[0] if clients else saved_client_id
    if not client_id or client_id == "dynamic_agent_client":
        raise AuthError("ChatGPT sign-in did not issue a client ID. Please try again.")
    if saved_client_id and client_id != saved_client_id:
        raise AuthError("ChatGPT returned a different account registration. Your saved connection is unchanged.")
    return codes[0], client_id


def auth_metadata():
    metadata = request_json(ISSUER + "/.well-known/openid-configuration")
    if metadata.get("issuer") != ISSUER:
        raise AuthError("ChatGPT sign-in metadata could not be verified.")
    return metadata


def trusted_auth_url(url):
    parsed = urllib.parse.urlparse(url or "")
    if parsed.scheme != "https" or parsed.hostname != "auth.openai.com" or parsed.port not in (None, 443):
        raise AuthError("ChatGPT sign-in metadata contained an unexpected endpoint.")
    return url


def jwt_library():
    try:
        import jwt
        return jwt
    except ImportError:
        raise AuthError("Sign-in support needs installation. Run .venv/bin/python3 -m pip install -r requirements.txt from the project folder.") from None


def validate_id_token(token, client_id, nonce=None, previous=None):
    jwt = jwt_library()
    try:
        metadata = auth_metadata()
        key_url = trusted_auth_url(metadata.get("jwks_uri"))
        keys = jwt.PyJWKClient(key_url, timeout=20)
        signing_key = keys.get_signing_key_from_jwt(token)
        claims = jwt.decode(token, signing_key.key, algorithms=["RS256"],
                            issuer=ISSUER, audience=client_id,
                            options={"require": ["iss", "aud", "exp", "sub", "iat"]})
        if nonce is not None and claims.get("nonce") != nonce:
            raise AuthError("ChatGPT identity did not match this sign-in attempt.")
        if previous and (claims["sub"] != previous["subject"] or claims["iss"] != previous["issuer"]):
            raise AuthError("ChatGPT returned a different identity. Your saved connection is unchanged.")
        return claims
    except jwt.PyJWTError:
        raise AuthError("ChatGPT identity could not be verified. Please sign in again.") from None


def apply_token_response(profile, tokens, claims=None, initial=False):
    if not isinstance(tokens.get("access_token"), str) or not tokens["access_token"]:
        raise AuthError("ChatGPT did not return an access token.")
    if str(tokens.get("token_type", "")).lower() != "bearer":
        raise AuthError("ChatGPT returned an unsupported token type.")
    scopes = tokens.get("scope", "")
    if initial and not scopes:
        raise AuthError("ChatGPT did not confirm the granted permissions.")
    updated = dict(profile)
    if claims:
        updated.update(email=claims.get("email", ""), subject=claims["sub"], issuer=claims["iss"])
    try:
        lifetime = int(tokens["expires_in"])
        if lifetime <= 0:
            raise ValueError()
    except (KeyError, ValueError, TypeError):
        raise AuthError("ChatGPT returned an invalid session expiry.") from None
    for field in ("access_token", "refresh_token", "id_token", "token_type", "earliest_refresh_at"):
        if field in tokens:
            updated[field] = tokens[field]
    if initial and not updated.get("id_token"):
        raise AuthError("ChatGPT did not return a verified identity token.")
    updated["scopes"] = scopes.split() if scopes else profile.get("scopes", [])
    updated["expires_at"] = time.time() + lifetime
    return updated


def login(new_account=False, account=None, open_browser=True):
    jwt_library()
    with locked_store() as data:
        previous = active_profile(data)
        if account is not None:
            if not 1 <= account <= len(data["profiles"]):
                raise AuthError("That saved account number does not exist.")
            previous = data["profiles"][account - 1]
        if new_account:
            previous = None
        host_id = data["ext_agent_host_id"]
    state, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(3))
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    received = {}

    class CallbackHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Callback URLs contain authorization codes; never log them.

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/auth/callback":
                self.send_error(404)
                return
            query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
            try:
                result = validate_callback(query, state, previous["client_id"] if previous else None)
            except AuthError as error:
                # Ignore unrelated callbacks without consuming the valid attempt.
                if query.get("state") != [state]:
                    self.send_error(400, "Invalid sign-in attempt")
                    return
                received["error"] = error
            else:
                received["result"] = result
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b"<h1>Return to VisionAssist</h1><p>The browser step is complete. VisionAssist is verifying your connection.</p>")

    with HTTPServer(("127.0.0.1", 0), CallbackHandler) as server:
        redirect_uri = f"http://127.0.0.1:{server.server_port}/auth/callback"
        params = dict(client_id=previous["client_id"] if previous else "dynamic_agent_client",
                      ext_agent_host_id=host_id, response_type="code", redirect_uri=redirect_uri,
                      scope=SCOPES, resource=RESOURCE, state=state, nonce=nonce,
                      code_challenge_method="S256", code_challenge=challenge)
        if previous:
            if previous.get("id_token"):
                params["id_token_hint"] = previous["id_token"]
            if PLAN_SCOPE not in previous.get("scopes", []):
                params["prompt"] = "consent"
        else:
            params["agent_name_hint"] = "VisionAssist"
        url = AUTHORIZE_URL + "?" + urllib.parse.urlencode(params)
        print("Continue with ChatGPT. Sign in in your browser and allow VisionAssist to use your ChatGPT plan.", flush=True)
        if open_browser:
            if not webbrowser.open(url):
                raise AuthError("Could not open the browser. Retry sign-in from a desktop terminal.")
        else:
            # Initial registration URL has no credentials. Used only for an
            # explicit command-line browser handoff, never for reauthorization.
            if previous:
                raise AuthError("Returning sign-in requires the browser option.")
            print(url, flush=True)
        server.timeout = 1
        deadline = time.monotonic() + 300
        while not received and time.monotonic() < deadline:
            server.handle_request()
        if not received:
            raise AuthError("ChatGPT sign-in timed out. Press L to try again.")
        if "error" in received:
            raise received["error"]
    code, client_id = received["result"]
    tokens = request_json(TOKEN_URL, dict(grant_type="authorization_code", client_id=client_id,
                                         code=code, code_verifier=verifier, redirect_uri=redirect_uri,
                                         resource=RESOURCE))
    claims = validate_id_token(tokens.get("id_token", ""), client_id, nonce, previous)
    profile = apply_token_response(dict(client_id=client_id, ext_agent_host_id=host_id), tokens, claims, initial=True)
    with locked_store() as data:
        profiles = data["profiles"]
        existing = next((i for i, p in enumerate(profiles) if p["client_id"] == client_id), None)
        if existing is None:
            profiles.append(profile)
        else:
            profiles[existing] = profile
        data["active_client_id"] = client_id
    print("Connected to ChatGPT as " + (profile.get("email") or "your selected account") + ".", flush=True)
    if PLAN_SCOPE not in profile["scopes"]:
        print("ChatGPT plan usage was not enabled. Sign in again and allow plan usage before analyzing an image.", flush=True)
    else:
        print("You're using your ChatGPT plan. Analysis counts toward your existing plan limits. Manage app limits in ChatGPT Settings, Usage.", flush=True)
    return profile


def access_token():
    # Lock across refresh and atomic replacement to prevent rotating-token races
    # between the camera helper and another local command.
    with locked_store() as data:
        profile = active_profile(data)
        if not profile or not profile.get("access_token"):
            raise AuthError("Press L in VisionAssist to Continue with ChatGPT and connect your Plus plan.")
        if PLAN_SCOPE not in profile.get("scopes", []):
            raise AuthError("ChatGPT plan usage is not enabled. Press L and allow VisionAssist to use your plan.")
        if profile.get("expires_at", 0) <= time.time() + 60:
            if not profile.get("refresh_token"):
                raise AuthError("Your ChatGPT connection has expired. Press L to sign in again.")
            tokens = request_json(TOKEN_URL, dict(grant_type="refresh_token", client_id=profile["client_id"],
                                                 refresh_token=profile["refresh_token"], resource=RESOURCE))
            claims = validate_id_token(tokens["id_token"], profile["client_id"], previous=profile) if tokens.get("id_token") else None
            updated = apply_token_response(profile, tokens, claims)
            if not tokens.get("refresh_token"):
                raise AuthError("ChatGPT did not renew the session. Press L to sign in again.")
            data["profiles"][data["profiles"].index(profile)] = profile = updated
        token = profile["access_token"]
        plan_enabled = PLAN_SCOPE in profile["scopes"]
    if not plan_enabled:
        raise AuthError("ChatGPT plan usage is no longer enabled. Press L to reconnect.")
    return token


def choose_model(token, preferred=""):
    catalog = request_json(RESOURCE + "/models", access_token=token)
    if not isinstance(catalog, dict) or not isinstance(catalog.get("models"), list):
        raise AuthError("ChatGPT returned an invalid model list. Please try again.")
    entries = [entry for entry in catalog.get("models", []) if entry.get("visibility") == "list"]
    models = {entry.get("slug"): entry for entry in entries}
    if preferred:
        if preferred not in models:
            raise AuthError("The selected ChatGPT model is unavailable on this account. Check chatgpt_model in .local/openai.conf.")
        chosen = preferred
    else:
        chosen = next((model for model in LIGHTWEIGHT_MODELS if model in models), None)
    if not chosen:
        raise AuthError("No supported lightweight vision model is listed for this account. Run chatgpt_auth.py models and select a vision model with chatgpt_model in .local/openai.conf.")
    modalities = models[chosen].get("input_modalities")
    if modalities is not None and "image" not in modalities:
        raise AuthError("The selected ChatGPT model does not accept images. Select a vision model.")
    return chosen


def show_status():
    with locked_store() as data:
        if not data["profiles"]:
            print("ChatGPT is not connected. Press L in VisionAssist to Continue with ChatGPT.")
        for i, profile in enumerate(data["profiles"], 1):
            active = " (active)" if profile["client_id"] == data.get("active_client_id") else ""
            status = "plan enabled" if PLAN_SCOPE in profile.get("scopes", []) and profile.get("access_token") else "sign-in required"
            print(f"{i}. {profile.get('email') or 'ChatGPT account'}{active}: {status}")


def logout():
    confirmed = False
    with locked_store() as data:
        profile = active_profile(data)
        if not profile:
            print("ChatGPT is already disconnected locally.")
            return
        if profile.get("refresh_token"):
            try:
                endpoint = trusted_auth_url(auth_metadata().get("revocation_endpoint"))
                form = dict(token=profile["refresh_token"], token_type_hint="refresh_token", client_id=profile["client_id"])
                req = urllib.request.Request(endpoint, data=urllib.parse.urlencode(form).encode(),
                                             headers={"Content-Type": "application/x-www-form-urlencoded"})
                with urllib.request.urlopen(req, timeout=20) as response:
                    confirmed = response.status == 200
            except (AuthError, OSError):
                pass
        for key in ("access_token", "refresh_token", "id_token", "expires_at", "scopes"):
            profile.pop(key, None)
    print("Signed out locally." + (" ChatGPT session revoked." if confirmed else " Remote revocation was not confirmed; disconnect VisionAssist in ChatGPT Settings."))


def main():
    parser = argparse.ArgumentParser(description="Connect VisionAssist to your ChatGPT plan.")
    parser.add_argument("command", choices=["login", "status", "models", "use", "logout", "usage"])
    parser.add_argument("--new-account", action="store_true")
    parser.add_argument("--account", type=int)
    args = parser.parse_args()
    try:
        if args.command == "login":
            profile = login(args.new_account, args.account)
            if PLAN_SCOPE not in profile["scopes"]:
                return 1
        elif args.command == "status":
            show_status()
        elif args.command == "models":
            catalog = request_json(RESOURCE + "/models", access_token=access_token())
            for entry in catalog.get("models", []):
                if entry.get("visibility") == "list":
                    print(f"{entry['slug']}: {entry.get('display_name', entry['slug'])}")
        elif args.command == "use":
            with locked_store() as data:
                if args.account is None or not 1 <= args.account <= len(data["profiles"]):
                    raise AuthError("Choose a saved account using --account and its number from status.")
                data["active_client_id"] = data["profiles"][args.account - 1]["client_id"]
            show_status()
        elif args.command == "logout":
            logout()
        elif args.command == "usage":
            webbrowser.open(USAGE_URL)
    except (AuthError, OSError, ValueError) as error:
        print(str(error) if isinstance(error, AuthError) else "Could not complete ChatGPT sign-in. Please try again.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
