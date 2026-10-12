import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch
import urllib.parse

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

import chatgpt_auth as auth


class ChatGPTAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.storage = Path(self.directory.name) / 'private'
        self.patch_storage = patch.object(auth, 'STORAGE_DIR', self.storage)
        self.patch_storage.start()
        self.addCleanup(self.patch_storage.stop)

    def save_profile(self, **overrides):
        profile = dict(client_id='oaiapp_test', issuer=auth.ISSUER, subject='user-1',
                       access_token='test-access', refresh_token='test-refresh',
                       expires_at=time.time() + 3600, scopes=[auth.PLAN_SCOPE])
        profile.update(overrides)
        with auth.locked_store() as data:
            data['profiles'] = [profile]
            data['active_client_id'] = profile['client_id']
        return profile

    def test_private_atomic_storage_and_stable_host(self):
        with auth.locked_store() as data:
            host = data['ext_agent_host_id']
        with auth.locked_store() as data:
            self.assertEqual(host, data['ext_agent_host_id'])
        self.assertEqual(self.storage.stat().st_mode & 0o777, 0o700)
        self.assertEqual((self.storage / 'sessions.json').stat().st_mode & 0o777, 0o600)
        self.assertEqual((self.storage / 'session.lock').stat().st_mode & 0o777, 0o600)

    def test_callback_state_and_client_binding(self):
        good = dict(state=['state'], code=['code'], client_id=['oaiapp_test'])
        self.assertEqual(auth.validate_callback(good, 'state'), ('code', 'oaiapp_test'))
        self.assertEqual(auth.validate_callback(dict(state=['state'], code=['code']), 'state', 'saved'), ('code', 'saved'))
        for query, existing in [(dict(good, state=['wrong']), None),
                                (dict(good, state=['state', 'state']), None),
                                (dict(good, error=['access_denied']), None),
                                (dict(good, client_id=['dynamic_agent_client']), None),
                                (dict(good, client_id=['different']), 'saved'),
                                (dict(state=['state'], code=['code']), None)]:
            with self.subTest(query=query), self.assertRaises(auth.AuthError):
                auth.validate_callback(query, 'state', existing)

    def validate(self, claims, nonce='nonce', previous=None, key=None):
        token = jwt.encode(claims, key or self.key, algorithm='RS256')
        key_client = SimpleNamespace(get_signing_key_from_jwt=lambda token: SimpleNamespace(key=self.key.public_key()))
        with patch.object(auth, 'auth_metadata', return_value={'jwks_uri': auth.ISSUER + '/keys'}), patch.object(jwt, 'PyJWKClient', return_value=key_client):
            return auth.validate_id_token(token, 'oaiapp_test', nonce, previous)

    def test_verified_identity_and_invalid_claims(self):
        claims = dict(iss=auth.ISSUER, aud='oaiapp_test', sub='user-1',
                      iat=int(time.time()), exp=int(time.time()) + 600, nonce='nonce')
        self.assertEqual(self.validate(claims)['sub'], 'user-1')
        for changes in [{'iss': 'https://untrusted.example'}, {'aud': 'wrong-client'},
                        {'exp': int(time.time()) - 60}, {'nonce': 'wrong-nonce'}]:
            with self.subTest(changes=changes), self.assertRaises(auth.AuthError):
                self.validate(dict(claims, **changes))
        with self.assertRaises(auth.AuthError):
            self.validate(claims, previous={'subject': 'other-user', 'issuer': auth.ISSUER})
        other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        with self.assertRaises(auth.AuthError):
            self.validate(claims, key=other_key)

    def test_metadata_endpoint_must_be_openai(self):
        for endpoint in ['http://auth.openai.com/keys', 'https://example.com/keys', 'https://auth.openai.com:8443/keys']:
            with self.subTest(endpoint=endpoint), self.assertRaises(auth.AuthError):
                auth.trusted_auth_url(endpoint)

    def test_sign_in_registration_pkce_and_exact_redirect(self):
        parameters = {}
        exchange = {}
        class FakeServer:
            def __init__(self, address, handler):
                self.handler = handler
                self.server_port = 54321
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def handle_request(self):
                handler = object.__new__(self.handler)
                query = urllib.parse.urlencode(dict(state=parameters['state'][0], code='test-code', client_id='oaiapp_issued'))
                handler.path = '/auth/callback?' + query
                handler.wfile = io.BytesIO()
                handler.send_response = lambda *args: None
                handler.send_header = lambda *args: None
                handler.end_headers = lambda: None
                handler.do_GET()
        def browser(url):
            parameters.update(urllib.parse.parse_qs(urllib.parse.urlparse(url).query))
            return True
        def token_request(url, form):
            exchange.update(form)
            return dict(access_token='new-access', refresh_token='new-refresh', id_token='id',
                        token_type='Bearer', expires_in=3600, scope=auth.SCOPES)
        identity = dict(iss=auth.ISSUER, sub='user-1', email='test@example.com')
        with patch.object(auth, 'HTTPServer', FakeServer), patch.object(auth.webbrowser, 'open', side_effect=browser), patch.object(auth, 'request_json', side_effect=token_request), patch.object(auth, 'validate_id_token', return_value=identity) as validate, patch('sys.stdout', new_callable=io.StringIO) as stdout:
            profile = auth.login()
        self.assertEqual(parameters['client_id'], ['dynamic_agent_client'])
        self.assertEqual(parameters['agent_name_hint'], ['VisionAssist'])
        self.assertIn(auth.PLAN_SCOPE, parameters['scope'][0])
        self.assertEqual(exchange['client_id'], 'oaiapp_issued')
        self.assertEqual(exchange['redirect_uri'], parameters['redirect_uri'][0])
        self.assertEqual(exchange['redirect_uri'], 'http://127.0.0.1:54321/auth/callback')
        digest = hashlib.sha256(exchange['code_verifier'].encode()).digest()
        self.assertEqual(parameters['code_challenge'][0], base64.urlsafe_b64encode(digest).rstrip(b'=').decode())
        self.assertEqual(validate.call_args.args[2], parameters['nonce'][0])
        self.assertEqual(profile['client_id'], 'oaiapp_issued')
        self.assertNotIn('new-access', stdout.getvalue())
        self.assertNotIn('new-refresh', stdout.getvalue())

    def test_scope_and_missing_login_block_inference(self):
        with self.assertRaises(auth.AuthError): auth.access_token()
        self.save_profile(scopes=['openid'])
        with patch.object(auth, 'request_json') as request, self.assertRaises(auth.AuthError):
            auth.access_token()
        request.assert_not_called()

    def test_refresh_rotates_saved_credentials(self):
        self.save_profile(expires_at=0)
        response = dict(access_token='rotated-access', refresh_token='rotated-refresh',
                        expires_in=3600, token_type='Bearer')
        with patch.object(auth, 'request_json', return_value=response) as request:
            self.assertEqual(auth.access_token(), 'rotated-access')
        form = request.call_args.args[1]
        self.assertEqual(form['client_id'], 'oaiapp_test')
        self.assertEqual(form['refresh_token'], 'test-refresh')
        self.assertNotIn('scope', form)
        saved = json.loads((self.storage / 'sessions.json').read_text())['profiles'][0]
        self.assertEqual(saved['refresh_token'], 'rotated-refresh')
        self.assertEqual(saved['scopes'], [auth.PLAN_SCOPE])

    def test_removed_scope_is_persisted_and_blocks_inference(self):
        self.save_profile(expires_at=0)
        response = dict(access_token='rotated-access', refresh_token='rotated-refresh',
                        expires_in=3600, token_type='Bearer', scope='openid')
        with patch.object(auth, 'request_json', return_value=response), self.assertRaises(auth.AuthError):
            auth.access_token()
        saved = json.loads((self.storage / 'sessions.json').read_text())['profiles'][0]
        self.assertEqual(saved['refresh_token'], 'rotated-refresh')
        self.assertNotIn(auth.PLAN_SCOPE, saved['scopes'])

    def test_failed_refresh_preserves_session(self):
        self.save_profile(expires_at=0)
        with patch.object(auth, 'request_json', side_effect=auth.AuthError('offline')), self.assertRaises(auth.AuthError):
            auth.access_token()
        saved = json.loads((self.storage / 'sessions.json').read_text())['profiles'][0]
        self.assertEqual(saved['refresh_token'], 'test-refresh')

    def test_lightweight_model_choice_requires_account_access(self):
        catalog = {'models': [{'slug': 'gpt-5.6-luna', 'visibility': 'list', 'input_modalities': ['text', 'image']},
                              {'slug': 'gpt-5.4-nano', 'visibility': 'list'}]}
        with patch.object(auth, 'request_json', return_value=catalog):
            self.assertEqual(auth.choose_model('token'), 'gpt-5.6-luna')
            self.assertEqual(auth.choose_model('token', 'gpt-5.4-nano'), 'gpt-5.4-nano')
            with self.assertRaises(auth.AuthError): auth.choose_model('token', 'unavailable')
        with patch.object(auth, 'request_json', return_value={'models': [{'slug': 'expensive-model', 'visibility': 'list'}]}), self.assertRaises(auth.AuthError):
            auth.choose_model('token')
        catalog['models'][0]['input_modalities'] = ['text']
        with patch.object(auth, 'request_json', return_value=catalog), self.assertRaises(auth.AuthError):
            auth.choose_model('token')

    def test_logout_preserves_registration_and_removes_tokens(self):
        self.save_profile(id_token='id')
        with patch.object(auth, 'auth_metadata', side_effect=auth.AuthError('offline')), patch('sys.stdout', new_callable=io.StringIO) as stdout:
            auth.logout()
        saved = json.loads((self.storage / 'sessions.json').read_text())['profiles'][0]
        self.assertEqual(saved['client_id'], 'oaiapp_test')
        self.assertNotIn('access_token', saved)
        self.assertNotIn('refresh_token', saved)
        self.assertNotIn('id_token', saved)
        self.assertIn('not confirmed', stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
