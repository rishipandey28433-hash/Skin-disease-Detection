import os
import unittest
from io import BytesIO
from app import app


class TestSkinAPI(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

    def test_index_redirects_when_not_logged_in(self):
        rv = self.client.get('/')
        self.assertEqual(rv.status_code, 302)
        self.assertIn('/login', rv.location)

    def test_login_correct_credentials(self):
        rv = self.client.post('/login', data=dict(
            username='admin',
            password='admin123'
        ), follow_redirects=True)
        self.assertEqual(rv.status_code, 200)
        self.assertTrue(b'admin' in rv.data or b'Skin' in rv.data or b'Upload' in rv.data)

    def test_login_wrong_credentials(self):
        rv = self.client.post('/login', data=dict(
            username='admin',
            password='wrongpassword'
        ), follow_redirects=True)
        self.assertEqual(rv.status_code, 200)
        self.assertIn(b'Invalid credentials', rv.data)

    def test_health_endpoint(self):
        rv = self.client.get('/health')
        self.assertEqual(rv.status_code, 200)
        data = rv.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("timestamp", data)

    def test_predict_without_file(self):
        self.client.post('/login', data=dict(
            username='admin',
            password='admin123'
        ), follow_redirects=True)

        rv = self.client.post('/predict', data={})
        self.assertEqual(rv.status_code, 200)
        self.assertTrue(b'No file uploaded' in rv.data or b'No file' in rv.data or b'invalid_input' in rv.data or b'Invalid' in rv.data)

    def test_predict_invalid_file_type(self):
        self.client.post('/login', data=dict(
            username='admin',
            password='admin123'
        ), follow_redirects=True)

        data = {
            'file': (BytesIO(b'some text content'), 'test.txt')
        }

        rv = self.client.post('/predict', data=data, content_type='multipart/form-data')
        self.assertEqual(rv.status_code, 200)
        self.assertTrue(b'Invalid file type' in rv.data or b'invalid' in rv.data.lower())


if __name__ == "__main__":
    unittest.main()
