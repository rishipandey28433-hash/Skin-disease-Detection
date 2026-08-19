import os
import unittest
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock

from pipeline import SkinAnalysisPipeline
from dataset_config import CLASS_NAMES, get_disease_info, get_risk_percentage
from utils.preprocess import preprocess_image


class TestSkinPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

        # Create valid image
        self.temp_image = str(self.tmp_path / "valid_image.jpg")
        img = Image.new('RGB', (224, 224), color='red')
        img.save(self.temp_image)

        # Create corrupt image
        self.corrupt_image = str(self.tmp_path / "corrupt.jpg")
        with open(self.corrupt_image, 'wb') as f:
            f.write(b"not an image")

        # Create small image
        self.small_image = str(self.tmp_path / "small.jpg")
        img = Image.new('RGB', (10, 10), color='red')
        img.save(self.small_image)

        # Mock model
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.array([[0.1, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]])

        self.pipeline = SkinAnalysisPipeline(self.mock_model, CLASS_NAMES)

    def tearDown(self):
        self.temp_dir.cleanup()

    # 1. Image validation tests
    def test_valid_image_path(self):
        is_valid, msg = self.pipeline._validate_image(self.temp_image)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")

    def test_non_existent_file(self):
        is_valid, msg = self.pipeline._validate_image("non_existent_file.jpg")
        self.assertFalse(is_valid)
        self.assertIn("does not exist", msg.lower())

    def test_invalid_extension(self):
        img_path = self.tmp_path / "test.txt"
        with open(img_path, "w") as f:
            f.write("test data")
        is_valid, msg = self.pipeline._validate_image(str(img_path))
        self.assertFalse(is_valid)
        self.assertTrue("unsupported" in msg.lower() or "invalid" in msg.lower())

    def test_corrupt_file(self):
        is_valid, msg = self.pipeline._validate_image(self.corrupt_image)
        self.assertFalse(is_valid)
        self.assertTrue("corrupted" in msg.lower() or "invalid image" in msg.lower() or "unreadable" in msg.lower())

    # 2. Quality check tests
    def test_quality_check(self):
        with patch('pipeline.validate_skin_image') as mock_validate:
            mock_validate.return_value = {"score": 85.0, "status": "good"}
            result = self.pipeline._check_quality(self.temp_image)
            self.assertIn("score", result)
            self.assertEqual(result["score"], 85.0)

    def test_small_image_validation(self):
        is_valid, msg = self.pipeline._validate_image(self.small_image)
        self.assertFalse(is_valid)
        self.assertIn("too small", msg.lower())

    # 3. Pipeline result structure tests
    def test_pipeline_result_structure(self):
        with patch.object(self.pipeline, '_check_quality', return_value={"score": 85.0, "status": "good"}):
            with patch.object(self.pipeline, '_check_human_skin', return_value=("human_skin", "Dermatology photo")):
                result = self.pipeline.analyze(self.temp_image)

                self.assertIn("status", result)
                self.assertIn("condition", result)
                self.assertIn("confidence", result)
                self.assertIn("message", result)
                self.assertIn("medical_disclaimer", result)

                valid_statuses = ["healthy", "disease", "non_disease_condition", "invalid_input", "low_quality", "uncertain"]
                self.assertIn(result["status"], valid_statuses)
                self.assertGreater(len(result["medical_disclaimer"]), 0)

    # 4. Dataset config tests
    def test_class_names_count(self):
        self.assertEqual(len(CLASS_NAMES), 8)

    def test_classes_have_display_names(self):
        from dataset_config import CLASS_DISPLAY_NAMES
        for c in CLASS_NAMES:
            self.assertIn(c, CLASS_DISPLAY_NAMES)

    def test_classes_have_descriptions(self):
        from dataset_config import CLASS_DESCRIPTIONS
        for c in CLASS_NAMES:
            self.assertIn(c, CLASS_DESCRIPTIONS)

    def test_get_disease_info(self):
        info = get_disease_info("mel")
        self.assertIsInstance(info, dict)
        self.assertIn("id", info)
        self.assertIn("name", info)
        self.assertIn("description", info)
        self.assertIn("symptoms", info)
        self.assertIn("precautions", info)
        self.assertIn("risk_level", info)
        self.assertIn("category", info)

    def test_get_risk_percentage(self):
        risk_mel = get_risk_percentage("mel", 0.9)
        risk_healthy = get_risk_percentage("healthy", 0.9)
        self.assertEqual(risk_healthy, 0.0)
        self.assertGreater(risk_mel, 70.0)

    # 5. Preprocessing tests
    def test_preprocess_image(self):
        processed = preprocess_image(self.temp_image)

        self.assertEqual(processed.shape, (1, 224, 224, 3))
        self.assertEqual(processed.dtype, np.float32)

        self.assertGreaterEqual(np.min(processed), 0.0)
        self.assertLessEqual(np.max(processed), 255.0)


if __name__ == "__main__":
    unittest.main()
