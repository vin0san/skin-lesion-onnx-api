import io
import pytest
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
from app.inference import SkinInference

# Preprocessing tests
@pytest.fixture
def valid_image_bytes() -> bytes:
    img = Image.new('RGB', (256, 256), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    return img_bytes.getvalue()

@pytest.fixture
def invalid_image_bytes() -> bytes:
    return b'\x89INVALID_IMAGE_DATA\x00\x00'

@pytest.fixture
def empty_bytes() -> bytes:
    return b''

class TestPreprocessing:
    @patch('app.inference.ort.InferenceSession')
    def test_valid_jpeg_preprocessing(self, mock_session, valid_image_bytes):
        mock_instance = MagicMock()
        mock_instance.get_inputs.return_value = [Mock(name='input_image')]
        mock_instance.get_outputs.return_value = [Mock(name='logits')]
        mock_session.return_value = mock_instance
        
        inference = SkinInference('dummy.onnx')
        processed = inference.preprocess(valid_image_bytes)
        
        assert processed.shape == (1, 3, 300, 300)
        assert processed.dtype == 'float32'
    
    @patch('app.inference.ort.InferenceSession')
    def test_invalid_image_bytes_raises_error(self, mock_session, invalid_image_bytes):
        mock_instance = MagicMock()
        mock_instance.get_inputs.return_value = [Mock(name='input_image')]
        mock_instance.get_outputs.return_value = [Mock(name='logits')]
        mock_session.return_value = mock_instance
        
        inference = SkinInference('dummy.onnx')
        
        with pytest.raises(ValueError, match="Failed to decode image"):
            inference.preprocess(invalid_image_bytes)
    
    @patch('app.inference.ort.InferenceSession')
    def test_empty_bytes_raises_error(self, mock_session, empty_bytes):
        mock_instance = MagicMock()
        mock_instance.get_inputs.return_value = [Mock(name='input_image')]
        mock_instance.get_outputs.return_value = [Mock(name='logits')]
        mock_session.return_value = mock_instance
        
        inference = SkinInference('dummy.onnx')
        
        with pytest.raises(ValueError, match="Image is empty"):
            inference.preprocess(empty_bytes)

class TestInference:
    @patch('app.inference.ort.InferenceSession')
    def test_predict_returns_sorted_results(self, mock_session, valid_image_bytes):
        mock_instance = MagicMock()
        mock_instance.get_inputs.return_value = [Mock(name='input_image')]
        mock_instance.get_outputs.return_value = [Mock(name='logits')]
        
        import numpy as np
        logits = np.array([[1.0, 0.5, 2.0, 0.1, -0.5, 3.0, 0.2, 0.0, -1.0]], dtype=np.float32)
        mock_instance.run.return_value = [logits]
        mock_session.return_value = mock_instance
        
        inference = SkinInference('dummy.onnx')
        predictions = inference.predict(valid_image_bytes)
        
        assert len(predictions) == 9
        confidences = [p[1] for p in predictions]
        assert confidences == sorted(confidences, reverse=True)
        assert 0.95 < sum(confidences) < 1.05
    
    @patch('app.inference.ort.InferenceSession')
    def test_predict_with_metadata(self, mock_session, valid_image_bytes):
        mock_instance = MagicMock()
        mock_instance.get_inputs.return_value = [Mock(name='input_image')]
        mock_instance.get_outputs.return_value = [Mock(name='logits')]
        
        import numpy as np
        logits = np.array([[0.1, 0.15, 0.12, 0.1, 0.08, 0.2, 0.13, 0.11, 0.09]], dtype=np.float32)
        mock_instance.run.return_value = [logits]
        mock_session.return_value = mock_instance
        
        inference = SkinInference('dummy.onnx')
        result = inference.predict_with_metadata(valid_image_bytes)
        
        assert "prediction" in result
        assert "confidence" in result
        assert "is_uncertain" in result
        assert "all_predictions" in result

# Sync API tests (skip async for now)
class TestAPISync:
    def test_api_imports(self):
        """Smoke test: verify API can be imported"""
        try:
            from app.main import app
            assert app is not None
        except Exception as e:
            pytest.fail(f"Failed to import app: {e}")