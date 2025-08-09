"""
Einfacher Test für EmbeddingManager
"""
from unittest.mock import Mock, patch
from rag_chatbot.embeddings.embedding_manager import EmbeddingManager


@patch('rag_chatbot.embeddings.embedding_manager.HuggingFaceEmbeddings')
@patch('rag_chatbot.embeddings.embedding_manager.Config')
def test_embedding_manager_loads_model(mock_config_class, mock_hf_embeddings):
    """Test dass das Embedding-Model geladen wird"""
    # Arrange
    mock_config = Mock()
    mock_config.EMBEDDING_MODEL = "test-model"
    mock_config_class.return_value = mock_config
    
    mock_model = Mock()
    mock_hf_embeddings.return_value = mock_model
    
    # Act
    manager = EmbeddingManager()
    
    # Assert
    assert manager.model is not None
    assert manager.get_model() == mock_model
    mock_hf_embeddings.assert_called_once_with(model_name="test-model")