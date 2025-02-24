// src/App.js

import React, { useState } from "react";
import API_BASE_URL from "./config";
import "./App.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [annotatedImage, setAnnotatedImage] = useState("");
  const [detections, setDetections] = useState([]);
  const [error, setError] = useState("");

  // Lida com a seleção do arquivo
  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  // Envia o arquivo para a API
  const handleUpload = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      alert("Por favor, selecione uma imagem!");
      return;
    }

    setLoading(true);
    setError("");
    setAnnotatedImage("");
    setDetections([]);

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      // Chamada à API de detecção
      const response = await fetch(`${API_BASE_URL}/detect/`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error("Erro ao processar a imagem");
      }
      const data = await response.json();
      setAnnotatedImage(`data:image/jpeg;base64,${data.annotated_image}`);
      setDetections(data.detections);
    } catch (err) {
      console.error("Erro:", err);
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <h1>Detecção de Objetos com YOLO</h1>
      <p>Selecione uma imagem e clique em "Enviar" para detectar objetos.</p>

      <form onSubmit={handleUpload} className="upload-form">
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button type="submit">Enviar para Detecção</button>
      </form>

      {loading && <div className="loader">Processando a imagem...</div>}
      {error && <div className="error">{error}</div>}

      {annotatedImage && (
        <div className="result">
          <h2>Imagem Anotada</h2>
          <img src={annotatedImage} alt="Imagem com detecções" />
        </div>
      )}

      {detections.length > 0 && (
        <div className="detections">
          <h2>Detecções</h2>
          <ul>
            {detections.map((det, index) => (
              <li key={index}>
                <strong>Classe:</strong> {det.class} <br />
                <strong>Confiança:</strong> {(det.confidence * 100).toFixed(2)}% <br />
                <strong>Caixa:</strong> [{det.box.join(", ")}]
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
