import streamlit as st
from PIL import Image
import numpy as np
import cv2
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from typing import Tuple, List
import warnings
import io

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. YOUR FEATURE EXTRACTOR CODE (No changes)
# ============================================================================

class FRQI_QHED_Extractor:
    """
    Quantum-enhanced image feature extractor using:
    - FRQI (Flexible Representation of Quantum Images) encoding
    - QHED (Quantum Hadamard Edge Detection)
    """

    def __init__(self, image_size=16, edge_strength=0.4, max_qubits=8):
        self.image_size = image_size
        self.edge_strength = edge_strength
        self.max_qubits = max_qubits
        self.n_position_qubits = int(np.ceil(np.log2(image_size * image_size)))
        if self.n_position_qubits > max_qubits:
            print(f"⚠ Warning: {self.n_position_qubits} qubits needed, limiting to {max_qubits}")
            self.n_position_qubits = max_qubits
        self.simulator = AerSimulator()

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_norm)
        resized = cv2.resize(enhanced, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_LANCZOS4)
        return resized

    def create_frqi_circuit(self, pixel_angles: np.ndarray) -> QuantumCircuit:
        n_pixels = len(pixel_angles)
        n_qubits = self.n_position_qubits + 1
        qr = QuantumRegister(n_qubits, 'q')
        qc = QuantumCircuit(qr)
        for i in range(self.n_position_qubits):
            qc.h(i)
        for pos in range(min(n_pixels, 2**self.n_position_qubits)):
            angle = pixel_angles[pos]
            if angle != 0:
                qc.ry(angle, self.n_position_qubits)
        return qc

    def frqi_encode(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img_flat = image.flatten()
        angles = (img_flat / 255.0) * (np.pi / 2)
        patch_size = min(2**self.n_position_qubits, len(angles))
        state_0_full = np.zeros_like(img_flat, dtype=np.float64)
        state_1_full = np.zeros_like(img_flat, dtype=np.float64)

        for start_idx in range(0, len(angles), patch_size):
            end_idx = min(start_idx + patch_size, len(angles))
            patch_angles = angles[start_idx:end_idx]
            qc = self.create_frqi_circuit(patch_angles)
            statevec = Statevector(qc)
            amplitudes = statevec.data
            n_states = len(amplitudes)
            half = n_states // 2
            patch_state_0 = np.abs(amplitudes[:half])[:len(patch_angles)]
            patch_state_1 = np.abs(amplitudes[half:])[:len(patch_angles)]
            state_0_full[start_idx:end_idx] = patch_state_0
            state_1_full[start_idx:end_idx] = patch_state_1

        state_0 = state_0_full.reshape(image.shape)
        state_1 = state_1_full.reshape(image.shape)
        norm = np.sqrt(state_0**2 + state_1**2)
        state_0 = state_0 / (norm + 1e-10)
        state_1 = state_1 / (norm + 1e-10)
        return state_0, state_1

    def qhed_circuit(self, n_qubits: int) -> QuantumCircuit:
        qr = QuantumRegister(n_qubits, 'q')
        qc = QuantumCircuit(qr)
        for i in range(n_qubits):
            qc.h(i)
        return qc

    def qhed_edges(self, quantum_state: np.ndarray) -> np.ndarray:
        sobelx = cv2.Sobel(quantum_state, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(quantum_state, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = np.power(edges, 1.2)
        edges = cv2.normalize(edges, None, 0, 1, cv2.NORM_MINMAX)
        return edges

    def extract_edge_statistics(self, edges: np.ndarray) -> np.ndarray:
        stats = np.array([
            np.mean(edges),
            np.std(edges),
            np.median(edges),
            np.percentile(edges, 25),
            np.percentile(edges, 75),
            np.min(edges),
            np.max(edges),
            np.sum(edges > 0.5) / edges.size,
            np.sum(edges > 0.3) / edges.size,
            np.sum(edges > 0.1) / edges.size
        ], dtype=np.float32)
        return stats

    def extract_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        preprocessed = self.preprocess(image)
        state_0, state_1 = self.frqi_encode(preprocessed)
        edges_0 = self.qhed_edges(state_0)
        edges_1 = self.qhed_edges(state_1)
        quantum_edges = np.sqrt(edges_0**2 + edges_1**2)
        quantum_edges = cv2.normalize(quantum_edges, None, 0, 1, cv2.NORM_MINMAX)
        original_norm = preprocessed / 255.0
        combined = ((1 - self.edge_strength) * original_norm +
                      self.edge_strength * quantum_edges)
        combined = cv2.normalize(combined, None, 0, 1, cv2.NORM_MINMAX)
        quantum_edges_2ch = np.stack([edges_0, edges_1], axis=0)
        edge_stats = self.extract_edge_statistics(quantum_edges)
        return combined, quantum_edges_2ch, edge_stats

# ============================================================================
# 2. [!!!] LOAD YOUR CLASSIFIER MODEL HERE [!!!]
# ============================================================================

# You MUST replace this placeholder with your real model
@st.cache_resource
def get_classifier():
    """
    This is a DUMMY classifier.
    Replace this with code to load your REAL trained model.
    (See instructions above for scikit-learn or PyTorch)
    """
    def dummy_classifier(stats):
        # 'stats' is your 10-feature vector from 'extract_edge_statistics'
        # We'll make a simple dummy rule:
        # If mean edge intensity (stats[0]) > 0.15, guess "fractured"
        
        # Simulate prediction (0 for NORMAL, 1 for FRACTURED)
        if stats[0] > 0.15:
            prediction = 1
            confidence = 0.88 # Dummy confidence
        else:
            prediction = 0
            confidence = 0.92 # Dummy confidence
            
        return prediction, confidence

    # Return the dummy function
    return dummy_classifier

# ============================================================================
# 3. YOUR STREAMLIT WEB APP INTERFACE (Redesigned)
# ============================================================================

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Quantum Fracture Detector")

# --- Custom CSS for X-Ray/Medical Theme ---
st.markdown("""
    <style>
    /* Dark theme with blue/teal accents */
    .stApp {
        background-color: #0d1117; /* GitHub dark mode background */
        color: #c9d1d9; /* GitHub dark mode text */
    }
    
    /* Title */
    .st-emotion-cache-10trblm { 
        color: #58a6ff; /* Bright blue for title */
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #c9d1d9;
    }
    
    /* Main container */
    .st-emotion-cache-z5fcl4 {
        background-color: #161b22;
        border-radius: 10px;
        padding: 2rem;
    }
    
    /* Result Boxes */
    .result-box {
        border: 2px solid;
        border-radius: 10px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
    }
    .result-box h2 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .result-box h3 {
        margin: 0;
        font-size: 1.2rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* Red for Fractured */
    .fractured {
        border-color: #e74c3c;
        color: #e74c3c;
        box-shadow: 0 0 20px rgba(231, 76, 60, 0.5);
    }
    
    /* Green for Normal */
    .normal {
        border-color: #2ecc71;
        color: #2ecc71;
        box-shadow: 0 0 20px rgba(46, 204, 113, 0.5);
    }

    /* Image captions */
    .stImage > figcaption {
        color: #8b949e !important;
    }
    
    </style>
""", unsafe_allow_html=True)


# --- App Title ---
st.title("🔬 Quantum-Enhanced Fracture Detector")
st.write("Upload an X-ray image to run the Quantum Edge Detection (FRQI-QHED) and a classification model to predict fractures.")

# --- Load Models ---
# Use st.cache_resource to load models only once
@st.cache_resource
def get_extractor():
    extractor = FRQI_QHED_Extractor(image_size=256, edge_strength=0.4, max_qubits=8)
    return extractor

extractor = get_extractor()
classifier = get_classifier() # This gets your DUMMY classifier

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 1. Load the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    
    # 2. Create side-by-side columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Original Image")
        st.image(image_np, caption="Uploaded X-Ray", use_column_width=True)

    with col2:
        st.header("Analysis Result")
        
        # 3. Run feature extraction
        with st.spinner("Running quantum feature extractor..."):
            combined_output, _, edge_stats = extractor.extract_features(image_np)
        
        st.image(combined_output, caption="Quantum-Enhanced Edge Features", use_column_width=True, clamp=True)
        
        # 4. Run classification
        with st.spinner("Running classification model..."):
            # 'edge_stats' is the 10-feature vector for your classifier
            prediction_index, confidence = classifier(edge_stats) # Calls your (dummy) model

        # 5. Display the final prediction
        st.subheader("Final Diagnosis")
        if prediction_index == 1:
            st.markdown(f'<div class="result-box fractured"><h2>PREDICTION: FRACTURED</h2><h3>Confidence: {confidence*100:.1f}%</h3></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-box normal"><h2>PREDICTION: NORMAL</h2><h3>Confidence: {confidence*100:.1f}%</h3></div>', unsafe_allow_html=True)

else:
    st.info("Please upload an image to begin processing.")