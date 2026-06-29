from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import numpy as np
import onnxruntime as ort

# Inline necessary constants from train_model.py to avoid importing torch/PyTorch
BASE_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
MAX_LEN = 1170

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load ONNX model weights
onnx_path = os.path.join(os.path.dirname(__file__), 'enhancer_model.onnx')
try:
    # Use CPU Execution Provider for serverless Vercel environment
    ort_sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    print("Successfully loaded ONNX model weights.")
except Exception as e:
    print(f"Warning: Could not load ONNX model from {onnx_path}: {e}")
    ort_sess = None

def predict_sequence(seq):
    seq = seq.upper()
    seq_encoded = np.full(MAX_LEN, 4, dtype=np.int64)
    for i, base in enumerate(seq):
        if i >= MAX_LEN: break
        seq_encoded[i] = BASE_TO_INT.get(base, 4)
    
    if ort_sess is None:
        print("ONNX session not initialized. Returning fallback prediction.")
        return 0.5
        
    try:
        input_name = ort_sess.get_inputs()[0].name
        output_name = ort_sess.get_outputs()[0].name
        
        # Batch of size 1: shape (1, MAX_LEN)
        batch = np.expand_dims(seq_encoded, axis=0)
        ort_outs = ort_sess.run([output_name], {input_name: batch})
        logits = ort_outs[0]
        
        # Apply sigmoid to convert logits to probabilities
        prob = 1 / (1 + np.exp(-logits))
        return float(prob[0])
    except Exception as e:
        print("Prediction error in ONNX runtime:", e)
        return 0.5

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_text', methods=['POST'])
def predict_text():
    data = request.get_json()
    sequence = data.get('sequence', '').strip()
    if not sequence:
        return jsonify({'error': 'No sequence provided'}), 400
    
    prob = predict_sequence(sequence)
    is_enhancer = prob > 0.5
    
    return jsonify({
        'prediction': 'Regulatory Element (Enhancer)' if is_enhancer else 'Non-Regulatory',
        'confidence': f"{(prob if is_enhancer else 1-prob) * 100:.2f}%",
        'raw_probability': prob
    })

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'})
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath, nrows=150)
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath, nrows=150)
        else:
            return jsonify({'error': 'Invalid file type. Please upload a CSV or XLSX file.'}), 400
            
        # Find combination of columns
        seq_col = None
        for col in df.columns:
            if 'seq' in col.lower() or 'sequence' in col.lower():
                seq_col = col
                break
        
        if not seq_col:
            # Fallback to the second column if similar to CDD data
            if 'sequence' in df.iloc[0].values or len(df.columns) >= 2:
                seq_col = df.columns[1]
            else:
                return jsonify({'error': 'Could not identify a Sequence column. Please ensure there is a column named "sequence".'}), 400
                
        class_col = None
        for col in df.columns:
            if 'class' in col.lower() or 'label' in col.lower():
                class_col = col
                break
                
        if not class_col:
            result_vals = [str(v).lower() for v in df.iloc[0].values]
            if 'class' in result_vals or 'label' in result_vals:
                class_col = df.columns[0]
                
        # Collect up to 100 valid sequences
        sequences_to_predict = []
        rows_to_process = []
        
        for idx, row in df.iterrows():
            if len(rows_to_process) >= 100:
                break
                
            seq = str(row[seq_col])
            # If the CSV includes headers inside the data occasionally
            if seq.lower() == 'sequence': continue 
            
            if len(seq) < 10: continue
            
            rows_to_process.append(row)
            sequences_to_predict.append(seq)
            
        # Perform batched ONNX prediction
        probs = []
        if sequences_to_predict and ort_sess is not None:
            encoded_list = []
            for seq in sequences_to_predict:
                seq = seq.upper()
                seq_encoded = np.full(MAX_LEN, 4, dtype=np.int64)
                for i, base in enumerate(seq):
                    if i >= MAX_LEN: break
                    seq_encoded[i] = BASE_TO_INT.get(base, 4)
                encoded_list.append(seq_encoded)
                
            try:
                input_name = ort_sess.get_inputs()[0].name
                output_name = ort_sess.get_outputs()[0].name
                
                # Batch of size N: shape (N, MAX_LEN)
                batch = np.array(encoded_list, dtype=np.int64)
                ort_outs = ort_sess.run([output_name], {input_name: batch})
                logits = ort_outs[0]
                
                # Apply sigmoid to convert logits to probabilities
                probs_arr = 1 / (1 + np.exp(-logits))
                probs = probs_arr.tolist()
            except Exception as e:
                print("Batch prediction error in ONNX runtime:", e)
                probs = [0.5] * len(sequences_to_predict)
        else:
            probs = [0.5] * len(sequences_to_predict)
                    
        # Construct results
        results = []
        correct = 0
        total = 0
        
        for i, row in enumerate(rows_to_process):
            seq = sequences_to_predict[i]
            prob = probs[i]
            pred_label = 1 if prob > 0.5 else 0
            
            res_dict = {
                'sequence_preview': seq[:30] + '...',
                'full_sequence': seq,
                'prediction': 'Enhancer' if pred_label == 1 else 'Non-Enhancer',
                'confidence': f"{(prob if pred_label == 1 else 1-prob) * 100:.2f}%"
            }
            
            if class_col and not pd.isna(row[class_col]):
                try: # try parsing label
                    actual_label = int(float(row[class_col]))
                    res_dict['actual'] = 'Enhancer' if actual_label == 1 else 'Non-Enhancer'
                    if pred_label == actual_label:
                        correct += 1
                    total += 1
                except:
                    pass
            results.append(res_dict)
            
        accuracy = None
        if total > 0:
            accuracy = f"{(correct / total) * 100:.2f}%"
            
        return jsonify({
            'results': results,
            'accuracy': accuracy,
            'total_processed': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
