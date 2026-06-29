let scene, camera, renderer, controls;
let dnaGroup;
let tfComplex;
let particleSystem;
let currentLoopAmount = 0;
let targetLoopAmount = 0;
let autoRotate = true;
let currentSequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG";
let sphereMeshes = [];
let rungMeshes = [];
let backbone1Mesh, backbone2Mesh;

function init3DVisualizer() {
    const container = document.getElementById('dna-canvas-container');
    if (!container) return;

    // Clear previous canvas if any
    container.innerHTML = '';

    // Create Scene
    scene = new THREE.Scene();

    // Create Camera
    camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(0, 0, 18);

    // Create Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // Create Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 30;
    controls.minDistance = 5;
    controls.enableZoom = true;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight1.position.set(5, 10, 7);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x3b82f6, 0.5); // Blue highlight
    dirLight2.position.set(-5, -5, -5);
    scene.add(dirLight2);

    // DNA Group
    dnaGroup = new THREE.Group();
    scene.add(dnaGroup);

    // Build TF complex (hidden initially)
    const tfGeom = new THREE.SphereGeometry(1.5, 32, 32);
    const tfMat = new THREE.MeshBasicMaterial({
        color: 0xf59e0b, // Gold/amber
        transparent: true,
        opacity: 0.0,
        blending: THREE.AdditiveBlending
    });
    tfComplex = new THREE.Mesh(tfGeom, tfMat);
    tfComplex.position.set(0, 3, 0); // Center of the loop
    scene.add(tfComplex);

    // Build Particles
    initParticles();

    // Build initial DNA
    updateDNAModel(currentSequence);

    // Handle Window Resize
    window.addEventListener('resize', onWindowResize);

    // Start Animation Loop
    animate();
}

function initParticles() {
    const particleCount = 100;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    const color1 = new THREE.Color(0xf59e0b); // Gold
    const color2 = new THREE.Color(0x8b5cf6); // Purple

    for (let i = 0; i < particleCount; i++) {
        // Distribute particles in a sphere around (0, 3, 0)
        const radius = Math.random() * 2;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos((Math.random() * 2) - 1);

        positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
        positions[i * 3 + 1] = 3 + radius * Math.sin(phi) * Math.sin(theta);
        positions[i * 3 + 2] = radius * Math.cos(phi);

        // Mix gold and purple
        const mixedColor = color1.clone().lerp(color2, Math.random());
        colors[i * 3] = mixedColor.r;
        colors[i * 3 + 1] = mixedColor.g;
        colors[i * 3 + 2] = mixedColor.b;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Create a circular texture for soft particles
    const canvas = document.createElement('canvas');
    canvas.width = 16;
    canvas.height = 16;
    const ctx = canvas.getContext('2d');
    const grad = ctx.createRadialGradient(8, 8, 0, 8, 8, 8);
    grad.addColorStop(0, 'rgba(255, 255, 255, 1)');
    grad.addColorStop(1, 'rgba(255, 255, 255, 0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 16, 16);
    const texture = new THREE.CanvasTexture(canvas);

    const material = new THREE.PointsMaterial({
        size: 0.35,
        map: texture,
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        vertexColors: true,
        opacity: 0.0
    });

    particleSystem = new THREE.Points(geometry, material);
    scene.add(particleSystem);
}

function getFrame(t, loopAmount) {
    // Straight position: path along Y axis from -8 to 8
    const yStraight = (t - 0.5) * 16;
    const pStraight = new THREE.Vector3(0, yStraight, 0);
    
    // Loop position: circle in XY plane of radius 3
    const r = 3;
    const angle = t * Math.PI * 2 - Math.PI / 2;
    const pLoop = new THREE.Vector3(r * Math.cos(angle), r * Math.sin(angle) + r, 0);
    
    // Interpolate center position
    const center = new THREE.Vector3().lerpVectors(pStraight, pLoop, loopAmount);
    
    // Calculate tangent numerically
    const dt = 0.01;
    const tNext = Math.min(t + dt, 1.0);
    const tPrev = Math.max(t - dt, 0.0);
    
    const pNextStraight = new THREE.Vector3(0, (tNext - 0.5) * 16, 0);
    const pNextLoop = new THREE.Vector3(r * Math.cos(tNext * Math.PI * 2 - Math.PI / 2), r * Math.sin(tNext * Math.PI * 2 - Math.PI / 2) + r, 0);
    const pNext = new THREE.Vector3().lerpVectors(pNextStraight, pNextLoop, loopAmount);
    
    const pPrevStraight = new THREE.Vector3(0, (tPrev - 0.5) * 16, 0);
    const pPrevLoop = new THREE.Vector3(r * Math.cos(tPrev * Math.PI * 2 - Math.PI / 2), r * Math.sin(tPrev * Math.PI * 2 - Math.PI / 2) + r, 0);
    const pPrev = new THREE.Vector3().lerpVectors(pPrevStraight, pPrevLoop, loopAmount);
    
    const tangent = new THREE.Vector3().subVectors(pNext, pPrev).normalize();
    
    let up = new THREE.Vector3(0, 0, 1);
    if (Math.abs(tangent.dot(up)) > 0.99) {
        up.set(1, 0, 0);
    }
    const normal = new THREE.Vector3().crossVectors(tangent, up).normalize();
    const binormal = new THREE.Vector3().crossVectors(tangent, normal).normalize();
    
    return { center, normal, binormal };
}

function updateDNAModel(sequence) {
    currentSequence = sequence.toUpperCase();
    
    // Clear previous DNA meshes
    sphereMeshes.forEach(m => {
        dnaGroup.remove(m.mesh1);
        dnaGroup.remove(m.mesh2);
    });
    rungMeshes.forEach(m => {
        dnaGroup.remove(m.rung1);
        dnaGroup.remove(m.rung2);
    });
    if (backbone1Mesh) dnaGroup.remove(backbone1Mesh);
    if (backbone2Mesh) dnaGroup.remove(backbone2Mesh);
    
    sphereMeshes = [];
    rungMeshes = [];
    
    const numBases = Math.min(currentSequence.length, 60);
    if (numBases === 0) return;
    
    // Materials map for base pairs
    const colors = {
        'A': 0x00f0ff, // Cyan
        'T': 0xff9f00, // Orange
        'G': 0xbd00ff, // Purple
        'C': 0x00ff66, // Lime
        'N': 0x888888  // Gray
    };
    
    const sphereGeom = new THREE.SphereGeometry(0.22, 16, 16);
    
    for (let i = 0; i < numBases; i++) {
        const char = currentSequence[i] || 'N';
        const pairChar = getPairChar(char);
        
        const col1 = colors[char] || colors['N'];
        const col2 = colors[pairChar] || colors['N'];
        
        const mat1 = new THREE.MeshPhongMaterial({ color: col1, shininess: 80, specular: 0x555555 });
        const mat2 = new THREE.MeshPhongMaterial({ color: col2, shininess: 80, specular: 0x555555 });
        
        const s1 = new THREE.Mesh(sphereGeom, mat1);
        const s2 = new THREE.Mesh(sphereGeom, mat2);
        
        dnaGroup.add(s1);
        dnaGroup.add(s2);
        
        sphereMeshes.push({ mesh1: s1, mesh2: s2 });
        
        // Rungs (each rung is composed of two cylinders meeting at midpoint)
        const rungGeom = new THREE.CylinderGeometry(0.05, 0.05, 1, 8);
        const r1 = new THREE.Mesh(rungGeom, mat1);
        const r2 = new THREE.Mesh(rungGeom, mat2);
        
        dnaGroup.add(r1);
        dnaGroup.add(r2);
        rungMeshes.push({ rung1: r1, rung2: r2 });
    }
    
    // Generate backbone lines
    const lineMaterial1 = new THREE.LineBasicMaterial({ color: 0x3b82f6, linewidth: 2 });
    const lineMaterial2 = new THREE.LineBasicMaterial({ color: 0x8b5cf6, linewidth: 2 });
    
    const geom1 = new THREE.BufferGeometry();
    const geom2 = new THREE.BufferGeometry();
    
    const positions1 = new Float32Array(numBases * 3);
    const positions2 = new Float32Array(numBases * 3);
    
    geom1.setAttribute('position', new THREE.BufferAttribute(positions1, 3));
    geom2.setAttribute('position', new THREE.BufferAttribute(positions2, 3));
    
    backbone1Mesh = new THREE.Line(geom1, lineMaterial1);
    backbone2Mesh = new THREE.Line(geom2, lineMaterial2);
    
    dnaGroup.add(backbone1Mesh);
    dnaGroup.add(backbone2Mesh);
    
    // Initial compute
    updateDNAJoints();
}

function getPairChar(char) {
    if (char === 'A') return 'T';
    if (char === 'T') return 'A';
    if (char === 'G') return 'C';
    if (char === 'C') return 'G';
    return 'N';
}

function updateDNAJoints() {
    const numBases = Math.min(currentSequence.length, 60);
    if (numBases === 0) return;
    
    const R = 1.2; 
    const turns = 4.0; 
    
    const positions1 = backbone1Mesh.geometry.attributes.position.array;
    const positions2 = backbone2Mesh.geometry.attributes.position.array;
    
    for (let i = 0; i < numBases; i++) {
        const t = i / (numBases - 1);
        const { center, normal, binormal } = getFrame(t, currentLoopAmount);
        const theta = t * turns * Math.PI * 2;
        
        const offset1 = new THREE.Vector3().addScaledVector(normal, Math.cos(theta) * R).addScaledVector(binormal, Math.sin(theta) * R);
        const offset2 = new THREE.Vector3().addScaledVector(normal, -Math.cos(theta) * R).addScaledVector(binormal, -Math.sin(theta) * R);
        
        const p1 = new THREE.Vector3().addVectors(center, offset1);
        const p2 = new THREE.Vector3().addVectors(center, offset2);
        
        // Update Spheres
        const sPair = sphereMeshes[i];
        if (sPair) {
            sPair.mesh1.position.copy(p1);
            sPair.mesh2.position.copy(p2);
        }
        
        // Update Rungs
        const rPair = rungMeshes[i];
        if (rPair) {
            const mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
            
            // Rung 1: from p1 to mid
            rPair.rung1.position.copy(new THREE.Vector3().addVectors(p1, mid).multiplyScalar(0.5));
            rPair.rung1.scale.set(1, p1.distanceTo(mid), 1);
            rPair.rung1.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3().subVectors(mid, p1).normalize());
            
            // Rung 2: from mid to p2
            rPair.rung2.position.copy(new THREE.Vector3().addVectors(mid, p2).multiplyScalar(0.5));
            rPair.rung2.scale.set(1, mid.distanceTo(p2), 1);
            rPair.rung2.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3().subVectors(p2, mid).normalize());
        }
        
        // Update Backbones
        positions1[i * 3] = p1.x;
        positions1[i * 3 + 1] = p1.y;
        positions1[i * 3 + 2] = p1.z;
        
        positions2[i * 3] = p2.x;
        positions2[i * 3 + 1] = p2.y;
        positions2[i * 3 + 2] = p2.z;
    }
    
    backbone1Mesh.geometry.attributes.position.needsUpdate = true;
    backbone2Mesh.geometry.attributes.position.needsUpdate = true;
}

function animate() {
    requestAnimationFrame(animate);

    // Loop transition animation
    if (Math.abs(currentLoopAmount - targetLoopAmount) > 0.001) {
        currentLoopAmount += (targetLoopAmount - currentLoopAmount) * 0.06;
        updateDNAJoints();
    }

    // Auto rotate
    if (autoRotate) {
        dnaGroup.rotation.y += 0.004;
        if (particleSystem) {
            particleSystem.rotation.y -= 0.002;
        }
    }

    // Pulsing particles and TF complex
    if (currentLoopAmount > 0.01) {
        const time = Date.now() * 0.003;
        const tfScale = 1.0 + Math.sin(time) * 0.08;
        tfComplex.scale.set(tfScale, tfScale, tfScale);
        tfComplex.material.opacity = currentLoopAmount * 0.45 * (0.8 + Math.sin(time) * 0.15);
        
        particleSystem.material.opacity = currentLoopAmount * 0.95;
        
        const positions = particleSystem.geometry.attributes.position.array;
        const count = positions.length / 3;
        for (let i = 0; i < count; i++) {
            positions[i * 3 + 1] += Math.sin(time + i) * 0.002; 
        }
        particleSystem.geometry.attributes.position.needsUpdate = true;
    } else {
        tfComplex.material.opacity = 0;
        particleSystem.material.opacity = 0;
    }

    controls.update();
    renderer.render(scene, camera);
}

function onWindowResize() {
    const container = document.getElementById('dna-canvas-container');
    if (!container) return;
    
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function toggleRotation() {
    autoRotate = !autoRotate;
    const btn = document.getElementById('btn-rotate');
    if (autoRotate) {
        btn.innerText = 'Pause Rotation';
        btn.classList.add('active');
    } else {
        btn.innerText = 'Auto Rotate';
        btn.classList.remove('active');
    }
}

function resetVisualizerView() {
    if (controls) {
        controls.reset();
        camera.position.set(0, 0, 18);
        camera.lookAt(0, 0, 0);
    }
}

function updateVisualizerState(seq, isEnhancer, confidenceText) {
    updateDNAModel(seq);
    
    if (isEnhancer) {
        targetLoopAmount = 1.0;
        const stateLabel = document.getElementById('viz-state-label');
        stateLabel.innerText = "Enhancer (Active Loop)";
        stateLabel.className = "state-active";
        
        document.getElementById('viz-chromatin-status').innerHTML = 
            `<strong>Regulatory Element Detected!</strong> The BiLSTM model predicts chromatin looping with <strong>${confidenceText}</strong> confidence. Golden particles represent transcription factor engagement at the loop junction.`;
    } else {
        targetLoopAmount = 0.0;
        const stateLabel = document.getElementById('viz-state-label');
        stateLabel.innerText = "Non-Regulatory (Straight)";
        stateLabel.className = "state-nonreg";
        
        document.getElementById('viz-chromatin-status').innerHTML = 
            `<strong>Non-Regulatory Element.</strong> The sequence remains in a straight, resting double-helix conformation (prediction confidence: <strong>${confidenceText}</strong>). No chromatin loop is formed.`;
    }
}

// Tab Switching
function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    
    document.getElementById(tabId).classList.add('active');
    const btn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
    if (btn) btn.classList.add('active');
}

// Single Sequence Analysis
async function predictText() {
    const seq = document.getElementById('sequence-text').value.trim();
    if (!seq) return alert('Please enter a sequence.');
    
    const btn = document.querySelector('#text-input .primary-btn');
    const origText = btn.innerText;
    btn.innerText = 'Analyzing...';
    
    try {
        const response = await fetch('/predict_text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sequence: seq })
        });
        
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        
        const resBox = document.getElementById('text-result');
        const resClassEl = document.getElementById('res-class');
        
        resClassEl.innerText = data.prediction;
        document.getElementById('res-prob').innerText = data.confidence;
        
        const isEnhancer = data.prediction.includes('Enhancer');
        if (isEnhancer) {
            resClassEl.style.color = 'var(--success)';
        } else {
            resClassEl.style.color = '#cbd5e1';
        }
        resBox.classList.remove('hidden');
        
        updateVisualizerState(seq, isEnhancer, data.confidence);
    } catch (err) {
        alert(err.message);
    } finally {
        btn.innerText = origText;
    }
}

// File Drag & Drop
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const fileNameDisplay = document.getElementById('file-name');

if (dropArea) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults (e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
    });

    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        let dt = e.dataTransfer;
        let files = dt.files;
        fileInput.files = files;
        updateFileName();
    }
}

if (fileInput) {
    fileInput.addEventListener('change', updateFileName);
}

function updateFileName() {
    if(fileInput && fileInput.files.length > 0) {
        fileNameDisplay.innerText = fileInput.files[0].name;
    }
}

// Batch Dataset Processing
async function uploadDataset() {
    if(fileInput.files.length === 0) return alert('Please select a file.');
    
    const btn = document.querySelector('#csv-upload .primary-btn');
    const origText = btn.innerText;
    btn.innerText = 'Processing Batch...';
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    try {
        const response = await fetch('/predict_csv', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        
        const resBox = document.getElementById('csv-result');
        const batchAcc = document.getElementById('batch-acc');
        const batchStats = document.getElementById('batch-stats');
        const tbody = document.getElementById('results-body');
        
        if (data.accuracy) {
            batchAcc.innerText = data.accuracy;
            batchStats.innerText = `Processed ${data.total_processed} sequences with ground truth labels.`;
        } else {
            batchAcc.innerText = 'N/A';
            batchStats.innerText = `Processed ${data.total_processed} sequences (no ground truth found).`;
        }
        
        tbody.innerHTML = '';
        data.results.forEach((res, idx) => {
            const tr = document.createElement('tr');
            
            let statusClass = '';
            let actualHTML = `<td>--</td>`;
            if (res.actual) {
                const isCorrect = res.prediction === res.actual;
                statusClass = isCorrect ? 'correct' : 'incorrect';
                actualHTML = `<td class="${statusClass}">${res.actual}</td>`;
            }
            
            const isEnhancer = res.prediction === 'Enhancer';
            tr.innerHTML = `
                <td style="font-family: monospace; font-size: 0.85rem">${res.sequence_preview}</td>
                <td style="color: ${isEnhancer ? 'var(--success)' : '#94a3b8'}">${res.prediction}</td>
                <td>${res.confidence}</td>
                ${actualHTML}
            `;
            
            tr.addEventListener('click', () => {
                document.querySelectorAll('#results-table tbody tr').forEach(r => r.classList.remove('selected-row'));
                tr.classList.add('selected-row');
                updateVisualizerState(res.full_sequence, isEnhancer, res.confidence);
            });
            
            tbody.appendChild(tr);
            
            if (idx === 0) {
                tr.classList.add('selected-row');
                updateVisualizerState(res.full_sequence, isEnhancer, res.confidence);
            }
        });
        
        resBox.classList.remove('hidden');
    } catch (err) {
        alert(err.message);
    } finally {
        btn.innerText = origText;
    }
}

// Initializing on Page Load
document.addEventListener('DOMContentLoaded', () => {
    // Generate a random DNA sequence for initial visualization
    const bases = ['A', 'C', 'G', 'T'];
    let initialSeq = '';
    for (let i = 0; i < 50; i++) {
        initialSeq += bases[Math.floor(Math.random() * 4)];
    }
    
    init3DVisualizer();
    updateDNAModel(initialSeq);
});

// Expose controls to window
window.toggleRotation = toggleRotation;
window.resetVisualizerView = resetVisualizerView;
window.switchTab = switchTab;
