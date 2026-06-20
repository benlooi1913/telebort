/**
 * Smart Vision Camera - AI-Powered Real-Time Vision Analysis
 * Using Hugging Face Transformers.js and SmolVLM model
 *
 * STUDENT TODOs: Look for "TODO" comments throughout this file
 * There are 4 main areas you need to complete
 */

import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@latest";

// Configure environment
env.allowLocalModels = false;
env.allowRemoteModels = true;

// ============================================================================
// DOM Elements
// ============================================================================
const video = document.getElementById('videoElement');
const canvas = document.getElementById('canvasElement');
const startBtn = document.getElementById('startButton');
const stopBtn = document.getElementById('stopButton');
const responseArea = document.getElementById('responseArea');
const instructionText = document.getElementById('instructionText');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');
const loadingOverlay = document.getElementById('loadingOverlay');
const processingIndicator = document.getElementById('processingIndicator');

// Challenge DOM Elements
const captureBtn = document.getElementById('captureButton');
const historyList = document.getElementById('historyList');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const confidenceSection = document.getElementById('confidenceSection');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceScoreEl = document.getElementById('confidenceScore');

// ============================================================================
// Global State
// ============================================================================
let model = null;
let stream = null;
let isProcessing = false;
let processingInterval = null;
const PROCESSING_DELAY = 3000; // Process every 3 seconds
let isModelReady = false;
let isCameraReady = false;

// Challenge 2: Response history state
const responseHistory = [];
const MAX_HISTORY = 10;

// ============================================================================
// TODO 1: Model Initialization (MEDIUM DIFFICULTY)
// ============================================================================
/**
 * Initialize the AI vision model
 *
 * INSTRUCTIONS:
 * 1. The model ID should be: "HuggingFaceTB/SmolVLM-500M-Instruct"
 * 2. Replace the ______ with the correct model ID
 * 3. The pipeline type is "image-to-text" (already provided)
 *
 * HINT: Look at the project description for the exact model name
 * EXPECTED RESULT: Loading overlay should disappear when model is ready
 */
async function initializeModel() {
    try {
        updateStatus('Loading AI model...', false);

        // TODO 1: Replace ______ with the correct model ID
        // Using 256M model - optimized for speed on slow connections
        const modelId = "HuggingFaceTB/SmolVLM-256M-Instruct";

        console.log('Loading model:', modelId);
        console.log('⏳ On slow internet: This may take 3-5 minutes first load, then cached...');
        console.log('💡 Tip: Keep this tab open while loading, close other apps for better speed');
        
        // Set timeout to detect if loading is stuck
        const timeoutPromise = new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Model loading timeout - check internet connection')), 5 * 60 * 1000)
        );

        let lastProgressTime = Date.now();
        let lastStatus = '';
        let lastLoggedPercent = null;
        
        const loadPromise = pipeline("image-to-text", modelId, {
            device: "auto", // Auto-detect: WebGPU, then CPU
            progress_callback: (progress) => {
                lastProgressTime = Date.now();
                const status = progress.status;
                const rawProgress = progress.progress || 0;
                const normalizedProgress = rawProgress > 1 ? rawProgress : rawProgress * 100;
                const percent = Math.min(100, Math.max(0, Math.round(normalizedProgress)));

                if (status === 'downloading' || status === 'progress_total') {
                    if (percent !== lastLoggedPercent || status !== lastStatus) {
                        lastLoggedPercent = percent;
                        lastStatus = status;
                        const message = `Downloading model... ${percent}%`;
                        console.log(`📥 ${message}`);
                        updateStatus(message, false);
                    }
                } else if (status === 'progress') {
                    if (lastStatus !== 'progress') {
                        lastStatus = 'progress';
                        const message = 'Processing model...';
                        console.log(`⚙️ ${message}`);
                        updateStatus(message, false);
                    }
                } else if (status === 'done') {
                    lastStatus = 'done';
                    const message = 'Model download complete. Finalizing startup...';
                    console.log(`✅ ${message}`);
                    updateStatus(message, false);
                }
            }
        });

        model = await Promise.race([loadPromise, timeoutPromise]);
        isModelReady = true;

        console.log('✅ Model loaded successfully!');
        console.log('📦 Model cached in browser - next reload will be instant!');
        updateStatus('Model loaded. Waiting for camera...', false);
        finalizeInitialization();

    } catch (error) {
        console.error('❌ Error loading model:', error);
        console.error('Error details:', error.message);
        
        // Provide helpful debugging info
        if (error.message.includes('timeout')) {
            updateStatus('Loading timeout - check internet connection', false);
            alert('Model loading took too long. Check your internet connection and try refreshing the page.');
        } else if (error.message.includes('CORS') || error.message.includes('network')) {
            updateStatus('Network error - check internet connection', false);
            alert('Network error loading model. Check your internet connection and try again.');
        } else {
            updateStatus('Error loading model: ' + error.message, false);
            alert('Failed to load AI model.\n\nError: ' + error.message + '\n\nCheck the browser console (F12) for more details.');
        }
    }
}

// ============================================================================
// TODO 2: Camera Setup (EASY DIFFICULTY)
// ============================================================================
/**
 * Initialize camera and request permissions
 *
 * INSTRUCTIONS:
 * 1. Replace ______ with "getUserMedia"
 * 2. This is the standard browser API for camera access
 *
 * HINT: The MediaDevices API has a method called getUserMedia
 * EXPECTED RESULT: Browser should ask for camera permission
 */
async function initializeCamera() {
    try {
        updateStatus('Requesting camera access...', false);

        // TODO 2: Replace ______ with the correct method name
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'user',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });

        video.srcObject = stream;
        console.log('Camera initialized successfully!');
        updateStatus('Camera ready', true);

    } catch (error) {
        console.error('Error accessing camera:', error);
        updateStatus('Camera access denied', false);
        alert('Please allow camera access to use this app.');
        return;
    }

    isCameraReady = true;
    console.log('✅ Camera initialized successfully!');
    finalizeInitialization();
}

function finalizeInitialization() {
    if (isModelReady && isCameraReady) {
        updateStatus('Ready to start', true);
        startBtn.disabled = false;
        loadingOverlay.classList.add('hidden');
    }
}

// ============================================================================
// TODO 3: Image Processing (HARD DIFFICULTY)
// ============================================================================
/**
 * Process the current video frame and get AI response
 *
 * INSTRUCTIONS:
 * 1. Draw the video frame to canvas (line is provided as hint)
 * 2. Convert canvas to blob
 * 3. Send to AI model with instruction
 * 4. Display response
 *
 * HINTS:
 * - canvas.toBlob() is async and uses a callback
 * - model() expects an image and a prompt object
 * - The prompt should use the instruction from the input field
 *
 * EXPECTED RESULT: AI should analyze image and respond based on instruction
 */
async function processFrame() {
    if (!model || !video.videoWidth || isProcessing) return;

    try {
        isProcessing = true;
        processingIndicator.style.display = 'flex';

        // Set canvas size to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // TODO 3a: Draw video frame to canvas
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);

        // TODO 3b: Convert canvas to blob (image data)
        // HINT: canvas.toBlob() takes a callback function
        canvas.toBlob(async (blob) => {
            try {
                // TODO 3c: Get the instruction from the input field
                const instruction = instructionText.value || "Describe what you see";

                // Challenge 4: Append confidence score request to the prompt
                const enhancedInstruction = instruction +
                    ' At the end of your response, write "Confidence: X/10" where X is your confidence level from 1 to 10.';

                // TODO 3d: Send to AI model
                // Reduced max_new_tokens for faster responses on slow internet
                const result = await model(blob, {
                    prompt: enhancedInstruction,
                    max_new_tokens: 80,
                });

                const fullText = result[0].generated_text;

                // Challenge 4: Extract confidence score and strip it from displayed text
                const confidence = extractConfidence(fullText);
                const displayText = fullText.replace(/confidence:\s*\d+\s*\/\s*10\.?/gi, '').trim();

                // Display response and confidence
                displayResponse(displayText);
                displayConfidence(confidence);

            } catch (error) {
                console.error('Error processing image:', error);
                displayResponse('Error: ' + error.message);
            } finally {
                isProcessing = false;
                processingIndicator.style.display = 'none';
            }
        }, 'image/jpeg', 0.95);

    } catch (error) {
        console.error('Error in processFrame:', error);
        isProcessing = false;
        processingIndicator.style.display = 'none';
    }
}

// ============================================================================
// TODO 4: Control Functions (EASY DIFFICULTY)
// ============================================================================
/**
 * Start and stop continuous processing
 *
 * INSTRUCTIONS:
 * 1. Set isProcessing to correct boolean values
 * 2. true means processing is active
 * 3. false means processing is stopped
 *
 * EXPECTED RESULT: Start button begins analysis, Stop button pauses it
 */
function startProcessing() {
    if (!model) {
        alert('Model not loaded yet. Please wait...');
        return;
    }

    // TODO 4a: Set isProcessing to ______
    isProcessing = true;

    startBtn.disabled = true;
    stopBtn.disabled = false;
    updateStatus('Analyzing...', true);

    // Process first frame immediately
    processFrame();

    // Then process every PROCESSING_DELAY milliseconds
    processingInterval = setInterval(processFrame, PROCESSING_DELAY);
}

function stopProcessing() {
    // TODO 4b: Set isProcessing to ______
    isProcessing = false;

    if (processingInterval) {
        clearInterval(processingInterval);
        processingInterval = null;
    }

    startBtn.disabled = false;
    stopBtn.disabled = true;
    updateStatus('Stopped', true);
    processingIndicator.style.display = 'none';
}

// ============================================================================
// Helper Functions (Already Complete - No TODOs here!)
// ============================================================================

function displayResponse(text) {
    // Clear previous content
    responseArea.textContent = '';

    // Create response paragraph
    const responsePara = document.createElement('p');
    responsePara.className = 'response-text';
    responsePara.textContent = text;

    // Create instruction paragraph
    const instructionPara = document.createElement('p');
    instructionPara.className = 'small-text';
    instructionPara.style.marginTop = '10px';
    instructionPara.style.color = 'var(--text-secondary)';
    instructionPara.textContent = `Instruction: "${instructionText.value}"`;

    // Append to response area
    responseArea.appendChild(responsePara);
    responseArea.appendChild(instructionPara);

    // Challenge 2: Save to history
    addToHistory(text, instructionText.value);
}

function updateStatus(message, isActive) {
    statusText.textContent = message;
    statusIndicator.className = 'status-dot ' + (isActive ? 'active' : 'inactive');
}

function cleanup() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    if (processingInterval) {
        clearInterval(processingInterval);
    }
}

// ============================================================================
// Challenge 1: Preset Prompt Buttons
// ============================================================================
document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        instructionText.value = btn.dataset.prompt;
        instructionText.focus();
    });
});

// ============================================================================
// Challenge 2: Response History
// ============================================================================
function addToHistory(response, instruction) {
    responseHistory.unshift({ response, instruction, timestamp: new Date().toLocaleTimeString() });
    if (responseHistory.length > MAX_HISTORY) {
        responseHistory.pop();
    }
    renderHistory();
}

function renderHistory() {
    if (responseHistory.length === 0) {
        historyList.innerHTML = '<p class="history-empty">No responses yet</p>';
        return;
    }
    historyList.innerHTML = responseHistory.map(entry => {
        const shortInstruction = entry.instruction.length > 35
            ? entry.instruction.substring(0, 35) + '\u2026'
            : entry.instruction;
        const shortResponse = entry.response.length > 100
            ? entry.response.substring(0, 100) + '\u2026'
            : entry.response;
        return `<div class="history-item">
            <div class="history-meta">
                <span class="history-time">${entry.timestamp}</span>
                <span class="history-instruction">${shortInstruction}</span>
            </div>
            <p class="history-response">${shortResponse}</p>
        </div>`;
    }).join('');
}

// ============================================================================
// Challenge 3: Image Capture
// ============================================================================
function captureFrame() {
    if (!video.videoWidth) {
        alert('No video feed available. Start the camera first.');
        return;
    }
    const captureCanvas = document.createElement('canvas');
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    captureCanvas.getContext('2d').drawImage(video, 0, 0);

    const timestamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
    const lastResponse = responseHistory[0];
    const slug = lastResponse
        ? lastResponse.response.slice(0, 25).replace(/[^a-zA-Z0-9\s]/g, '').trim().replace(/\s+/g, '-')
        : 'frame';

    const link = document.createElement('a');
    link.download = `capture-${timestamp}-${slug}.png`;
    link.href = captureCanvas.toDataURL('image/png');
    link.click();
}

// ============================================================================
// Challenge 4: Confidence Score
// ============================================================================
function extractConfidence(text) {
    const match = text.match(/confidence:\s*(\d+)\s*\/\s*10/i);
    if (!match) return null;
    return Math.min(10, Math.max(1, parseInt(match[1], 10)));
}

function displayConfidence(score) {
    confidenceSection.style.display = 'block';
    if (score === null) {
        confidenceScoreEl.textContent = 'N/A';
        confidenceBar.style.width = '0%';
        confidenceBar.style.background = 'var(--secondary-color)';
        return;
    }
    confidenceScoreEl.textContent = score + '/10';
    confidenceBar.style.width = (score * 10) + '%';
    if (score <= 3) {
        confidenceBar.style.background = 'var(--danger-color)';
    } else if (score <= 6) {
        confidenceBar.style.background = '#f59e0b';
    } else {
        confidenceBar.style.background = 'var(--success-color)';
    }
}

// ============================================================================
// Event Listeners
// ============================================================================
startBtn.addEventListener('click', startProcessing);
stopBtn.addEventListener('click', stopProcessing);
captureBtn.addEventListener('click', captureFrame);
clearHistoryBtn.addEventListener('click', () => {
    responseHistory.length = 0;
    renderHistory();
});

// Cleanup on page unload
window.addEventListener('beforeunload', cleanup);

// ============================================================================
// TODO 5: Initialize Everything (EASY DIFFICULTY)
// ============================================================================
/**
 * Start the application
 *
 * INSTRUCTIONS:
 * 1. Call initializeModel() first (it loads the AI)
 * 2. Call initializeCamera() second (it sets up video)
 * 3. Both are async functions, so use await
 *
 * HINT: Order matters! Load model before camera
 * EXPECTED RESULT: Page loads with model ready and camera preview showing
 */

// TODO 5: Call the initialization functions in the correct order
(async () => {
    console.log('Starting Smart Vision Camera...');

    // TODO 5a: Initialize model first
    await initializeModel();

    // TODO 5b: Initialize camera second
    await initializeCamera();

    console.log('Initialization complete!');
})();

// ============================================================================
// BONUS CHALLENGES (Optional - for advanced students)
// ============================================================================

/**
 * BONUS 1: Add preset instruction buttons
 * - Create buttons for common questions
 * - "What colors do you see?"
 * - "How many objects are visible?"
 * - "Describe the lighting"
 *
 * BONUS 2: Add response history
 * - Save last 5 responses
 * - Display in a sidebar
 * - Allow user to clear history
 *
 * BONUS 3: Add image capture
 * - Button to save current frame
 * - Download as PNG with AI description
 *
 * BONUS 4: Add confidence visualization
 * - Modify prompt to ask for confidence score
 * - Display as progress bar or percentage
 */

console.log('🎓 Smart Vision Camera loaded! Complete the TODOs to make it work.');
console.log('📝 Look for TODO comments in the code to get started.');
