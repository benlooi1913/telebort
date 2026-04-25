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

// ============================================================================
// Global State
// ============================================================================
let model = null;
let stream = null;
let isProcessing = false;
let processingInterval = null;
const PROCESSING_DELAY = 3000; // Process every 3 seconds

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
        const modelId = "HuggingFaceTB/SmolVLM-256M-Instruct";

        console.log('Loading model:', modelId);
        model = await pipeline("image-to-text", modelId, {
            device: "webgpu", // Use GPU acceleration
        });

        console.log('Model loaded successfully!');
        updateStatus('Ready to start', true);
        startBtn.disabled = false;
        loadingOverlay.classList.add('hidden');

    } catch (error) {
        console.error('Error loading model:', error);
        updateStatus('Error loading model: ' + error.message, false);
        alert('Failed to load AI model. Check console for details.');
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

                // TODO 3d: Send to AI model
                // HINT: model expects (image, {prompt: "your instruction"})
                const result = await model(blob, {
                    prompt: instruction,
                    max_new_tokens: 100, // TODO: Set to 100 for reasonable response length
                });

                // Display response
                displayResponse(result[0].generated_text);

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
// Event Listeners
// ============================================================================
startBtn.addEventListener('click', startProcessing);
stopBtn.addEventListener('click', stopProcessing);

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
