# Smart Vision Camera - Discovery Challenge

## 🎯 Learning Objectives
- Understand how AI vision models work in web browsers
- Learn to integrate Hugging Face Transformers.js library
- Practice JavaScript programming with AI APIs
- Experience prompt engineering for AI vision tasks
- Build a real-time AI application

## 🚀 Getting Started (See Results in 30 Seconds!)

### Prerequisites
- Modern web browser with WebGPU support (Chrome 113+, Safari Technology Preview, or Edge 113+)
- Working webcam
- Local web server (Live Server extension in VS Code, or Python's `http.server`)

### Quick Start
```bash
# If using VS Code:
# 1. Open this folder in VS Code
# 2. Install Live Server extension if not installed
# 3. Right-click on index.html → "Open with Live Server"

# If using Python:
cd project-01-huggingface
python3 -m http.server 8000
# Then open: http://localhost:8000
```

### 🎯 What's Already Working
- ✅ HTML structure with camera preview (fully working)
- ✅ CSS styling with professional UI (fully working)
- ✅ Camera permission handling (fully working)
- ✅ Basic page layout and buttons (fully working)
- ⚠️ **TODO**: AI model loading (70% complete - needs your work!)
- ⚠️ **TODO**: Image processing logic (60% complete - needs your work!)
- ⚠️ **TODO**: Prompt engineering customization (30% complete - needs your work!)

## 📋 Tasks to Complete

### TODO 1: Set Up AI Model (Medium)
**Location**: `script.js` - Line ~45
**Success Criteria:**
- [ ] Model ID is correctly specified
- [ ] Model loads successfully (check browser console)
- [ ] "Ready to start" message appears
- [ ] No errors in console during model loading

**Hint**: Look for `// TODO 1` in script.js

### TODO 2: Complete Camera Setup (Easy)
**Location**: `script.js` - Line ~78
**Success Criteria:**
- [ ] Browser asks for camera permission
- [ ] Camera feed displays in preview
- [ ] Video element shows live camera
- [ ] No "camera not found" errors

**Hint**: You need to call the getUserMedia API

### TODO 3: Implement Image Processing (Hard)
**Location**: `script.js` - Line ~120
**Success Criteria:**
- [ ] Image is captured from video every 3 seconds
- [ ] Image is sent to AI model for analysis
- [ ] Response appears in the output area
- [ ] Processing indicator shows during analysis

**Hint**: Look for the `processFrame()` function

### TODO 4: Customize AI Prompts (Medium)
**Location**: `script.js` - Line ~150
**Success Criteria:**
- [ ] Default instruction is meaningful
- [ ] User can change instruction
- [ ] AI responds based on instruction
- [ ] Responses are relevant to the image

**Current Challenge**: The default prompt is generic. Experiment with different prompts!

## 🚀 Extension Challenges

### Challenge 1: Multi-Prompt Buttons (Easy)
Add preset instruction buttons for common questions:
- "What colors do you see?"
- "How many objects are visible?"
- "Describe the lighting"

### Challenge 2: Response History (Medium)
Save the last 10 AI responses and display them in a sidebar

### Challenge 3: Image Capture (Advanced)
Add a button to capture and save the current frame with AI description

### Challenge 4: Confidence Score (Advanced)
Modify prompt to ask AI for confidence level (1-10) and display it visually

## 🛠️ Troubleshooting

### Issue: "WebGPU not available"
**Solution**: Update to Chrome 113+ or enable WebGPU in browser flags

### Issue: Camera permission denied
**Solution**:
1. Click the camera icon in browser address bar
2. Allow camera access
3. Refresh the page

### Issue: Model loading takes forever
**Solution**:
1. Be patient! First load can take 2-5 minutes
2. Check internet connection
3. Try the smaller model variant (256M instead of 500M)

### Issue: No AI response
**Solution**:
1. Open browser console (F12) and check for errors
2. Ensure model has finished loading
3. Verify camera is showing video

## 📚 Technologies Used
- **HTML5**: Semantic markup, video element
- **CSS3**: Modern UI, animations, responsive design
- **JavaScript ES6+**: Async/await, modules
- **Transformers.js**: Hugging Face ML library for browser
- **WebGPU**: GPU-accelerated AI inference
- **MediaDevices API**: Camera access

## 🤖 Ethical AI Considerations
- **Privacy**: All processing happens in your browser - no data sent to servers
- **Transparency**: Users should know they're using AI
- **Limitations**: AI may misidentify objects or make mistakes
- **Accessibility**: Consider adding screen reader support
- **Appropriate Use**: Don't use for surveillance without consent

## 🎓 Learning Resources
- [Transformers.js Documentation](https://huggingface.co/docs/transformers.js)
- [WebGPU Guide](https://webgpu.io)
- [MediaDevices API](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices)
- [Prompt Engineering Tips](https://www.promptingguide.ai/)

## 💡 Tips for Success
1. **Test incrementally**: After completing each TODO, test immediately
2. **Read error messages**: Browser console provides valuable debugging info
3. **Ask for help**: Use AI assistants (Claude, ChatGPT) to explain errors
4. **Experiment**: Try different models, prompts, and processing intervals
5. **Document**: Add comments explaining your changes

Good luck! Remember: debugging is part of learning. Every error is an opportunity to understand the system better! 🚀
