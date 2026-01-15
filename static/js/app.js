// JavaScript for CLIP Scene Classifier Web Interface

document.addEventListener('DOMContentLoaded', function () {
    // DOM Elements
    const uploadSection = document.getElementById('uploadSection');
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const loadingSection = document.getElementById('loadingSection');
    const resultsSection = document.getElementById('resultsSection');
    const errorSection = document.getElementById('errorSection');
    const errorMessage = document.getElementById('errorMessage');
    const previewImage = document.getElementById('previewImage');
    const newImageBtn = document.getElementById('newImageBtn');
    const retryBtn = document.getElementById('retryBtn');

    // Click to upload
    uploadBtn.addEventListener('click', (e) => {
        e.preventDefault();
        fileInput.click();
    });

    uploadArea.addEventListener('click', (e) => {
        // Only trigger if clicking the area itself, not the button
        if (e.target === uploadArea || e.target.closest('.upload-icon, h2, .upload-hint')) {
            e.preventDefault();
            fileInput.click();
        }
    });

    // File selection
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // New image button
    newImageBtn.addEventListener('click', resetUI);
    retryBtn.addEventListener('click', resetUI);

    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.add('drag-over');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('drag-over');
    }

    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('drag-over');

        const files = Array.from(e.dataTransfer.files).filter(file => file.type.startsWith('image/'));
        if (files.length > 0) {
            handleFiles(files);
        }
    }

    function handleFileSelect(e) {
        const files = Array.from(e.target.files);
        if (files.length > 0) {
            handleFiles(files);
        }
        // Reset input value to allow re-uploading the same file
        e.target.value = '';
    }

    function handleFiles(files) {
        // Validate all files
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp', 'image/tiff'];
        const validFiles = [];

        for (const file of files) {
            if (!validTypes.includes(file.type)) {
                showError(`Invalid file type for "${file.name}". Please upload image files only.`);
                return;
            }
            if (file.size > 16 * 1024 * 1024) {
                showError(`File "${file.name}" is too large. Maximum size is 16MB.`);
                return;
            }
            validFiles.push(file);
        }

        // If single image, use single classification
        if (validFiles.length === 1) {
            uploadAndClassify(validFiles[0]);
        } else {
            // Batch classification for multiple images
            uploadAndClassifyBatch(validFiles);
        }
    }

    function uploadAndClassify(file) {
        // Show loading state
        uploadSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        errorSection.classList.add('hidden');

        // Create form data
        const formData = new FormData();
        formData.append('image', file);

        // Send to backend
        fetch('/api/classify', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    displayResults(data);
                } else {
                    showError(data.error || 'Classification failed. Please try again.');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showError('Network error. Please check your connection and try again.');
            });
    }

    function uploadAndClassifyBatch(files) {
        // Show loading state
        uploadSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        errorSection.classList.add('hidden');

        // Update loading text
        loadingSection.querySelector('p').textContent = `Analyzing ${files.length} images...`;

        // Create form data
        const formData = new FormData();
        files.forEach(file => {
            formData.append('images', file);
        });

        // Send to backend
        fetch('/api/classify-batch', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    displayBatchResults(data);
                } else {
                    showError(data.error || 'Batch classification failed. Please try again.');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showError('Network error. Please check your connection and try again.');
            });
    }

    function displayResults(data) {
        // Hide loading, show results
        loadingSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        // Display image
        previewImage.src = data.image;

        // Display top prediction
        const topPred = data.top_prediction;
        document.getElementById('topType').textContent = formatSceneType(topPred.type);
        document.getElementById('topDescription').textContent = topPred.description;
        document.getElementById('topConfidence').textContent = `${topPred.confidence.toFixed(1)}%`;

        // Display recommended matching model
        const modelRec = topPred.recommended_model;
        const modelHtml = `
            <div class="model-recommendation">
                <h4>🎯 Recommended Matching Model</h4>
                <div class="model-card">
                    <div class="model-header">
                        <span class="model-name">${modelRec.name}</span>
                        <span class="model-speed speed-${modelRec.speed}">${modelRec.speed}</span>
                    </div>
                    <div class="model-reason">${modelRec.reason}</div>
                </div>
            </div>
        `;

        // Display all predictions
        const allPredictionsContainer = document.getElementById('allPredictions');
        allPredictionsContainer.innerHTML = '';

        data.all_predictions.forEach(pred => {
            const predItem = createPredictionItem(pred);
            allPredictionsContainer.appendChild(predItem);
        });

        // Insert model recommendation after scene types
        allPredictionsContainer.insertAdjacentHTML('afterend', modelHtml);
    }

    function createPredictionItem(pred) {
        const item = document.createElement('div');
        item.className = 'prediction-item';

        item.innerHTML = `
            <div class="prediction-header">
                <span class="prediction-name">${formatSceneType(pred.type)}</span>
                <span class="prediction-percentage">${pred.confidence.toFixed(1)}%</span>
            </div>
            <div class="prediction-desc">${pred.description}</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${pred.confidence}%"></div>
            </div>
        `;

        return item;
    }

    function displayBatchResults(data) {
        // Hide loading, show results
        loadingSection.classList.add('hidden');  // Fixed: was remove('hidden')
        resultsSection.classList.add('hidden');  // Keep single results hidden

        // Create or get batch results container
        let batchContainer = document.getElementById('batchResults');
        if (!batchContainer) {
            batchContainer = document.createElement('div');
            batchContainer.id = 'batchResults';
            batchContainer.className = 'batch-results';
            resultsSection.parentNode.insertBefore(batchContainer, resultsSection);
        }

        batchContainer.classList.remove('hidden');
        batchContainer.innerHTML = '';

        // Dataset summary
        const summary = data.summary;
        const summaryHTML = `
            <div class="dataset-summary">
                <h2>📊 Dataset Analysis</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">${summary.total_images}</div>
                        <div class="stat-label">Total Images</div>
                    </div>
                </div>

                <h3>Scene Distribution</h3>
                <div class="distribution-chart">
                    ${Object.entries(summary.scene_distribution)
                .sort((a, b) => b[1] - a[1])
                .map(([scene, pct]) => `
                            <div class="scene-bar">
                                <span class="scene-label">${formatSceneType(scene)}</span>
                                <div class="bar">
                                    <div class="bar-fill" style="width: ${pct}%"></div>
                                </div>
                                <span class="pct">${pct.toFixed(1)}%</span>
                            </div>
                        `).join('')}
                </div>

                <h3>🎯 Recommended Matching Models</h3>
                <div class="models-list">
                    ${summary.recommended_models.map(rec => `
                        <div class="model-recommendation">
                            <div class="model-card">
                                <div class="model-header">
                                    <span class="model-name">${rec.model.name}</span>
                                    <span class="model-speed speed-${rec.model.speed}">${rec.model.speed}</span>
                                </div>
                                <div class="model-reason">
                                    For ${formatSceneType(rec.scene_type)} scenes (${rec.percentage}% of dataset): ${rec.model.reason}
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>

            <div class="images-grid-section">
                <h3>Individual Image Results (${data.results.length})</h3>
                <div class="images-grid">
                    ${data.results.map(img => `
                        <div class="grid-item" title="${img.filename}">
                            <img src="${img.image}" alt="${img.filename}">
                            <div class="grid-label">
                                <strong>${formatSceneType(img.scene_type)}</strong>
                                <span>${img.confidence.toFixed(1)}%</span>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>

            <div class="actions">
                <button class="btn-primary" id="newBatchBtn">Analyze More Images</button>
            </div>
        `;

        batchContainer.innerHTML = summaryHTML;

        // Add event listener for new batch button
        document.getElementById('newBatchBtn').addEventListener('click', resetUI);
    }

    function formatSceneType(type) {
        // Convert "indoor_small" to "Indoor Small"
        return type
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    function showError(message) {
        uploadSection.classList.add('hidden');
        loadingSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        errorSection.classList.remove('hidden');

        errorMessage.textContent = message;
    }

    function resetUI() {
        uploadSection.classList.remove('hidden');
        loadingSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        errorSection.classList.add('hidden');

        // Reset file input
        fileInput.value = '';
    }
});
