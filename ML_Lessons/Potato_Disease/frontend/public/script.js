// Get DOM elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const browseBtn = document.getElementById('browse-btn');
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview-image');
const results = document.getElementById('results');
const loading = document.getElementById('loading');
const diseaseClass = document.getElementById('disease-class');
const confidence = document.getElementById('confidence');

// Browse button functionality
browseBtn.addEventListener('click', () => {
    fileInput.click();
});

// Handle file selection from browse button
fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

// Drag & Drop Events
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// Handle the uploaded file
function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
    }

    // Show image preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.style.display = 'block';
    };
    reader.readAsDataURL(file);

    // Upload and classify
    uploadAndClassify(file);
}

// Upload file and get prediction
async function uploadAndClassify(file) {
    loading.style.display = 'block';
    results.style.display = 'none';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        // Display results
        diseaseClass.textContent = data.class;
        confidence.textContent = (data.confidence * 100).toFixed(2);
        
        loading.style.display = 'none';
        results.style.display = 'block';

    } catch (error) {
        console.error('Error:', error);
        alert('Error classifying image. Please try again.');
        loading.style.display = 'none';
    }
}