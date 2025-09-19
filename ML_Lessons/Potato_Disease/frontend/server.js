const express = require('express');
const multer = require('multer');
const axios = require('axios');
const path = require('path');
const fs = require('fs');

const app = express();
const port = 3000;

// Configure multer for file uploads
const upload = multer({ dest: 'uploads/' });

// Serve static files from public directory
app.use(express.static('public'));

// Handle file upload and prediction
// Handle file upload and prediction
app.post('/predict', upload.single('file'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file uploaded' });
        }

        // Create form data to send to Python API
        const FormData = require('form-data');
        const formData = new FormData();
        
        // Read the uploaded file and append to form data
        const fileStream = fs.createReadStream(req.file.path);
        formData.append('file', fileStream, req.file.originalname);

        // Send to Python API
        const response = await axios.post('http://localhost:8000/predict', formData, {
            headers: {
                ...formData.getHeaders()
            }
        });

        // Clean up uploaded file
        fs.unlinkSync(req.file.path);

        // Return prediction results
        res.json(response.data);

    } catch (error) {
        console.error('Error:', error.message);
        res.status(500).json({ error: 'Prediction failed' });
        
        // Clean up file if it exists
        if (req.file) {
            fs.unlinkSync(req.file.path);
        }
    }
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});