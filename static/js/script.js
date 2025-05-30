/**
 * Global variable to store the prediction chart instance
 */
let predictionChart;

/**
 * Fetches predictions from the server and updates the UI
 * Retrieves prediction data, updates the chart and displays prediction history
 */
async function fetchPredictions() {
    const response = await fetch('/predictions');
    const predictions = await response.json();
    console.log(predictions);
    const predictionsDiv = document.getElementById('predictions');
    predictionsDiv.innerHTML = '';

    // Prepare data for chart
    const labels = Object.keys(predictions);

    // Extract prediction, loss, and percentage_loss values
    const predictionData = Object.values(predictions).map(p => p.prediction);
    const lossData = Object.values(predictions).map(p => p.loss);
    const percentageLossData = Object.values(predictions).map(p => p.percentage_loss);

    // Update or create chart
    updateChart(labels, predictionData, lossData, percentageLossData);

    if (Object.keys(predictions).length > 0) {
        const lastId = Object.keys(predictions).pop();
        const lastPrediction = predictions[lastId];
        const lastPredictionContainer = document.getElementById('last-prediction-container');  // Get the container
        lastPredictionContainer.innerHTML = ''; // Clear existing content

        // Get expected shrinkage from input
        const expectedShrinkage = parseFloat(document.getElementById('expectedShrinkageInput').value) || 0;
        const currentShrinkage = lastPrediction.percentage_loss;
        
        // Check if current shrinkage exceeds expected threshold
        const isProductReady = currentShrinkage >= expectedShrinkage;
        const cardClass = isProductReady ? 'card prediction-item border-success' : 'card prediction-item';
        const footerClass = isProductReady ? 'justify-content-center card-footer text-center bg-success text-white' : 'justify-content-center card-footer text-center bg-dark text-white';
        const readyMessage = isProductReady ? '<div class="alert alert-success text-center mt-2 mb-2"><strong>PRODUCTO LISTO</strong></div>' : '';

        const lastPredictionItem = document.createElement('div');
        lastPredictionItem.className = 'last-prediction-item';
        lastPredictionItem.innerHTML = `
        <div class="${cardClass}">
            <h4 class="card-title text-center prediction-item" style="font-size: 1rem;"><strong>Última Predicción</strong></h4>
            ${readyMessage}
            <div class="card-body prediction-item">
                <div class="text-center">  <!-- Center the text content -->
                    <p class="card-text">
                        <strong>Fecha:</strong> ${lastId}<br>
                        <strong>Imagen:</strong> ${lastPrediction.name}<br>
                        <strong>Carpeta:</strong> ${lastPrediction.folder}<br>
                        <strong>Días transcurridos:</strong> ${lastPrediction.days}<br>
                        ${lastPrediction.initial_weight !== undefined && lastPrediction.initial_weight !== null ? `<strong>Peso inicial:</strong> ${lastPrediction.initial_weight}g<br>` : ''}
                    </p>
                </div>
                <img src="/images/${lastPrediction.folder}/${lastPrediction.name}" class="img-fluid p-4">
            </div>
            <div class="${footerClass}">
                <strong class="pr-2">Predicción:</strong> ${lastPrediction.prediction.toFixed(0)} gramos
                <br>
                <strong class="pl-2">Merma:</strong> ${lastPrediction.percentage_loss.toFixed(1)}%
                <br>
                <strong class="pl-2">Pérdida Peso:</strong> ${lastPrediction.loss.toFixed(0)}
            </div>
        </div>
        `;
        lastPredictionContainer.appendChild(lastPredictionItem);

         // Iterate in reverse to show latest predictions first
        for (const [id, prediction] of Object.entries(predictions).slice(0, -1).reverse()) {
            const p = document.createElement('div');
            p.className = 'prediction-item';
            p.innerHTML = `
                <div class="card-body">
                    <h5 class="card-title">Predicción del  ${id}</h5>
                    <p class="card-text">
                        <strong>Imagen:</strong> ${prediction.name}<br>
                        <strong>Días transcurridos:</strong> ${prediction.days}<br>
                        ${prediction.initial_weight !== undefined && prediction.initial_weight !== null ? `<strong>Peso inicial:</strong> ${prediction.initial_weight}g<br>` : ''}
                        <strong>Predicción:</strong> ${prediction.prediction.toFixed(0)} gramos
                        <strong>Merma:</strong> ${prediction.percentage_loss.toFixed(1)}%
                        <strong>Pérdida Peso:</strong> ${prediction.loss.toFixed(0)}
                    </p>
                </div>`;
            predictionsDiv.appendChild(p);
        }
    }
}

/**
 * Resets all predictions by calling the reset endpoint
 */
async function resetPredictions() {
    const response = await fetch('/reset', {
        method: 'POST'
    });
    if (response.ok) {
        fetchPredictions();
        // Reset the model status
        updateModelStatus('', false);
        // Reset the model selector
        document.getElementById('modelSelect').value = '';
    }
}

/**
 * Set the initial weight on the server
 */
async function setInitialWeight() {
    const weight = document.getElementById('initialWeightInput').value;
    const weightNum = parseFloat(weight);
    
    // Initial weight validation range
    if (!weight || isNaN(weightNum) || weightNum < 100 || weightNum > 10000) {
        alert('Por favor ingrese un peso inicial válido entre 100g y 10000g (10kg).\n\nEjemplos válidos:\n- 2500g (2.5kg)\n- 1000g (1kg)\n- 5000g (5kg)');
        return;
    }
    
    try {
        const response = await fetch('/set_initial_weight', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ initial_weight: weightNum })
        });
        
        const data = await response.json();
        if (response.ok) {
            alert('Peso inicial establecido correctamente: ' + weight + 'g (' + (weightNum/1000).toFixed(2) + 'kg)');
        } else {
            alert('Error: ' + data.message);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error al establecer el peso inicial');
    }
}

/**
 * Set the expected shrinkage percentage
 */
async function setExpectedShrinkage() {
    const shrinkage = document.getElementById('expectedShrinkageInput').value;
    const shrinkageNum = parseFloat(shrinkage);
    
    // Expected shrinkage validation range
    if (!shrinkage || isNaN(shrinkageNum) || shrinkageNum < 0 || shrinkageNum > 100) {
        alert('Por favor ingrese una merma esperada válida entre 0% y 100%.\n\nEjemplos válidos:\n- 15.0%\n- 12.5%\n- 20.0%');
        return;
    }
    
    try {
        alert('Merma esperada establecida correctamente: ' + shrinkage + '%');
        fetchPredictions(); // Refresh predictions to apply new threshold
    } catch (error) {
        console.error('Error:', error);
        alert('Error al establecer la merma esperada');
    }
}

/**
 * Loads a selected model by sending a request to the server
 */
async function loadModel() {
    const modelName = document.getElementById('modelSelect').value;
    
    // Validate that a model has been selected
    if (!modelName || modelName === '') {
        alert('Por favor selecciona un modelo antes de cargar.');
        return;
    }
    
    const formData = new FormData();
    formData.append('model_name', modelName);

    try {
        const response = await fetch('/load_model', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            // If the model requires days elapsed, ask the user
            if (modelName.includes('SecaderoDaysFeature') || modelName.includes('SecaderoDaysAndWeightFeature')) {
                let dias = prompt('Ingrese los días transcurridos iniciales:', '0');
                dias = dias === null ? '0' : dias;
                await fetch('/set_initial_days', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ days: dias })
                });
            }
            // Update model status indicator
            updateModelStatus(modelName, true);
            alert(data.message);
            fetchPredictions(); // Just reload predictions
        } else {
            alert(`Error: ${data.message}`);
        }
    } catch (error) {
        console.error('Network error:', error);
        alert('An error occurred while loading the model.');
    }
}

/**
 * Updates the chart with new prediction data
 * @param {Array} labels - Timestamps for x-axis
 * @param {Array} predictionData - Prediction values
 * @param {Array} lossData - Weight loss values
 * @param {Array} percentageLossData - Percentage loss values
 */
function updateChart(labels, predictionData, lossData, percentageLossData) {
    const ctx = document.getElementById('predictionChart').getContext('2d');
    if (predictionChart) {
        predictionChart.destroy(); // Destroy existing chart
    }

    predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Predicciones (gramos)',
                    data: predictionData,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.4,
                    yAxisID: 'y-prediction' // Assign the y-axis
                },
                {
                    label: 'Pérdida de Peso',
                    data: lossData,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.4,
                    yAxisID: 'y-loss' // Assign the y-axis
                },
                {
                    label: 'Merma (%)',
                    data: percentageLossData,
                    borderColor: 'rgba(255, 205, 86, 1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.4,
                    yAxisID: 'y-percentage' // Assign the y-axis
                }
            ]
        },
        options: {
            animation: false,  // Disable animation
            scales: {
                'y-prediction': {
                    type: 'linear',
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Predicciones (gramos)',
                        color: '#e9e9e9'
                    },
                    ticks: {
                        color: '#e9e9e9'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'  // Change grid color
                    }
                },
                'y-loss': {
                    type: 'linear',
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Pérdida de Peso',
                        color: '#e9e9e9'
                    },
                    ticks: {
                        color: '#e9e9e9'
                    },
                    grid: {
                        drawOnChartArea: false, // prevent grid lines from overlapping
                        color: 'rgba(255, 255, 255, 0.1)'  // Change grid color
                    }
                },
                'y-percentage': {
                    type: 'linear',
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Merma (%)',
                        color: '#e9e9e9'
                    },
                    ticks: {
                        color: '#e9e9e9'
                    },
                    grid: {
                        drawOnChartArea: false, // prevent grid lines from overlapping
                        color: 'rgba(255, 255, 255, 0.1)'  // Change grid color
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Timestamp',
                        color: '#e9e9e9'
                    },
                    ticks: {
                        color: '#e9e9e9'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'  // Change grid color
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#e9e9e9'
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';

                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += context.parsed.y;
                                if (context.dataset.label === 'Predicciones (gramos)') {
                                    label += " gramos";
                                } else if (context.dataset.label === 'Merma (%)') {
                                    label += "%";
                                }
                            }
                            return label;
                        }
                    }
                },
                datalabels: {
                    display: false
                }
            }
        }
    });
}

/**
 * Starts periodic fetching of predictions
 * Initializes the first fetch and sets up an interval
 */
function startFetching() {
    fetchPredictions();
    setInterval(fetchPredictions, 5000); // Fetch predictions every 5 seconds
}

window.onload = startFetching;

/**
 * Update the model status indicator
 */
function updateModelStatus(modelName, isLoaded) {
    const statusDiv = document.getElementById('modelStatus');
    if (isLoaded) {
        statusDiv.className = 'alert alert-success text-center';
        statusDiv.innerHTML = `<strong>✅ Modelo cargado:</strong> ${modelName}`;
    } else {
        statusDiv.className = 'alert alert-warning text-center';
        statusDiv.innerHTML = '<strong>⚠️ No hay modelo cargado</strong> - Por favor selecciona y carga un modelo para comenzar el monitoreo';
    }
}

/**
 * Send the image and extra parameters to the backend for inference
 */
async function predictImage(imageFile) {
    const model = document.getElementById('modelSelect').value;
    const formData = new FormData();
    formData.append('image', imageFile);
    
    // Always add the initial weight from the UI field
    const weight = document.getElementById('initialWeightInput').value;
    if (weight && weight > 0) {
        formData.append('initial_weight', weight);
    }
    
    try {
        const response = await fetch('/predict_image', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (response.ok) {
            alert('Predicción: ' + data.prediction + ' gramos');
            fetchPredictions();
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error de red: ' + error);
    }
}