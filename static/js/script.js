let predictionChart;

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

        const lastPredictionItem = document.createElement('div');
        lastPredictionItem.className = 'last-prediction-item';
        lastPredictionItem.innerHTML = `
        <div class="card prediction-item">
            <h4 class="card-title text-center prediction-item" style="font-size: 1rem;"><strong>Last Prediction</strong></h4>
            <div class="card-body prediction-item">
                <div class="text-center">  <!-- Center the text content -->
                    <p class="card-text">
                        <strong>Fecha:</strong> ${lastId}<br>
                        <strong>Imagen:</strong> ${lastPrediction.name}
                    </p>
                </div>
                <img src="/images/${lastPrediction.name}" class="img-fluid p-4">
            </div>
            <div class="justify-content-center card-footer text-center bg-dark text-white">
                <strong class="pr-2">Predicción:</strong> ${lastPrediction.prediction} gramos
                <br>
                <strong class="pl-2">Merma:</strong> ${lastPrediction.percentage_loss.toFixed(1)}%
                <br>
                <strong class="pl-2">Pérdida Peso:</strong> ${lastPrediction.loss}
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
                        <strong>Imagen:</strong> ${prediction.name}
                        <br>
                        <strong>Predicción:</strong> ${prediction.prediction} gramos
                        <strong>Merma:</strong> ${prediction.percentage_loss}%
                        <strong>Pérdida Peso:</strong> ${prediction.loss}
                    </p>
                </div>`;
            predictionsDiv.appendChild(p);
        }
    }
}

async function resetPredictions() {
    const response = await fetch('/reset', {
        method: 'POST'
    });
    if (response.ok) {
        fetchPredictions();
    }
}

async function loadModel() {
    const modelName = document.getElementById('modelSelect').value;
    const formData = new FormData();
    formData.append('model_name', modelName);

    try {
        const response = await fetch('/load_model', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
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

function startFetching() {
    fetchPredictions();
    setInterval(fetchPredictions, 5000); // Fetch predictions every 5 seconds
}

window.onload = startFetching;