const form = document.getElementById('prediction-form');
const predictionElement = document.getElementById('prediction');

form.addEventListener('submit', (event) => {
  event.preventDefault();

  const data = {
    tenure: form.elements.tenure.value,
    contract: form.elements.contract.value,
    total_charges: form.elements.total_charges.value,
  };

  fetch('http://127.0.0.1:5000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  .then(response => response.json())
  .then(data => {
    predictionElement.textContent = `Predicted Churn: ${data.churn_prediction}`;
  })
  .catch(error => console.error(error));
});
